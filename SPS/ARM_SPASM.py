import numpy as np
import numpy.typing as npt
import copy
import os
import time
import datetime
import psutil
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

from py_src.star.python.render import *
from SPS.SPS_filter import *


# GLOBALS

n: int = 15
alpha: float = 1.0
beta: float = 2.0
kappa: float = 0.0
lam: float = alpha ** 2 * (n + kappa) - n
# lam: float = 0.0

I4 = np.identity(4)


def normalizeSigma(sigma: npt.NDArray) -> npt.NDArray:
    q = ModifiedRodriguesParameters.FromVector(sigma).ToQuat()
    return ModifiedRodriguesParameters.FromQuat(q.normalize()).ToVector()


def propagateQuat(q: Quaternion, omega: npt.NDArray, dt: float) -> Quaternion:
    omegaNorm = np.linalg.norm(omega)

    # If angvel is zero or elapsed time is zero, early out.
    if omegaNorm == 0.0 or dt == 0.0:
        return q

    qk = q.as_w_first_array()
    
    # Effective way to reverse propagation instead of taking conjugate quaternions, etc
    dt *= -1

    wx = omega[0]
    wy = omega[1]
    wz = omega[2]

    Omega4 = np.array([[0.0, -wx, -wy, -wz],
                       [wx, 0.0, wz, -wy],
                       [wy, -wz, 0.0, wx],
                       [wz, wy, -wx, 0.0]])
    qk1 = (np.cos(0.5 * omegaNorm * dt) * I4 +
        (1.0 / omegaNorm) * np.sin(0.5 * omegaNorm * dt) * Omega4) @ np.array([qk]).T
    
    return Quaternion(qk1[0, 0], qk1[1, 0], qk1[2, 0], qk1[3, 0]).normalize()


def propagateMRP(MRP: npt.NDArray, omega: npt.NDArray, dt: float):
    s1 = MRP[0]
    s2 = MRP[1]
    s3 = MRP[2]
    sSquared = s1 ** 2 + s2 ** 2 + s3 ** 2
    B = np.array([[1.0 - sSquared + 2.0 * s1 ** 2, 2.0 * (s1 * s2 - s3), 2.0 * (s1 * s3 + s2)],
                  [2.0 * (s1 * s2 + s3), 1.0 - sSquared + 2.0 * s2 ** 2, 2.0 * (s2 * s3 - s1)],
                  [2.0 * (s1 * s3 - s2), 2.0 * (s2 * s3 + s1), 1.0 - sSquared + 2.0 * s3 ** 2]])
    return 0.25 * (B @ np.array([omega]).T).T[0]


def xNext(dt: float, x: npt.NDArray, mu: float):
    # rb = x[0:3]
    r = x[3:6]
    v = x[6:9]
    sigma = x[9:12]
    omega = x[12:15]

    sigma_MRP = ModifiedRodriguesParameters.FromVector(sigma)
    # if sigma_MRP.sigmaSquared >= 1.0:
    #     sigma_MRP = sigma_MRP.shadow()
    # sigma = sigma_MRP.ToVector()

    dt_substeps = 10
    dt_substep = dt / float(dt_substeps)

    r_next = copy.deepcopy(r)
    v_next = copy.deepcopy(v)

    for _ in range(dt_substeps):  # Symplectic Euler??
        v_next -= dt_substep * mu * r_next / np.linalg.norm(r_next) ** 3
        r_next += dt_substep * v_next

    # q_next = propagateQuat(sigma_MRP.ToQuat(), omega, dt)
    # sigma_next_MRP = ModifiedRodriguesParameters.FromQuat(q_next)

    sigma_next_MRP = ModifiedRodriguesParameters.FromVector(propagateMRP(sigma, omega, dt))
    if sigma_next_MRP.sigmaSquared >= 1.0:
        sigma_next_MRP = sigma_next_MRP.shadow()
    sigma_next = sigma_next_MRP.ToVector()
    
    omega_next = copy.deepcopy(omega)
    # T_next = q_next.to_matrix().T
    T_next = sigma_next_MRP.ToMatrix()
    rb_next = (T_next @ np.array([r_next]).T).T[0]

    return np.array([rb_next[0], rb_next[1], rb_next[2], r_next[0], r_next[1], r_next[2], 
                     v_next[0], v_next[1], v_next[2], sigma_next[0], sigma_next[1], sigma_next[2],
                     omega_next[0], omega_next[1], omega_next[2]])


def h(x: npt.NDArray):
    # rb = x[0:3]
    r = x[3:6]
    v = x[6:9]
    sigma = x[9:12]
    omega = x[12:15]

    T = ModifiedRodriguesParameters.FromVector(sigma).ToMatrix()
    rb = (T @ np.array([r]).T).T[0]

    return np.array([rb[0], rb[1], rb[2], r[0], r[1], r[2], 
                     v[0], v[1], v[2], omega[0], omega[1], omega[2]])


def sigmaPointsProp(n: int, dt: float, mx: npt.NDArray, Sxx: npt.NDArray, gravModel: grav_base):
    sp: list[npt.NDArray] = [xNext(dt, mx, gravModel.mu)]
    wm: list[float] = [lam / (n + lam)]
    wc: list[float] = [lam / (n + lam) + 1.0 - alpha ** 2 + beta]
    for i in range(n):
        sp.append(xNext(dt, mx + np.sqrt(n + lam) * Sxx[:, i], gravModel.mu))
        sp.append(xNext(dt, mx - np.sqrt(n + lam) * Sxx[:, i], gravModel.mu))
        wm.append(1.0 / (2.0 * (n + lam)))
        wm.append(1.0 / (2.0 * (n + lam)))
        wc.append(1.0 / (2.0 * (n + lam)))
        wc.append(1.0 / (2.0 * (n + lam)))
    
    if(abs(np.sum(wm) - 1.0) > 1e-6):
        raise Exception(f"Error: sum of mean weights not equal to 1 (= {np.sum(wm)} instead)")
    # if(abs(np.sum(wc) - 1.0) > 1e-6):
    #     raise Exception(f"Error: sum of covariance weights not equal to 1 (= {np.sum(wc)} instead)")
    
    return sp, wm, wc


def sigmaPointsMeas(n: int, dt: float, mx: npt.NDArray, Sxx: npt.NDArray, gravModel: grav_base):
    sp_prior: list[npt.NDArray] = [mx]
    sp: list[npt.NDArray] = [h(mx)]
    wm: list[float] = [lam / (n + lam)]
    wc: list[float] = [lam / (n + lam) + 1.0 - alpha ** 2 + beta]
    for i in range(n):
        sp1 = mx + np.sqrt(n + lam) * Sxx[:, i]
        sp2 = mx - np.sqrt(n + lam) * Sxx[:, i]
        sp_prior.append(sp1)
        sp_prior.append(sp2)

        sp.append(h(sp1))
        sp.append(h(sp2))
        wm.append(1.0 / (2.0 * (n + lam)))
        wm.append(1.0 / (2.0 * (n + lam)))
        wc.append(1.0 / (2.0 * (n + lam)))
        wc.append(1.0 / (2.0 * (n + lam)))
    
    if(abs(np.sum(wm) - 1.0) > 1e-6):
        raise Exception(f"Error: sum of mean weights not equal to 1 (= {np.sum(wm)} instead)")
    # if(abs(np.sum(wc) - 1.0) > 1e-6):
    #     raise Exception(f"Error: sum of covariance weights not equal to 1 (= {np.sum(wc)} instead)")
    
    return sp_prior, sp, wm, wc


if __name__ == "__main__":
    np.random.seed(100)

    # Planet
    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel

    # Astrophysical parameters
    spice.furnsh("./py_src/star/data/metakernel.txt")
    tJ2000 = '2000 Jan 1, 00:00:00 UTC'
    tNow = globalConfig.tNow
    etJ2000 = spice.str2et(tJ2000)
    etNow = spice.str2et(tNow)
    etOriginal = copy.deepcopy(etNow)

    # Initial states
    planetRotState = spice.sxform("J2000", planet.planetFrame, etNow)
    planetRot, planetAngVel = spice.xf2rav(planetRotState)
    planetMRP = ModifiedRodriguesParameters.FromMatrix(planetRot.T)
    planetMRP_vec = planetMRP.ToVector()
    planetQuat = planetMRP.ToQuat()

    planetAngVelBody = planetRot.T @ planetAngVel

    print(f"planetMRP_vec = {planetMRP_vec}")
    print(f"planetQuat    = {planetQuat.as_w_first_array()}")
    print(f"planetAngVel  = {planetAngVelBody}\n")

    r_init: float = 1800.0e3
    r0 = np.array([r_init, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(gravModel.mu / r_init), 0.0])
    r0_b = (planetRot @ np.array([r0]).T).T[0]
    x_0 = np.array([r0_b[0], r0_b[1], r0_b[2], r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], 
                    planetMRP_vec[0], planetMRP_vec[1], planetMRP_vec[2],
                    planetAngVelBody[0], planetAngVelBody[1], planetAngVelBody[2]])
    Pxx_0 = np.diag(np.array([5e3, 5e3, 5e3, 5e3, 5e3, 5e3, 1e-1, 1e-1, 1e-1, 
                              1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])) ** 2
    
    Pww = np.diag(np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e-3, 1e-3, 1e-3, 
                            1e-5, 1e-5, 1e-5, 1e-7, 1e-7, 1e-7])) ** 2
    
    # Covariances
    Pvv_XNav = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3]) * 5e3 ** 2
    Pvv_SPASM = np.array([1.0, 1.0, 1.0]) * 5e2 ** 2
    Pvv_SPS = np.array([1.0, 1.0, 1.0]) * 1e-5 ** 2
    Pvv = np.diag(np.concat((Pvv_XNav, Pvv_SPASM, Pvv_SPS)))

    # Underweighting
    alpha_underweight = 10.0  # >= 1
    gamma_underweight = 0.1  # <= 1

    # Logging
    x_true: list[npt.NDArray] = [copy.deepcopy(x_0)]
    mx_est: list[npt.NDArray] = [x_0 + np.linalg.cholesky(Pxx_0) @ np.random.randn(15)]
    Pxx_est: list[npt.NDArray] = [copy.deepcopy(Pxx_0)]

    # Time
    N = 100
    times = np.linspace(etOriginal, etOriginal + 20000.0, N + 1)
    dt = times[1] - times[0]
    print(f"dt = {dt}")

    update: bool = True
    measurementCadence: int = 4
    for k in range(1, len(times)):
        print(f"Starting iteration {k}/{len(times)}")
        # etNow = times[k]

        # Propagate
        Sxx_prop = np.linalg.cholesky(Pxx_est[k-1])
        sp_prop, wm_prop, wc_prop = sigmaPointsProp(n, dt, mx_est[k-1], Sxx_prop, gravModel)
        
        mx_k_minus = wm_prop[0] * sp_prop[0]
        for i in range(1, 2 * n + 1):
            mx_k_minus += wm_prop[i] * sp_prop[i]
        
        mx_k_minus_MRP = ModifiedRodriguesParameters.FromVector(mx_k_minus[9:12])
        if mx_k_minus_MRP.sigmaSquared >= 1.0:
            mx_k_minus_MRP = mx_k_minus_MRP.shadow()
        mx_k_minus[9:12] = mx_k_minus_MRP.ToVector()
        
        Pxx_k_minus = wc_prop[0] * np.outer(sp_prop[0] - mx_k_minus, sp_prop[0] - mx_k_minus)
        for i in range(1, 2 * n + 1):
            Pxx_k_minus += wc_prop[i] * np.outer(sp_prop[i] - mx_k_minus, sp_prop[i] - mx_k_minus)
        Pxx_k_minus += Pww

        # Single sigma point propagation (testing only)
        # mx_k_minus = sp_prop[0]
        # Pxx_k_minus = np.outer(sp_prop[0] - mx_k_minus, sp_prop[0] - mx_k_minus) + Pww
        
        # Update
        if update and k % measurementCadence == 0:
            print(f"Updating with new measurements...")
            Sxx_meas = np.linalg.cholesky(Pxx_k_minus)
            sp_prior, sp_meas, wm_meas, wc_meas = sigmaPointsMeas(n, dt, mx_k_minus, Sxx_meas, gravModel)

            mz_k_minus = wm_meas[0] * sp_meas[0]
            for i in range(1, 2 * n + 1):
                mz_k_minus += wm_meas[i] * sp_meas[i]
            
            Pxz_k_minus = wc_meas[0] * np.outer(sp_prior[0] - mx_k_minus, sp_meas[0] - mz_k_minus)
            Pzz_k_minus = wc_meas[0] * np.outer(sp_meas[0] - mz_k_minus, sp_meas[0] - mz_k_minus)
            for i in range(1, 2 * n + 1):
                Pxz_k_minus = wc_meas[i] * np.outer(sp_prior[i] - mx_k_minus, sp_meas[i] - mz_k_minus)
                Pzz_k_minus = wc_meas[i] * np.outer(sp_meas[i] - mz_k_minus, sp_meas[i] - mz_k_minus)
            Pzz_k_minus += alpha_underweight * Pvv

            # Process measurement
            x_next_true = xNext(dt, x_true[k-1], gravModel.mu)
            x_next_true_MRP = ModifiedRodriguesParameters.FromVector(x_next_true[9:12])
            if x_next_true_MRP.sigmaSquared >= 1.0:
                x_next_true_MRP = x_next_true_MRP.shadow()
            x_next_true[9:12] = x_next_true_MRP.ToVector()
            
            x_true.append(x_next_true)
            z_k = h(x_true[k]) + np.linalg.cholesky(Pvv) @ np.random.randn(12)
        
            K = Pxz_k_minus @ (np.linalg.inv(Pzz_k_minus))
            mx_k_plus = mx_k_minus + gamma_underweight * K @ (z_k - mz_k_minus)
            Pxx_k_plus = Pxx_k_minus - Pxz_k_minus @ K.T - K @ Pxz_k_minus.T + K @ Pzz_k_minus @ K.T

            mx_k_plus_MRP = ModifiedRodriguesParameters.FromVector(mx_k_plus[9:12])
            if mx_k_plus_MRP.sigmaSquared >= 1.0:
                mx_k_plus_MRP = mx_k_plus_MRP.shadow()
            mx_k_plus[9:12] = mx_k_plus_MRP.ToVector()
            
            mx_est.append(mx_k_plus)
            Pxx_est.append(Pxx_k_plus)
            
            # print(f"mx_est[k+1] = {mx_k_plus}")
        else:
            x_next_true = xNext(dt, x_true[k-1], gravModel.mu)
            x_next_true_MRP = ModifiedRodriguesParameters.FromVector(x_next_true[9:12])
            if x_next_true_MRP.sigmaSquared >= 1.0:
                x_next_true_MRP = x_next_true_MRP.shadow()
            x_next_true[9:12] = x_next_true_MRP.ToVector()

            x_true.append(x_next_true)
            mx_est.append(mx_k_minus)
            Pxx_est.append(Pxx_k_minus)

    print(f"len(x_true) = {len(x_true)}")
    print(f"len(mx_est) = {len(mx_est)}")
    print(f"len(Pxx_est) = {len(Pxx_est)}\n")
    
    ########################
    ####    Plotting    ####
    ########################
    
    # Translational states
    fig1 = plt.figure(layout='constrained')
    axs1 = fig1.subplots(3, 3)

    # Rotational states
    fig2 = plt.figure(layout='constrained')
    axs2 = fig2.subplots(2, 3)

    labels_mx = [r"$x^{\mathcal{B}}$", r"$y^{\mathcal{B}}$", r"$z^{\mathcal{B}}$", 
                 r"$x^{\mathcal{I}}$", r"$y^{\mathcal{I}}$", r"$z^{\mathcal{I}}$", 
                 r"$v_{x}^{\mathcal{I}}$", r"$v_{y}^{\mathcal{I}}$", r"$v_{z}^{\mathcal{I}}$",
                 r"$\sigma_{x}$", r"$\sigma_{y}$", r"$\sigma_{z}$",
                 r"$\omega_{x}$", r"$\omega_{y}$", r"$\omega_{z}$"]

    labels_Pxx = [r"$P_{x}^{\mathcal{B}}$", r"$P_{y}^{\mathcal{B}}$", r"$P_{z}^{\mathcal{B}}$", 
                  r"$P_{x}^{\mathcal{I}}$", r"$P_{y}^{\mathcal{I}}$", r"$P_{z}^{\mathcal{I}}$", 
                  r"$P_{v_{x}}^{\mathcal{I}}$", r"$P_{v_{y}}^{\mathcal{I}}$", r"$P_{v_{z}}^{\mathcal{I}}$",
                  r"$P_{\sigma_{x}}$", r"$P_{\sigma_{y}}$", r"$P_{\sigma_{z}}$",
                  r"$P_{\omega_{x}}$", r"$P_{\omega_{y}}$", r"$P_{\omega_{z}}$"]
    units = ["km", "km", "km", "km", "km", "km", "km/s", "km/s", "km/s", 
             "unitless", "unitless", "unitless", "rad/s", "rad/s", "rad/s"]

    idx = 0
    for ax in axs1.flat:
        mx_i = 0.001 * np.array([mx[idx] for mx in mx_est])
        x_true_i = 0.001 * np.array([x_t[idx] for x_t in x_true])
        Pxx_i = 1e-6 * np.array([Pxx[idx, idx] for Pxx in Pxx_est])

        # ax.plot(times - etOriginal, x_true_i, label=labels_mx[idx], color='green')
        # ax.plot(times - etOriginal, mx_i, label=labels_mx[idx], color='blue')
        ax.plot(times - etOriginal, mx_i - x_true_i, label=labels_mx[idx], color='blue')
        ax.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_i), linestyle='dashed', color='r', label=labels_Pxx[idx])
        ax.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_i), linestyle='dashed', color='r')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\Delta$" + f"{labels_mx[idx]} ({units[idx]})")
        ax.set_title(f"Error in {labels_mx[idx]} over Time")
        ax.grid()
        ax.legend()
        idx += 1
    
    for ax in axs2.flat:
        mx_i = np.array([mx[idx] for mx in mx_est])
        x_true_i = np.array([x_t[idx] for x_t in x_true])
        Pxx_i = np.array([Pxx[idx, idx] for Pxx in Pxx_est])

        # ax.plot(times - etOriginal, x_true_i, label=labels_mx[idx], color='green')
        # ax.plot(times - etOriginal, mx_i, label=labels_mx[idx], color='blue')
        ax.plot(times - etOriginal, mx_i - x_true_i, label=labels_mx[idx], color='blue')
        ax.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_i), linestyle='dashed', color='r', label=labels_Pxx[idx])
        ax.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_i), linestyle='dashed', color='r')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\Delta$" + f"{labels_mx[idx]} ({units[idx]})")
        ax.set_title(f"Error in {labels_mx[idx]} over Time")
        ax.grid()
        ax.legend()
        idx += 1
    
    plt.show()
