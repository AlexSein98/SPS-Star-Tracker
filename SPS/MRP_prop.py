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
alpha: float = 0.5
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


def xNext(dt: float, x: npt.NDArray, mu: float):
    # rb = x[0:3]
    r = x[3:6]
    v = x[6:9]
    sigma = x[9:12]
    omega = x[12:15]

    dt_substep = 5.0
    r_next = copy.deepcopy(r)
    v_next = copy.deepcopy(v)
    sigma_next = copy.deepcopy(sigma)

    for _ in range(int(dt / dt_substep)):  # Symplectic Euler I guess?
        v_next -= dt_substep * mu * r_next / np.linalg.norm(r_next) ** 3
        r_next += dt_substep * v_next
    
    sigma_MRP = ModifiedRodriguesParameters.FromVector(sigma_next)
    q_next = propagateQuat(sigma_MRP.ToQuat().conjugate(), omega, dt).conjugate()
    sigma_next = ModifiedRodriguesParameters.FromQuat(q_next).ToVector()

    # print(f"q_next     = {q_next.as_w_first_array()}")
    # print(f"sigma_next = {sigma_next}")
    
    omega_next = copy.deepcopy(omega)
    # sigma_next = normalizeSigma(sigma_next)
    T_next = ModifiedRodriguesParameters.FromVector(sigma_next).ToQuat().to_matrix().T  # Convert back to passive
    
    rb_next = (T_next @ np.array([r_next]).T).T[0]

    return np.array([rb_next[0], rb_next[1], rb_next[2], r_next[0], r_next[1], r_next[2], 
                     v_next[0], v_next[1], v_next[2], sigma_next[0], sigma_next[1], sigma_next[2],
                     omega_next[0], omega_next[1], omega_next[2]])


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
    return sp, wm, wc


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
    planetMRP = ModifiedRodriguesParameters.FromMatrix(planetRot.T)  # MRPs are active
    planetMRP_vec = planetMRP.ToVector()

    # planetAngVel = planetRot.T @ planetAngVel

    print(f"planetMRP_vec = {planetMRP_vec}")
    print(f"planetAngVel = {planetAngVel}\n")

    r_init: float = 1800.0e3
    r0 = np.array([r_init, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(gravModel.mu / r_init), 0.0])
    r0_b = (planetRot @ np.array([r0]).T).T[0]
    x_0 = np.array([r0_b[0], r0_b[1], r0_b[2], r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], 
                    planetMRP_vec[0], planetMRP_vec[1], planetMRP_vec[2],
                    planetAngVel[0], planetAngVel[1], planetAngVel[2]])
    Pxx_0 = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 
                              1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]))
    
    Pww = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 
                            1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7])) * 1e-1
    
    # Covariances
    Pvv_XNav = np.array([1.0, 1.0, 1.0]) * 1e-1 ** 2
    Pvv_SPASM = np.array([1.0, 1.0, 1.0]) * 1e-1 ** 2
    Pvv_SPS = np.array([1.0, 1.0, 1.0]) * 1e-6 ** 2
    Pvv = np.diag(np.concat((Pvv_XNav, Pvv_SPASM, Pvv_SPS)))

    # Logging
    mx_est: list[npt.NDArray] = [copy.deepcopy(x_0)]
    Pxx_est: list[npt.NDArray] = [copy.deepcopy(Pxx_0)]
    x_true: list[npt.NDArray] = [copy.deepcopy(x_0)]

    # Time
    N = 100
    times = np.linspace(etOriginal, etOriginal + 20000.0, N + 1)
    dt = times[1] - times[0]
    print(f"dt = {dt}")

    for k in range(len(times)):
        print(f"Starting iteration {k}/{len(times)}")
        etNow = times[k]

        # Propagate
        Sxx_prop = np.linalg.cholesky(Pxx_est[k-1])
        sp_prop, wm_prop, wc_prop = sigmaPointsProp(n, dt, mx_est[k-1], Sxx_prop, gravModel)
        
        mx_k_minus = wm_prop[0] * sp_prop[0]
        for i in range(1, 2 * n + 1):
            mx_k_minus += wm_prop[i] * sp_prop[i]
        
        Pxx_k_minus = wc_prop[0] * np.outer(sp_prop[0] - mx_k_minus, sp_prop[0] - mx_k_minus)
        for i in range(1, 2 * n + 1):
            Pxx_k_minus += wc_prop[i] * np.outer(sp_prop[i] - mx_k_minus, sp_prop[i] - mx_k_minus)
        Pxx_k_minus += Pww
        
        # mx_k_minus[9:12] = normalizeSigma(mx_k_minus[9:12])

        # Update
        x_true.append(xNext(dt, x_true[k-1], gravModel.mu))
        mx_est.append(mx_k_minus)
        Pxx_est.append(Pxx_k_minus)

    
    ########################
    ####    Plotting    ####
    ########################
    
    fig1 = plt.figure(layout='constrained')
    ax1 = fig1.add_subplot(231)
    ax2 = fig1.add_subplot(232)
    ax3 = fig1.add_subplot(233)
    ax4 = fig1.add_subplot(234)
    ax5 = fig1.add_subplot(235)
    ax6 = fig1.add_subplot(236)

    fig2 = plt.figure(layout='constrained')
    ax7 = fig2.add_subplot(331)
    ax8 = fig2.add_subplot(332)
    ax9 = fig2.add_subplot(333)
    ax10 = fig2.add_subplot(334)
    ax11 = fig2.add_subplot(335)
    ax12 = fig2.add_subplot(336)
    ax13 = fig2.add_subplot(337)
    ax14 = fig2.add_subplot(338)
    ax15 = fig2.add_subplot(339)

    mx_rb1 = np.array([mx[0] for mx in mx_est])
    mx_rb2 = np.array([mx[1] for mx in mx_est])
    mx_rb3 = np.array([mx[2] for mx in mx_est])
    mx_r1 = np.array([mx[3] for mx in mx_est])
    mx_r2 = np.array([mx[4] for mx in mx_est])
    mx_r3 = np.array([mx[5] for mx in mx_est])
    mx_v1 = np.array([mx[6] for mx in mx_est])
    mx_v2 = np.array([mx[7] for mx in mx_est])
    mx_v3 = np.array([mx[8] for mx in mx_est])
    mx_s1 = np.array([mx[9] for mx in mx_est])
    mx_s2 = np.array([mx[10] for mx in mx_est])
    mx_s3 = np.array([mx[11] for mx in mx_est])
    mx_w1 = np.array([mx[12] for mx in mx_est])
    mx_w2 = np.array([mx[13] for mx in mx_est])
    mx_w3 = np.array([mx[14] for mx in mx_est])

    x_true_rb1 = np.array([x_t[0] for x_t in x_true])
    x_true_rb2 = np.array([x_t[1] for x_t in x_true])
    x_true_rb3 = np.array([x_t[2] for x_t in x_true])
    x_true_r1 = np.array([x_t[3] for x_t in x_true])
    x_true_r2 = np.array([x_t[4] for x_t in x_true])
    x_true_r3 = np.array([x_t[5] for x_t in x_true])
    x_true_v1 = np.array([x_t[6] for x_t in x_true])
    x_true_v2 = np.array([x_t[7] for x_t in x_true])
    x_true_v3 = np.array([x_t[8] for x_t in x_true])
    x_true_s1 = np.array([x_t[9] for x_t in x_true])
    x_true_s2 = np.array([x_t[10] for x_t in x_true])
    x_true_s3 = np.array([x_t[11] for x_t in x_true])
    x_true_w1 = np.array([x_t[12] for x_t in x_true])
    x_true_w2 = np.array([x_t[13] for x_t in x_true])
    x_true_w3 = np.array([x_t[14] for x_t in x_true])

    Pxx_rb1 = np.array([Pxx[0, 0] for Pxx in Pxx_est])
    Pxx_rb2 = np.array([Pxx[1, 1] for Pxx in Pxx_est])
    Pxx_rb3 = np.array([Pxx[2, 2] for Pxx in Pxx_est])
    Pxx_r1 = np.array([Pxx[3, 3] for Pxx in Pxx_est])
    Pxx_r2 = np.array([Pxx[4, 4] for Pxx in Pxx_est])
    Pxx_r3 = np.array([Pxx[5, 5] for Pxx in Pxx_est])
    Pxx_v1 = np.array([Pxx[6, 6] for Pxx in Pxx_est])
    Pxx_v2 = np.array([Pxx[7, 7] for Pxx in Pxx_est])
    Pxx_v3 = np.array([Pxx[8, 8] for Pxx in Pxx_est])
    Pxx_s1 = np.array([Pxx[9, 9] for Pxx in Pxx_est])
    Pxx_s2 = np.array([Pxx[10, 10] for Pxx in Pxx_est])
    Pxx_s3 = np.array([Pxx[11, 11] for Pxx in Pxx_est])
    Pxx_w1 = np.array([Pxx[12, 12] for Pxx in Pxx_est])
    Pxx_w2 = np.array([Pxx[13, 13] for Pxx in Pxx_est])
    Pxx_w3 = np.array([Pxx[14, 14] for Pxx in Pxx_est])

    # ATTITUDE
    ax1.plot(times - etOriginal, (mx_s1 - x_true_s1)[1:], label=r"$m_{x}(9)$", color='blue')
    ax1.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_s1[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(9,9)$")
    ax1.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_s1[1:]), linestyle='dashed', color='r')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("MRP error (unitless)")
    ax1.set_title(r"Error in $m_{x}(9)$ over Time")
    ax1.grid()
    ax1.legend()

    ax2.plot(times - etOriginal, (mx_s2 - x_true_s2)[1:], label=r"$m_{x}(10)$", color='blue')
    ax2.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_s2[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(10,10)$")
    ax2.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_s2[1:]), linestyle='dashed', color='r')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("MRP error (unitless)")
    ax2.set_title(r"Error in $m_{x}(10)$ over Time")
    ax2.grid()
    ax2.legend()
    
    ax3.plot(times - etOriginal, (mx_s3 - x_true_s3)[1:], label=r"$m_{x}(11)$", color='blue')
    ax3.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_s3[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(11,11)$")
    ax3.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_s3[1:]), linestyle='dashed', color='r')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("MRP error (unitless)")
    ax3.set_title(r"Error in $m_{x}(11)$ over Time")
    ax3.grid()
    ax3.legend()

    # ANGULAR VELOCITY
    ax4.plot(times - etOriginal, (mx_w1 - x_true_w1)[1:], label=r"$m_{x}(12)$", color='blue')
    ax4.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_w1[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(12,12)$")
    ax4.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_w1[1:]), linestyle='dashed', color='r')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angular velocity error (rad/s)")
    ax4.set_title(r"Error in $m_{x}(12)$ over Time")
    ax4.grid()
    ax4.legend()

    ax5.plot(times - etOriginal, (mx_w2 - x_true_w2)[1:], label=r"$m_{x}(13)$", color='blue')
    ax5.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_w2[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(13,13)$")
    ax5.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_w2[1:]), linestyle='dashed', color='r')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Angular velocity error (rad/s)")
    ax5.set_title(r"Error in $m_{x}(13)$ over Time")
    ax5.grid()
    ax5.legend()
    
    ax6.plot(times - etOriginal, (mx_w3 - x_true_w3)[1:], label=r"$m_{x}(14)$", color='blue')
    ax6.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_w3[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(14,14)$")
    ax6.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_w3[1:]), linestyle='dashed', color='r')
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Angular velocity error (rad/s)")
    ax6.set_title(r"Error in $m_{x}(14)$ over Time")
    ax6.grid()
    ax6.legend()

    plt.show()
