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


n: int = 15
alpha: float = 0.5
beta: float = 2.0
kappa: float = 0.0
# lam: float = alpha ** 2 * (n + kappa) - n
lam: float = 0.0


def normalizeSigma(sigma: npt.NDArray) -> npt.NDArray:
    T = ModifiedRodriguesParameters.FromVector(sigma).ToMatrix()
    q = Quaternion.FromMatrix(T)
    return ModifiedRodriguesParameters.FromQuat(q.normalize()).ToVector()


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
    for _ in range(int(dt / dt_substep)):
        r_next += dt_substep * v_next
        v_next -= dt_substep * mu * r_next / np.linalg.norm(r_next) ** 3
        sigma2 = sigma_next[0] ** 2 + sigma_next[1] ** 2 + sigma_next[2] ** 2
        sigma_next += 0.25 * dt_substep * ((1.0 - sigma2) * omega - 2.0 * np.cross(omega, sigma_next) + 
                                2.0 * np.dot(omega, sigma_next) * sigma_next)
    
    omega_next = copy.deepcopy(omega)
    sigma_next = normalizeSigma(sigma_next)
    T_next = ModifiedRodriguesParameters.FromVector(sigma_next).ToMatrix()
    
    rb_next = (T_next @ np.array([r_next]).T).T[0]

    return np.array([rb_next[0], rb_next[1], rb_next[2], r_next[0], r_next[1], r_next[2], 
                     v_next[0], v_next[1], v_next[2], sigma_next[0], sigma_next[1], sigma_next[2],
                     omega_next[0], omega_next[1], omega_next[2]])


def h(x: npt.NDArray):
    r = x[3:6]
    # v = x[6:9]
    sigma = x[9:12]
    omega = x[12:15]

    T = ModifiedRodriguesParameters.FromVector(sigma).ToMatrix()
    rb = (T @ np.array([r]).T).T[0]

    return np.array([rb[0], rb[1], rb[2], r[0], r[1], r[2], omega[0], omega[1], omega[2]])


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


def sigmaPointsMeas(n: int, dt: float, mx: npt.NDArray, Sxx: npt.NDArray, gravModel: grav_base):
    sp: list[npt.NDArray] = [h(mx)]
    wm: list[float] = [lam / (n + lam)]
    wc: list[float] = [lam / (n + lam) + 1.0 - alpha ** 2 + beta]
    for i in range(n):
        sp.append(h(mx + np.sqrt(n + lam) * Sxx[:, i]))
        sp.append(h(mx - np.sqrt(n + lam) * Sxx[:, i]))
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
    planetMRP = ModifiedRodriguesParameters.FromMatrix(planetRot)
    planetMRP_vec = planetMRP.ToVector()

    print(f"planetMRP_vec = {planetMRP_vec}")
    print(f"planetAngVel = {planetAngVel}\n")

    r_init: float = 1800.0e3
    r0 = np.array([r_init, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(gravModel.mu / r_init), 0.0])
    r0_b = (planetRot @ np.array([r0]).T).T[0]
    x_0 = np.array([r0_b[0], r0_b[1], r0_b[2], r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], 
                    planetMRP_vec[0], planetMRP_vec[1], planetMRP_vec[2],
                    planetAngVel[0], planetAngVel[1], planetAngVel[2]])
    Pxx_0 = np.diag(np.array([5e3, 5e3, 5e3, 5e1, 5e1, 5e1, 1.0, 1.0, 1.0, 
                              1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4]))
    
    Pww = np.diag(np.array([1e-2, 1e-2, 1e-2, 1e1, 1e1, 1e1, 2e-3, 2e-3, 2e-3, 
                            1e-4, 1e-4, 1e-4, 1e-8, 1e-8, 1e-8]))
    
    # Covariances
    Pvv_XNav = np.array([1.0, 1.0, 1.0]) * 1e2 ** 2
    Pvv_SPASM = np.array([1.0, 1.0, 1.0]) * 1e2 ** 2
    Pvv_SPS = np.array([1.0, 1.0, 1.0]) * 1e-7 ** 2
    Pvv = np.diag(np.concat((Pvv_XNav, Pvv_SPASM, Pvv_SPS)))

    # Logging
    mx_est: list[npt.NDArray] = [copy.deepcopy(x_0)]
    Pxx_est: list[npt.NDArray] = [copy.deepcopy(Pxx_0)]
    x_true: list[npt.NDArray] = [copy.deepcopy(x_0)]

    # Time
    times = np.linspace(etOriginal, etOriginal + 20000.0, 100)
    dt = times[1] - times[0]

    update: bool = True
    measurementCadence: int = 2
    for k in range(len(times)):
        print(f"Starting iteration {k+1}/{len(times)}")
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
        
        mx_k_minus[9:12] = normalizeSigma(mx_k_minus[9:12])

        # Update
        if update and k != 0 and k % measurementCadence == 0:
            Sxx_meas = np.linalg.cholesky(Pxx_k_minus)
            sp_meas, wm_meas, wc_meas = sigmaPointsMeas(n, dt, mx_k_minus, Sxx_meas, gravModel)

            mz_k_minus = wm_meas[0] * sp_meas[0]
            for i in range(1, 2 * n + 1):
                mz_k_minus += wm_meas[i] * sp_meas[i]
            
            Pxz_k_minus = wc_meas[0] * np.outer(sp_prop[0] - mx_k_minus, sp_meas[0] - mz_k_minus)
            Pzz_k_minus = wc_meas[0] * np.outer(sp_meas[0] - mz_k_minus, sp_meas[0] - mz_k_minus)
            for i in range(1, 2 * n + 1):
                Pxz_k_minus = wc_meas[i] * np.outer(sp_prop[i] - mx_k_minus, sp_meas[i] - mz_k_minus)
                Pzz_k_minus = wc_meas[i] * np.outer(sp_meas[i] - mz_k_minus, sp_meas[i] - mz_k_minus)
            Pzz_k_minus += Pvv

            # Process measurement
            x_true.append(xNext(dt, x_true[k-1], gravModel.mu))
            z_k = h(x_true[k]) + np.linalg.cholesky(Pvv) @ np.random.randn(9)
        
            K = Pxz_k_minus @ (np.linalg.inv(Pzz_k_minus))
            mx_k_plus = mx_k_minus + K @ (z_k - mz_k_minus)
            Pxx_k_plus = Pxx_k_minus - Pxz_k_minus @ K.T - K @ Pxz_k_minus.T + K @ Pzz_k_minus @ K.T

            mx_k_plus[9:12] = normalizeSigma(mx_k_plus[9:12])
            
            mx_est.append(mx_k_plus)
            Pxx_est.append(Pxx_k_plus)
            
            print(f"mx_est[k+1] = {mx_k_plus}")
        else:
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

    # PLANET-FIXED POSITION
    ax7.plot(times - etOriginal, (mx_rb1 - x_true_rb1)[1:], label=r"$m_{x}(0)$", color='blue')
    ax7.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_rb1[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(0,0)$")
    ax7.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_rb1[1:]), linestyle='dashed', color='r')
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Body-fixed position error (m)")
    ax7.set_title(r"Error in $m_{x}(0)$ over Time")
    ax7.grid()
    ax7.legend()

    ax8.plot(times - etOriginal, (mx_rb2 - x_true_rb2)[1:], label=r"$m_{x}(1)$", color='blue')
    ax8.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_rb2[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(1,1)$")
    ax8.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_rb2[1:]), linestyle='dashed', color='r')
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Body-fixed position error (m)")
    ax8.set_title(r"Error in $m_{x}(1)$ over Time")
    ax8.grid()
    ax8.legend()
    
    ax9.plot(times - etOriginal, (mx_rb3 - x_true_rb3)[1:], label=r"$m_{x}(2)$", color='blue')
    ax9.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_rb3[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(2,2)$")
    ax9.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_rb3[1:]), linestyle='dashed', color='r')
    ax9.set_xlabel("Time (s)")
    ax9.set_ylabel("Body-fixed position error (m)")
    ax9.set_title(r"Error in $m_{x}(2)$ over Time")
    ax9.grid()
    ax9.legend()

    # INERTIAL POSITION
    ax10.plot(times - etOriginal, (mx_r1 - x_true_r1)[1:], label=r"$m_{x}(3)$", color='blue')
    ax10.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_r1[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(3,3)$")
    ax10.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_r1[1:]), linestyle='dashed', color='r')
    ax10.set_xlabel("Time (s)")
    ax10.set_ylabel("Inertial position error (m)")
    ax10.set_title(r"Error in $m_{x}(3)$ over Time")
    ax10.grid()
    ax10.legend()

    ax11.plot(times - etOriginal, (mx_r2 - x_true_r2)[1:], label=r"$m_{x}(4)$", color='blue')
    ax11.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_r2[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(4,4)$")
    ax11.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_r2[1:]), linestyle='dashed', color='r')
    ax11.set_xlabel("Time (s)")
    ax11.set_ylabel("Inertial position error (m)")
    ax11.set_title(r"Error in $m_{x}(4)$ over Time")
    ax11.grid()
    ax11.legend()
    
    ax12.plot(times - etOriginal, (mx_r3 - x_true_r3)[1:], label=r"$m_{x}(5)$", color='blue')
    ax12.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_r3[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(5,5)$")
    ax12.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_r3[1:]), linestyle='dashed', color='r')
    ax12.set_xlabel("Time (s)")
    ax12.set_ylabel("Inertial position error (m)")
    ax12.set_title(r"Error in $m_{x}(5)$ over Time")
    ax12.grid()
    ax12.legend()

    # INERTIAL VELOCITY
    ax13.plot(times - etOriginal, (mx_v1 - x_true_v1)[1:], label=r"$m_{x}(6)$", color='blue')
    ax13.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_v1[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(6,6)$")
    ax13.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_v1[1:]), linestyle='dashed', color='r')
    ax13.set_xlabel("Time (s)")
    ax13.set_ylabel("Inertial velocity error (m/s)")
    ax13.set_title(r"Error in $m_{x}(6)$ over Time")
    ax13.grid()
    ax13.legend()

    ax14.plot(times - etOriginal, (mx_v2 - x_true_v2)[1:], label=r"$m_{x}(7)$", color='blue')
    ax14.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_v2[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(7,7)$")
    ax14.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_v2[1:]), linestyle='dashed', color='r')
    ax14.set_xlabel("Time (s)")
    ax14.set_ylabel("Inertial velocity error (m/s)")
    ax14.set_title(r"Error in $m_{x}(7)$ over Time")
    ax14.grid()
    ax14.legend()
    
    ax15.plot(times - etOriginal, (mx_v3 - x_true_v3)[1:], label=r"$m_{x}(8)$", color='blue')
    ax15.plot(times - etOriginal, -3.0 * np.sqrt(Pxx_v3[1:]), linestyle='dashed', color='r', label=r"$P_{xx}(8,8)$")
    ax15.plot(times - etOriginal, 3.0 * np.sqrt(Pxx_v3[1:]), linestyle='dashed', color='r')
    ax15.set_xlabel("Time (s)")
    ax15.set_ylabel("Inertial velocity error (m/s)")
    ax15.set_title(r"Error in $m_{x}(8)$ over Time")
    ax15.grid()
    ax15.legend()

    plt.show()
