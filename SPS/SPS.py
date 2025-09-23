# from SPS.grav_moon_GRAIL150 import *
from py_src.star.python.transformations import *
from SPS.gravity import *
from SPS.SPS_samples import *

import sys
import os
import csv
import copy

from matplotlib import pyplot as plt
import matplotlib
import spiceypy as spice


def read_csv(path: str, ignore: list=[], hasHeader=False):
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"', lineterminator='\n')
        data = []

        headerBool = False
        if hasHeader:
            headerBool = True
        for row in reader:
            if headerBool:
                headerBool = False
                continue
            data.append([float(row[i]) for i in range(len(row)) if i not in ignore])
        return data


def matrix_to_angleaxis(m: np.ndarray[float]):
    # Angle in radians
    phi_r = np.arccos(0.5 * (np.trace(m) - 1.0))

    # Axis components
    sin_phi_2 = 0.5 / np.sin(phi_r)
    e0 = (m[1][2] - m[2][1]) * sin_phi_2
    e1 = (m[2][0] - m[0][2]) * sin_phi_2
    e2 = (m[0][1] - m[1][0]) * sin_phi_2

    # Return angle (in degrees) + axis
    phi_d = np.rad2deg(phi_r)
    e = np.array([e0, e1, e2])
    return phi_d, e


def deg_to_arcsec(angle):
    return angle * 3600.0


def arcsec_to_rad(angle):
    return np.deg2rad(angle / 3600.0)


def plot_errors(truthDataPath: str, estDataPath: str, reject: float=0, planetRadius: float=6378136.3):
    truthData = read_csv(truthDataPath)
    estData = read_csv(estDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
    
    # Very basic error handling if datasets are not the same length
    if len(truthData) != len(estData):
        return

    # Compare data
    errorsArcsec = []
    for i in range(len(truthData)):
        truth_i = truthData[i]
        est_i = estData[i]
        
        if est_i[0] == 999 or est_i[1] == 999 or est_i[2] == 999 or est_i[3] == 999:
            continue
        
        q_real = Quaternion(truth_i[0], truth_i[1], truth_i[2], truth_i[3]).normalize()
        q_est = Quaternion(est_i[0], est_i[1], est_i[2], est_i[3]).normalize()
        q_err = q_est.conjugate().mult(q_real).normalize()
        m = spice.q2m(q_err.as_w_first_array())
        phi_d, _ = matrix_to_angleaxis(m)
        phi_arcsec = deg_to_arcsec(phi_d)
        
        if reject != 0 and phi_arcsec > reject:
            continue
        errorsArcsec.append(phi_arcsec)
    
    # Calculate statistics
    mean = np.mean(errorsArcsec)
    median = np.median(errorsArcsec)
    std = np.std(errorsArcsec)
    projectedSPSMeanError = arcsec_to_rad(mean) * planetRadius

    # Print statistics
    print(f'Mean                        = {round(mean, 3)} arcseconds')
    print(f'Median                      = {round(median, 3)} arcseconds')
    print(f'Standard Deviation          = {round(std, 3)} arcseconds')
    print(f'Projected SPS Mean Error    = {round(projectedSPSMeanError, 1)} meters')

    # Plot histogram and box plot of errors
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.hist(errorsArcsec, bins=200, color='skyblue', edgecolor='black')
    mpl_v = matplotlib.__version__.split(".")
    mpl_v_maj = float(mpl_v[0])
    mpl_v_min = float(mpl_v[1])
    # mpl_v_patch = float(mpl_v[2])

    # If matplotlib version is too low, the "orientation" kwarg is invalid. Need to use "vert: bool" for matplotlib < 3.10
    if mpl_v_maj < 3 or mpl_v_min < 10:
        ax2.boxplot(errorsArcsec, vert=False, showfliers=False)
    else:
        ax2.boxplot(errorsArcsec, orientation='horizontal', showfliers=False)

    # Nice plot stuff
    extraPrint = f' (errors > {reject} arcseconds rejected)'
    if reject == 0:
        extraPrint = f' (no errors rejected)'
    ax1.set_title(f'Attitude Error Counts for {len(errorsArcsec)}/{len(truthData)} Samples' + extraPrint)
    ax1.set_xlabel('Error (arcseconds)')
    ax1.set_ylabel('Count')
    ax1.grid()
    ax2.set_title(f'Attitude Error Statistics for {len(errorsArcsec)}/{len(truthData)} Samples' + extraPrint)
    ax2.set_xlabel('Error (arcseconds)')
    ax2.grid()

    # Show plot
    plt.show()


def estimate_position(truthDataPath: str, estDataPath: str, latLonDataPath: str):    
    # Transformation from inertial to planet frame
    home = ".\\py_src\\star\\"
    spice.furnsh(home + "data\\metakernel.txt")
    tNow = '2025 July 4, 00:00:00 UTC'
    etNow = spice.str2et(tNow)
    T_i_b = spice.pxform("J2000", "MOON_PA", etNow)

    dem = ReadDEM(moonDEM)

    truthData = read_csv(truthDataPath)
    estData = read_csv(estDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
    latLonData = read_csv(latLonDataPath)
    
    # Very basic error handling if datasets are not the same length
    if not (len(truthData) == len(estData) == len(latLonData)):
        return

    maxDegree = 128
    maxOrder = 128
    sampler = GravSampler(maxDegree, maxOrder)

    distanceErrors_m: list[float] = []
    distanceErrors_km: list[float] = []

    L: int = len(truthData)
    numLon: int = int(1.0 + np.sqrt(1.0 + 2.0 * float(L)))
    numLat: int = int(float(L) / float(numLon))
    distanceErrorArray_m: np.ndarray[float] = np.zeros((numLat, numLon))
    distanceErrorArray_km: np.ndarray[float] = np.zeros((numLat, numLon))

    for j in range(L):
    # for j in range(len(truthData) - 1, len(truthData)):
        truth_j = truthData[j]
        est_j = estData[j]
        latLon_j = latLonData[j]
        
        if est_j[0] == 999 or est_j[1] == 999 or est_j[2] == 999 or est_j[3] == 999:
            continue
        
        q_real = Quaternion(truth_j[0], truth_j[1], truth_j[2], truth_j[3]).normalize()
        q_est = Quaternion(est_j[0], est_j[1], est_j[2], est_j[3]).normalize()
        
        lat = 0.0 
        lon = 0.0
        
        T_g_c = np.identity(3)  # transformation from gravity to camera frame

        # print(f'T_i_b = {T_i_b}')
        
        T_s_g = np.identity(3)  # transformation from surface to gravity frame
        T_s_g_old = T_angle_axis(np.deg2rad(1.0), np.array([0, 0, 1]))
        
        T_i_c = q_est.to_matrix()
        
        i: int = 0
        tol_angle_arcsec = 0.001
        tol_angle_deg = tol_angle_arcsec / 3600.0

        p_hat_gc: np.ndarray[float] = np.array([0.0, 1.0, 0.0])
        p_hat_as: np.ndarray[float] = np.array([1.0, 0.0, 0.0])
        
        # while abs(np.rad2deg(arccos_safe(np.dot(normalize(p_hat_gc), normalize(p_hat_as))))) > tol_angle_deg:
        while abs(AttitudeError(T_s_g_old, T_s_g)) > tol_angle_deg:
            i += 1
            T_s_g_old = copy.deepcopy(T_s_g)
            
            T_b_s = T_s_g.T @ T_g_c.T @ T_i_c @ T_i_b.T  # transformation from planet to surface frame
            lat, lon = T_to_latlon(T_b_s.T)
            
            altEstimate = SampleDEM_LatLon(dem, lat, lon)
            g = sampler.SampleAcceleration(lat, lon, moon.radius + 0.001 * altEstimate, maxDegree)
            # g = sample_gravity(moon, Cilm, lat, lon, moon.radius, max_degree)
            p_hat_gc = T_b_s.T[:, 0]
            # g = -moon.mu * p_hat_gc / ((moon.radius + 0.001 * latLon_j[2]) ** 2)
            p_hat_as = -normalize(g)

            T_s_g = TwoVectors_to_T(p_hat_gc, p_hat_as)
            
            # print(f'Run {i}:')
            # print(f'p_hat_gc = {np.round(p_hat_gc, 6)}')
            # print(f'p_hat_as = {np.round(p_hat_as, 6)}')
            # print(f'err = {round(deg_to_arcsec(np.rad2deg(arccos_safe(np.dot(normalize(p_hat_gc), normalize(p_hat_as))))), 6)}"')
            # print(f'lat = {round(lat, 6)} deg')
            # print(f'lon = {round(lon, 6)} deg\n')
        
        latTruth = float(latLon_j[0])
        lonTruth = float(latLon_j[1])

        # print(f'Estimated lat = {round(lat, 6)} deg')
        # print(f'Estimated lon = {round(lon, 6)} deg')
        # print(f'True lat = {round(latTruth, 6)} deg')
        # print(f'True lon = {round(lonTruth, 6)} deg\n')

        distanceErr_km = archaversine(moon.radius, np.deg2rad(latTruth), np.deg2rad(lat), np.deg2rad(lonTruth), np.deg2rad(lon))
        distanceErr_m = 1000.0 * distanceErr_km
        distanceErrors_m.append(distanceErr_m)
        distanceErrors_km.append(distanceErr_km)

        # Distance errors for plotting
        latIdx = int(numLat * (90.0 - latTruth) / 180.0)
        lonIdx = int(numLon * (lonTruth + 180.0) / 360.0)
        limit_km: float = 10.0
        distanceErrorArray_m[latIdx, lonIdx] = distanceErr_m if distanceErr_m < 1000.0 * limit_km else 1000.0 * limit_km
        distanceErrorArray_km[latIdx, lonIdx] = distanceErr_km if distanceErr_km < limit_km else limit_km
        
        percentComplete = round(100.0 * float(j) / float(L), 3)
        print(f'Sample {j}/{L} ({percentComplete} %): Distance error = {round(distanceErr_m, 1)} m')
    
    # Calculate statistics
    mean = np.mean(distanceErrors_m)
    median = np.median(distanceErrors_m)
    std = np.std(distanceErrors_m)

    # Print statistics
    print("")
    print(f'Mean                    = {round(mean, 1)} m')
    print(f'Median                  = {round(median, 1)} m')
    print(f'Standard Deviation      = {round(std, 1)} m')
    print(f'Minimum                 = {round(min(distanceErrors_m), 3)} m = {round(min(distanceErrors_km), 6)} km')
    print(f'Maximum                 = {round(max(distanceErrors_m), 3)} m = {round(max(distanceErrors_km), 6)} km')

    # Plot histogram and box plot of errors
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.hist(distanceErrors_m, bins=200, color='skyblue', edgecolor='black')
    mpl_v = matplotlib.__version__.split(".")
    mpl_v_maj = float(mpl_v[0])
    mpl_v_min = float(mpl_v[1])
    # mpl_v_patch = float(mpl_v[2])

    # If matplotlib version is too low, the "orientation" kwarg is invalid. Need to use "vert: bool" for matplotlib < 3.10
    if mpl_v_maj < 3 or mpl_v_min < 10:
        ax2.boxplot(distanceErrors_m, vert=False, showfliers=False)
    else:
        ax2.boxplot(distanceErrors_m, orientation='horizontal', showfliers=False)

    # Nice plot stuff
    ax1.set_title(f'Position Error Counts for {len(distanceErrors_m)}/{len(truthData)} Samples')
    ax1.set_xlabel('Error (m)')
    ax1.set_ylabel('Count')
    ax1.grid()
    ax2.set_title(f'Position Error Statistics for {len(distanceErrors_m)}/{len(truthData)} Samples')
    ax2.set_xlabel('Error (m)')
    ax2.grid()

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    plt.sca(ax3)

    plt.imshow(distanceErrorArray_m, cmap='RdYlGn_r', aspect='equal', extent = [-180,180,-90,90])
    plt.colorbar(label='Position Error (m)')
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Position Estimation Error from Recursive SPS Algorithm")

    # Show plot
    plt.show()


if __name__ == "__main__":
    os.system('cls')

    rEarth = 6378136.3
    rMoon = 1737400.0
    rMars = 3396190.0
    rPhobos = 11000.0

    # Star tracker "measurements" file
    n = len(sys.argv)
    measurements = ".\\output.csv"
    if n > 1:
        measurements = sys.argv[1]
    
    # Truth source directory
    truthSourceDir = ".\\py_src\\star\\"
    if n > 2:
        truthSourceDir = sys.argv[2]
    
    # Latitude, longitude, and altitude
    latLonDataPath = ".\\sampleLatLongs.csv"
    if n > 3:
        latLonDataPath = sys.argv[3]
    
    # # Cutoff for "good measurements" in arcseconds
    # reject = 0.0
    # if n > 3:
    #     reject = float(sys.argv[3])
    
    # # Planet radius for evaluating location error
    # radius = 0.0
    # if n > 4:
    #     radius = float(sys.argv[4])
    # else:
    #     radius = rEarth

    estimate_position(truthSourceDir + "truth_data.csv", measurements, latLonDataPath)
    pass
