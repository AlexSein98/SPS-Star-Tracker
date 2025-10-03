from py_src.star.python.transformations import *
from SPS.gravity import *
from SPS.SPS_samples import ReadDEM, SampleGlobalDEM_LatLon
from SPS.global_config import *

import sys
import os
import csv
import copy
import time
import datetime

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


def SnapToSurface(xyz: np.ndarray[float], a: float, b: float, dem: np.ndarray[float]) -> np.ndarray[float]:
    phi_pg, lon, _ = cartesian_to_planetographic(xyz, a, b)
    h_ellp_true = SampleGlobalDEM_LatLon(dem, phi_pg, lon)
    if planet.planetName == "EARTH":
        h_ellp_true = max(h_ellp_true, 0.0)
    return planetographic_to_cartesian(phi_pg, lon, h_ellp_true, a, b)


def estimate_position(truthDataPath: str, estDataPath: str, latLonDataPath: str, planet: Planet, gravModel: grav_base):
    np.set_printoptions(suppress=True)

    # Transformation from inertial to planet frame
    home = "./py_src/star/"
    spice.furnsh(home + "data/metakernel.txt")
    tNow = '2025 July 4, 00:00:00 UTC'
    etNow = spice.str2et(tNow)
    T_i_b = spice.pxform("J2000", planet.planetFrame, etNow)

    dem = ReadDEM(planet.demName)

    truthData = read_csv(truthDataPath)
    estData = read_csv(estDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
    latLonData = read_csv(latLonDataPath)
    
    # Very basic error handling if datasets are not the same length
    if not (len(truthData) == len(estData) == len(latLonData)):
        print(f'Warning: early exit due to dataset length mismatch; truthData length = {len(truthData)}, estData length = {len(estData)}, and latLonData length = {len(latLonData)}.')
        return

    maxDegree: int = globalConfig.grav_maxDegree
    maxOrder: int = globalConfig.grav_maxOrder
    sampler = GravSampler(gravModel, maxDegree, maxOrder)

    distanceErrors_m: list[float] = []
    distanceErrors_km: list[float] = []

    L: int = len(truthData)
    numLon: int = int(1.0 + np.sqrt(1.0 + 2.0 * float(L)))
    numLat: int = int(float(L) / float(numLon))
    distanceErrorArray_m: np.ndarray[float] = np.zeros((numLat, numLon))
    distanceErrorArray_km: np.ndarray[float] = np.zeros((numLat, numLon))
    limit_km: float = 20 * planet.radius * 1e-6

    # Gravity gradient grid:
    # Gxx, Gyy, Gzz, Gxy, Gxz, Gyz = sampler.GetGradientGrid(maxDegree)

    scaleFactor = 1000.0 if planet.demUnits == "km" else 1.0

    startTime = time.perf_counter()
    elapsedSeconds: float = 0.0
    printInterval: int = 10

    for j in range(L):
        doPrint: bool = j % printInterval == 0

        truth_j = truthData[j]
        est_j = estData[j]
        latLon_j = latLonData[j]
        
        if est_j[0] == 999 or est_j[1] == 999 or est_j[2] == 999 or est_j[3] == 999:
            print(f'Warning: skipped measurement at index {j} (invalid quaternion).')
            continue
        
        q_real = Quaternion(truth_j[0], truth_j[1], truth_j[2], truth_j[3]).normalize()
        q_est = Quaternion(est_j[0], est_j[1], est_j[2], est_j[3]).normalize()
        
        T_i_c = q_est.to_matrix()
        T_g_c = np.identity(3)  # transformation from gravity to camera frame

        Omega = np.array([0.0, 0.0, gravModel.omega])
        g_sensorFrame: np.ndarray[float] = np.array([-gravModel.mu / (gravModel.radius ** 2), 0.0, 0.0])

        # Coarse 1 (from "Hardware Improvements" paper)
        r_coarse_1 = -(gravModel.radius ** 3 / gravModel.mu) * (T_i_b @ T_i_c.T @ T_g_c @ g_sensorFrame)

        # Coarse 2 (from "Hardware Improvements" paper)
        r_coarse_2 = -(gravModel.radius ** 3 / gravModel.mu) * (T_i_b @ T_i_c.T @ T_g_c @ g_sensorFrame + np.cross(Omega, np.cross(Omega, r_coarse_1)))

        # Coarse 3: apply angular velocity again
        r_coarse_3 = -(np.linalg.norm(gravModel.radius) ** 3 / gravModel.mu) * (T_i_b @ T_i_c.T @ T_g_c @ g_sensorFrame + np.cross(Omega, np.cross(Omega, r_coarse_2)))
        r_coarse_3 = SnapToSurface(r_coarse_3, gravModel.radius, gravModel.polarRadius, dem)

        phi_pg, lon2, _ = cartesian_to_planetographic(r_coarse_3, gravModel.radius, gravModel.polarRadius)
        alt2 = scaleFactor * SampleGlobalDEM_LatLon(dem, phi_pg, lon2)
        if planet.planetName == "EARTH":
                alt2 = max(alt2, 0.0)
        r_coarse_3 = planetographic_to_cartesian(phi_pg, lon2, alt2, gravModel.radius, gravModel.polarRadius)
        
        if doPrint:
            print(f'Sample point {j}:')
            # print(f'lat = {phi_pg}')
            # print(f'lon = {lon2}')
            # print(f'alt = {alt2}')
            # print(f'||r_coarse_1||   = {np.linalg.norm(r_coarse_1)}')
            # print(f'||r_coarse_2||   = {np.linalg.norm(r_coarse_2)}')
            # print(f'||r_coarse_3||   = {np.linalg.norm(r_coarse_3)}')
        
        #"""
        r_hat_i_plus_1 = copy.deepcopy(r_coarse_3)
        r_hat_i = np.zeros(3)

        phi_pg, lon, _ = cartesian_to_planetographic(r_hat_i_plus_1, gravModel.radius, gravModel.polarRadius)
        alt = scaleFactor * SampleGlobalDEM_LatLon(dem, phi_pg, lon)
        if planet.planetName == "EARTH":
            alt = max(alt, 0.0)

        # Gradient method:
        i: int = 0

        tol: float = planet.radius * 1e-6  # m
        dr: np.ndarray[float] = np.zeros(3)
        dr_prev: np.ndarray[float] = 2.0 * tol * np.ones(3)
        
        latTruth = float(latLon_j[0])
        lonTruth = float(latLon_j[1])
        altTruth = float(latLon_j[2])
        
        # while np.linalg.norm(dr - dr_prev) > tol:
        while np.linalg.norm(r_hat_i_plus_1 - r_hat_i) > tol:
            i += 1
            r_hat_i = copy.deepcopy(r_hat_i_plus_1)
            dr_prev = copy.deepcopy(dr)
            
            # Gravitational acceleration:
            phi_pc, _, _ = r_to_latlonalt(r_hat_i, gravModel.radius)
            phi_pg_test, _, _ = cartesian_to_planetographic(r_hat_i, gravModel.radius, gravModel.polarRadius)
            g = sampler.SampleAcceleration_Custom(phi_pc, lon, np.linalg.norm(r_hat_i), maxDegree)
            # g += np.cross(Omega, np.cross(Omega, r_hat_i))

            # Gradient
            # dXYZ: float = gravModel.radius * 0.5e-3  # m
            dXYZ: float = 1000.0  # m
            # G: np.ndarray[float] = sampler.SampleGradient(lat, lon, gravModel.radius + alt, maxDegree, dXYZ)
            G = sampler.SampleGradient_Numerical(r_hat_i, maxDegree, dXYZ)
            # G = sampler.InterpolateGradientGrid(phi_pc, lon, Gxx, Gyy, Gzz, Gxy, Gxz, Gyz)
            G_inv = np.linalg.inv(G)

            g_sensorFrame: np.ndarray[float] = np.array([-np.linalg.norm(g), 0.0, 0.0])
            g_est = T_i_b @ T_i_c.T @ T_g_c @ g_sensorFrame + np.cross(Omega, np.cross(Omega, r_hat_i))
            dg = g - g_est

            dr = G_inv @ dg
            r_hat_i_plus_1 = r_hat_i - 0.5 * dr

            # r_hat_i_plus_1 = SnapToSurface(r_hat_i_plus_1, gravModel.radius, gravModel.polarRadius, dem)

            phi_pg, lon, _ = cartesian_to_planetographic(r_hat_i_plus_1, gravModel.radius, gravModel.polarRadius)
            alt = scaleFactor * SampleGlobalDEM_LatLon(dem, phi_pg, lon)
            if planet.planetName == "EARTH":
                alt = max(alt, 0.0)
            
            r_hat_i_plus_1 = planetographic_to_cartesian(phi_pg, lon, alt, gravModel.radius, gravModel.polarRadius)
            
            # This shouldn't happen anymore? But it is???
            if i > 100:
                print(f'    Run {i} dr = {np.round(dr, 3)} m')
            
            if doPrint:
                print(f'    Run {i} r    = {np.round(r_hat_i_plus_1, 3)} m, phi_pc = {round(phi_pc, 6)}, phi_pg = {round(phi_pg_test, 6)}')

        if doPrint:
            print(f'Estimated lat = {round(phi_pg, 6)} deg')
            print(f'Estimated lon = {round(lon, 6)} deg')
            print(f'True lat = {round(latTruth, 6)} deg')
            print(f'True lon = {round(lonTruth, 6)} deg\n')
        #"""
        
        # latTruth = float(latLon_j[0])
        # lonTruth = float(latLon_j[1])
        # altTruth = float(latLon_j[2])
        # lon = lon2
        # alt = alt2
        # r_hat_i_plus_1 = r_coarse_3

        # distanceErr_m = archaversine(gravModel.radius, np.deg2rad(latTruth), np.deg2rad(phi_pg), np.deg2rad(lonTruth), np.deg2rad(lon))
        
        # Use direct vector distance instead of Haversine distance
        distanceErrVec_m = planetographic_to_cartesian(latTruth, lonTruth, altTruth, gravModel.radius, gravModel.polarRadius) - \
            planetographic_to_cartesian(phi_pg, lon, alt, gravModel.radius, gravModel.polarRadius)
        
        # Subtract the radial component (TODO: subtract average terrain normal instead for a better estimate?)
        distanceErr_m = np.linalg.norm(distanceErrVec_m - np.dot(normalize(r_hat_i_plus_1), 
                                                                 distanceErrVec_m) * normalize(r_hat_i_plus_1))
        distanceErr_km = 0.001 * distanceErr_m

        if distanceErr_km < limit_km:
            distanceErrors_m.append(distanceErr_m)
            distanceErrors_km.append(distanceErr_km)

        # Distance errors for plotting
        latIdx = int(numLat * (90.0 - latTruth) / 180.0)
        lonIdx = int(numLon * (lonTruth + 180.0) / 360.0)
        distanceErrorArray_m[latIdx, lonIdx] = distanceErr_m if distanceErr_m < 1000.0 * limit_km else 1000.0 * limit_km
        distanceErrorArray_km[latIdx, lonIdx] = distanceErr_km if distanceErr_km < limit_km else limit_km
        
        percentComplete = round(100.0 * float(j) / float(L), 3)
        
        endTime = time.perf_counter()
        elapsedSeconds = endTime - startTime
        elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

        if doPrint:
            print(f'Sample {j}/{L} after {i} iterations ({percentComplete} %): Distance error = {round(distanceErr_m, 1)} m')
            print(f'Elapsed time: {elapsedTime}')
            print("------------------------------------------------------------------------------------------------------\n")
    
    # Calculate statistics
    mean = np.mean(distanceErrors_m)
    median = np.median(distanceErrors_m)
    std = np.std(distanceErrors_m)

    # Print statistics
    print("======================================================================================================")
    print(f'Mean                    = {round(mean, 1)} m')
    print(f'Median                  = {round(median, 1)} m')
    print(f'Standard Deviation      = {round(std, 1)} m')
    print(f'Minimum                 = {round(min(distanceErrors_m), 3)} m = {round(min(distanceErrors_km), 6)} km')
    print(f'Maximum                 = {round(max(distanceErrors_m), 3)} m = {round(max(distanceErrors_km), 6)} km')
    print(f'Total elapsed time: {elapsedTime}')

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
    extraPrint = f' (errors > {limit_km} km rejected)'
    if len(distanceErrors_m) == len(truthData):
        extraPrint = f' (no errors rejected)'
    ax1.set_title(f'Position Error Counts for {len(distanceErrors_m)}/{len(truthData)} Samples' + extraPrint)
    ax1.set_xlabel('Error (m)')
    ax1.set_ylabel('Count')
    ax1.grid()
    ax2.set_title(f'Position Error Statistics for {len(distanceErrors_m)}/{len(truthData)} Samples' + extraPrint)
    ax2.set_xlabel('Error (m)')
    ax2.grid()

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    plt.sca(ax3)

    # cmap options: [RdYlGn_r, rainbow]
    plt.imshow(distanceErrorArray_m, cmap='rainbow', aspect='equal', extent = [-180,180,-90,90])
    plt.colorbar(label='Position Error (m)')
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Position Estimation Error from Recursive SPS Algorithm")

    # Show plot
    plt.show()


if __name__ == "__main__":
    os.system('cls')

    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel

    print(f'gravModel stats: name = {gravModel.name}, maxDegree = {gravModel.maxDegree}, radius = {gravModel.radius}, Clm[2, 0] = {gravModel.Clm[2][0]}')

    # Star tracker "measurements" file
    n = len(sys.argv)
    measurements = "./output_" + planet.planetName.title() + ".csv"
    if n > 1:
        measurements = sys.argv[1]
    
    # Truth source directory
    truthSourceDir = "./py_src/star/"
    if n > 2:
        truthSourceDir = sys.argv[2]
    
    # Latitude, longitude, and altitude
    latLonDataPath = "./sampleLatLongs_" + planet.planetName.title() + ".csv"
    if n > 3:
        latLonDataPath = sys.argv[3]

    estimate_position(truthSourceDir + "truth_data_" + planet.planetName.title() + ".csv", measurements, latLonDataPath, planet, gravModel)
    pass
