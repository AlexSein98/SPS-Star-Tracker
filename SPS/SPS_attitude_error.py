import numpy as np

import sys
import csv

from matplotlib import pyplot as plt
import matplotlib
import spiceypy as spice
from py_src.star.python.transformations import *
from SPS.global_config import *


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


def plot_errors(truthDataPath: str, estDataPath: str, latLonDataPath: str, reject: float, planetRadius):
    truthData = read_csv(truthDataPath)
    estData = read_csv(estDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
    latLonData = read_csv(latLonDataPath)
    
    # Very basic error handling if datasets are not the same length
    if len(truthData) != len(estData):
        print(f"Unequal length! Truth data length is {len(truthData)}, while estimate data length is {len(estData)}")
        return

    # Initialize error arrays
    errorsArcsec = []
    L: int = len(truthData)
    numLon: int = int(1.0 + np.sqrt(1.0 + 2.0 * float(L)))
    numLat: int = int(float(L) / float(numLon))
    errorsArcsecArray: np.ndarray[float] = np.zeros((numLat, numLon))

    # Compare data
    for i in range(len(truthData)):
        truth_i = truthData[i]
        est_i = estData[i]
        latLon_i = latLonData[i]
        
        if est_i[0] == 999 or est_i[1] == 999 or est_i[2] == 999 or est_i[3] == 999:
            continue
        
        q_real = Quaternion(truth_i[0], truth_i[1], truth_i[2], truth_i[3]).normalize()
        q_est = Quaternion(est_i[0], est_i[1], est_i[2], est_i[3]).normalize()
        q_err = q_est.conjugate().mult(q_real).normalize()
        m = spice.q2m(q_err.as_w_first_array())
        phi_d, _ = matrix_to_angleaxis(m)
        phi_arcsec = deg_to_arcsec(phi_d)

        # Add to plotting array even if it's rejected
        latTruth = float(latLon_i[0])
        lonTruth = float(latLon_i[1])
        latIdx = int(numLat * (90.0 - latTruth) / 180.0)
        lonIdx = int(numLon * (lonTruth + 180.0) / 360.0)

        # print(f'latIdx = {latIdx}')
        # print(f'lonIdx = {lonIdx}\n')

        if reject != 0:
            errorsArcsecArray[latIdx, lonIdx] = phi_arcsec if phi_arcsec < reject else reject
        else:
            errorsArcsecArray[latIdx, lonIdx] = phi_arcsec

        if reject != 0 and phi_arcsec > reject:
            continue
        errorsArcsec.append(phi_arcsec)
    
    # for i in range(len(errorsArcsecArray)):
    #     errorsArcsecArrayLine = errorsArcsecArray[i]
    #     for j in range(len(errorsArcsecArrayLine)):
    #         print(f'Error at [{i}, {j}]: {errorsArcsecArrayLine[j]}')
    
    # Calculate statistics
    mean = np.mean(errorsArcsec)
    median = np.median(errorsArcsec)
    std = np.std(errorsArcsec)
    projectedSPSMeanError = arcsec_to_rad(mean) * planetRadius

    # Print statistics
    print(f'Mean                        = {round(mean, 3)} arcseconds')
    print(f'Median                      = {round(median, 3)} arcseconds')
    print(f'Standard Deviation          = {round(std, 3)} arcseconds')
    print(f'Minimum                     = {round(min(errorsArcsec), 3)} arcseconds = {round(min(errorsArcsec) / 3600.0, 6)} deg')
    print(f'Maximum                     = {round(max(errorsArcsec), 3)} arcseconds = {round(max(errorsArcsec) / 3600.0, 6)} deg')
    print(f'Projected SPS Mean Error    = {round(projectedSPSMeanError, 1)} meters')

    # Plot histogram and box plot of errors
    fig = plt.figure(layout='constrained')
    fig.set_size_inches(9.6, 5.4)
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

    outputFile1 = globalConfig.outputDir + globalConfig.nameTitle + "AttitudeErrorStatistics.png"
    plt.savefig(outputFile1, bbox_inches='tight', facecolor='white', transparent="False", pad_inches=0.25, dpi=300)
    print(f"Saved {globalConfig.nameTitle} attitude error statistics to {outputFile1}")

    fig2 = plt.figure(layout='constrained')
    fig2.set_size_inches(9.6, 5.4)
    ax3 = fig2.add_subplot(111)
    plt.sca(ax3)

    # cmap options: [RdYlGn_r, rainbow]
    plt.imshow(errorsArcsecArray, cmap='rainbow', aspect='equal', extent = [-180,180,-90,90])
    plt.colorbar(label='Star Sensor Attitude Error (arcsec)')
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Star Sensor Attitude Estimation Error")

    outputFile2 = globalConfig.outputDir + globalConfig.nameTitle + "AttitudeErrorMap.png"
    fig2.savefig(outputFile2, bbox_inches='tight', facecolor='white', transparent="False", pad_inches=0.25, dpi=300)
    print(f"Saved {globalConfig.nameTitle} attitude error map to {outputFile2}")

    # Show plot
    # plt.show()


if __name__ == "__main__":
    planet = globalConfig.planet

    # Star tracker "measurements" file
    n = len(sys.argv)
    measurements = globalConfig.outputDir + "output_" + planet.planetName.title() + ".csv"
    
    # Truth source directory
    truthData = globalConfig.outputDir + "truth_data_" + planet.planetName.title() + ".csv"
    
    # Latitude, longitude, and altitude
    latLonData = globalConfig.outputDir + "sampleLatLongs_" + planet.planetName.title() + ".csv"
    
    # Cutoff for "good measurements" in arcseconds
    reject = 0.0  # If 0, no measurement cutoff

    plot_errors(truthData, measurements, latLonData, reject, planet.radius)
