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

import scipy.stats as stats


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


def write_csv(filepath: str, data: list[np.ndarray[float]]):
    with open(filepath, "w") as dataCSV:
        writer = csv.writer(dataCSV, delimiter=',', quotechar='"', lineterminator='\n')
        for line in data:
            writer.writerow(line)


# class CustomDist(stats.rv_continuous):
#     def _pdf(self, x):
#         sigma: float = 1.0
#         g = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
#         return np.where((x >= 0) & (x <= 1), 2 * x, 0)


def GetStats(latLonDataPath: str, posErrorDataPath: str, planet: Planet, 
             gravModel: grav_base, xlim: list[float] = []):
    np.set_printoptions(suppress=True)

    dem = ReadDEM(planet.demName)
    latLonData = read_csv(latLonDataPath)
    posErrorData = read_csv(posErrorDataPath, hasHeader=False)

    northErrors: list[float] = []
    eastErrors: list[float] = []
    errorMags: list[float] = []

    L: int = len(latLonData)
    for j in range(L):
        posError_j = posErrorData[j]
        latLon_j = latLonData[j]

        latTruth = float(latLon_j[0])
        lonTruth = float(latLon_j[1])
        altTruth = float(latLon_j[2])

        xErr = float(posError_j[2])
        yErr = float(posError_j[3])
        zErr = float(posError_j[4])
        rErr_pfix = np.array([xErr, yErr, zErr])
        
        # Probably fine this way?
        # upEastNorth = latlon_to_T(r_hat_to_latlon(normalize(
        #     planetographic_to_cartesian(latTruth, lonTruth, altTruth, planet.radius, gravModel.polarRadius))))
        
        # ...but probably better this way?
        upEastNorth = latlon_to_T(latTruth, lonTruth)
        
        up = upEastNorth[:, 0]
        east = upEastNorth[:, 1]
        north = upEastNorth[:, 2]

        northErrors.append(np.dot(north, rErr_pfix))
        eastErrors.append(np.dot(east, rErr_pfix))
        errorMags.append(np.linalg.norm(rErr_pfix))
    
    # Shapiro-Wilk test
    shapiro_north = stats.shapiro(northErrors)
    shapiro_east = stats.shapiro(eastErrors)

    # Print statistics
    print("======================================================================================================")
    print(f'North-south errors: Gaussianity test (Shapiro-Wilk)')
    print(f'    Statistic = {shapiro_north.statistic}')
    print(f'    P-Value   = {shapiro_north.pvalue}')
    print(f'East-west errors: Gaussianity test (Shapiro-Wilk)')
    print(f'    Statistic = {shapiro_east.statistic}')
    print(f'    P-Value   = {shapiro_east.pvalue}\n')
    
    fig1 = plt.figure(layout='constrained')
    fig1.set_size_inches(9.6, 5.4)
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    ax1.hist(northErrors, bins=200, color='skyblue', edgecolor='black')
    ax2.hist(eastErrors, bins=200, color='skyblue', edgecolor='black')

    ax1.set_title(f'North-South Position Error Counts for {len(northErrors)} Samples')
    ax1.set_xlabel('South <---   Error (m)   ---> North')
    ax1.set_ylabel('Count')
    ax1.grid()
    # if xlim != []:
    #     ax1.set_xlim(xlim)

    ax2.set_title(f'East-West Position Error Counts for {len(northErrors)} Samples')
    ax2.set_xlabel('West <---   Error (m)   ---> East')
    ax2.set_ylabel('Count')
    ax2.grid()
    # if xlim != []:
    #     ax2.set_xlim(xlim)

    outputFile1 = globalConfig.outputDir + globalConfig.nameTitle + "PositionErrorDetailedStatistics.png"
    plt.savefig(outputFile1, bbox_inches='tight', facecolor='white', transparent="False", pad_inches=0.5, dpi=300)
    print(f"Saved {globalConfig.nameTitle} SPS detailed position error statistics to {outputFile1}")

    fig2 = plt.figure(layout='constrained')
    fig2.set_size_inches(9.6, 5.4)
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)

    # paramsNorth = stats.levy_stable.fit(northErrors)
    # paramsEast = stats.levy_stable.fit(eastErrors)

    df_fit_north, loc_fit_north, scale_fit_north = stats.t.fit(northErrors)
    df_fit_east, loc_fit_east, scale_fit_east = stats.t.fit(eastErrors)

    print(f"Student's t Params (North):")
    print(f"    df:    {df_fit_north}")
    print(f"    loc:   {loc_fit_north}")
    print(f"    scale: {scale_fit_north}\n")
    print(f"Student's t Params (East):")
    print(f"    df:    {df_fit_east}")
    print(f"    loc:   {loc_fit_east}")
    print(f"    scale: {scale_fit_east}\n")

    plt.sca(ax3)
    # stats.probplot(northErrors, dist="norm", plot=plt)
    stats.probplot(northErrors, dist="cauchy", plot=plt)
    # stats.probplot(northErrors, sparams=(4,), dist="t", plot=plt)
    ax3.set_title(f'North-South Position Error QQ Plot')
    ax3.set_xlabel("Theoretical Quantiles (Student's t Distribution)")
    ax3.set_ylabel("Sample Quantiles")
    ax3.grid()
    
    plt.sca(ax4)
    # stats.probplot(eastErrors, dist="norm", plot=plt)
    stats.probplot(eastErrors, dist="cauchy", plot=plt)
    # stats.probplot(eastErrors, sparams=(6,), dist="t", plot=plt)
    ax4.set_title(f'East-West Position Error QQ Plot')
    ax4.set_xlabel("Theoretical Quantiles (Student's t Distribution)")
    ax4.set_ylabel("Sample Quantiles")
    ax4.grid()

    outputFile2 = globalConfig.outputDir + globalConfig.nameTitle + "PositionErrorQQPlots.png"
    plt.savefig(outputFile2, bbox_inches='tight', facecolor='white', transparent="False", pad_inches=0.5, dpi=300)
    print(f"Saved {globalConfig.nameTitle} SPS position error QQ plots to {outputFile2}")

    fig3 = plt.figure(layout='constrained')
    fig3.set_size_inches(9.6, 5.4)
    ax5 = fig3.add_subplot(111)

    df_fit, loc_fit, scale_fit = stats.chi2.fit(errorMags, floc=0)

    plt.sca(ax5)
    stats.probplot(errorMags, dist=stats.chi2, sparams=(df_fit,), plot=plt)
    ax5.set_title(f'Position Error Magnitude QQ Plot')
    ax5.set_xlabel("Theoretical Quantiles (Log Normal Distribution)")
    ax5.set_ylabel("Sample Quantiles")
    ax5.grid()

    outputFile3 = globalConfig.outputDir + globalConfig.nameTitle + "PositionErrorMagnitudeQQPlot.png"
    plt.savefig(outputFile3, bbox_inches='tight', facecolor='white', transparent="False", pad_inches=0.5, dpi=300)
    print(f"Saved {globalConfig.nameTitle} SPS position error magnitude QQ plot to {outputFile3}")

    plt.show()


if __name__ == "__main__":
    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel

    # Latitude, longitude, and altitude
    latLonDataPath = globalConfig.outputDir + "sampleLatLongs_" + planet.planetName.title() + ".csv"
    posErrorDataPath = globalConfig.outputDir + "position_errors_" + planet.planetName.title() + ".csv"

    GetStats(posErrorDataPath, posErrorDataPath, planet, gravModel)#, xlim=[-5.0, 5.0])
