import os
import csv

import PIL.Image as Image
import matplotlib.pyplot as plt

from py_src.star.python.transformations import *
from SPS.gravity import Planet

from SPS.global_config import *


Image.MAX_IMAGE_PIXELS = None


def ReadDEM(path: str) -> np.ndarray[float]:
    dem = np.asarray(Image.open(path))
    return dem


def SampleDEM(dem: np.ndarray[float], _i: float, _j: float) -> float:
    _i_minus: int = int(np.floor(_i))
    _i_plus: int = int((_i_minus + 1) % dem.shape[0])
    _j_minus: int = int(np.floor(_j))
    _j_plus: int = int((_j_minus + 1) % dem.shape[1])

    sample_i_minus_j_minus = dem[_i_minus, _j_minus]
    sample_i_plus_j_minus = dem[_i_plus, _j_minus]
    sample_i_minus_j_plus = dem[_i_minus, _j_plus]
    sample_i_plus_j_plus = dem[_i_plus, _j_plus]
    return 0.25 * (sample_i_minus_j_minus + sample_i_plus_j_minus + sample_i_minus_j_plus + sample_i_plus_j_plus)


def SampleGlobalDEM_LatLon(dem: np.ndarray[float], lat: float, lon: float) -> float:
    return SampleLocalDEM_LatLon(dem, lat, lon, [-90.0, 90.0, -180.0, 180.0])


def SampleLocalDEM_LatLon(dem: np.ndarray[float], lat: float, lon: float, demLimits: list[float]) -> float:
    minLat = demLimits[0]
    maxLat = demLimits[1]
    minLon = demLimits[2]
    maxLon = demLimits[3]

    countLat = dem.shape[0]
    countLon = dem.shape[1]
    _i = countLat * (lat - minLat) / (maxLat - minLat)
    _j = countLon * (lon - minLon) / (maxLon - minLon)

    return SampleDEM(dem, _i, _j)


if __name__ == "__main__":
    os.system('cls')

    planet = globalConfig.planet

    numLon: int = globalConfig.numLon
    numLat: int = globalConfig.numLat
    
    dem = ReadDEM(planet.demName)
    countLat = dem.shape[0]
    countLon = dem.shape[1]

    invNumLat = 1.0 / (numLat + 1)
    invNumLon = 1.0 / numLon
    
    lats = []
    lons = []
    alts = []
    
    scaleFactor = 1000.0 if planet.demUnits == "km" else 1.0

    for i in range(numLat):
        lat = (i + 1) * 180.0 * invNumLat - 90.0
        latIdx = (i + 1) * countLat * invNumLat
        lats.append([])
        lons.append([])
        alts.append([])

        for j in range(numLon):
            lon = j * 360.0 * invNumLon - 180.0
            lonIdx = j * countLon * invNumLon
            altitude_m = scaleFactor * SampleDEM(dem, latIdx, lonIdx)

            if planet.planetName == "EARTH":
                altitude_m = max(altitude_m, 0.0)
            
            lats[i].append(lat)
            lons[i].append(lon)
            alts[i].append(altitude_m)
    
    with open("./sampleLatLongs_" + planet.planetName.title() + ".csv", "w") as samples:
        writer = csv.writer(samples, delimiter=",", quotechar="|", lineterminator="\n")
        for i in range(len(alts)):
            for j in range(len(alts[0])):
                writer.writerow([lats[i][j], lons[i][j], alts[i][j]])
    
    # Visualize:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # cmap options: [RdYlGn_r, rainbow]
    plt.imshow(alts, cmap='rainbow', aspect='equal', extent = [-180, 180, -90, 90])
    plt.colorbar(label='Altitude (m)')
    plt.savefig("./" + planet.planetName.title() + "Heightmap.png", bbox_inches='tight', transparent="True", pad_inches=0)

    plt.show()
