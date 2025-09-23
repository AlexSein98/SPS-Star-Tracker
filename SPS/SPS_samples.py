import os
import csv

import PIL.Image as Image
import matplotlib.pyplot as plt

from py_src.star.python.transformations import *


os.system('cls')

Image.MAX_IMAGE_PIXELS = None
moonDEM = "./data/ldem_64.tif"


def ReadDEM(path: str) -> np.ndarray[float]:
    dem = np.asarray(Image.open(path))
    return dem


def SampleDEM(dem: np.ndarray[float], _i: float, _j: float):
    _i_minus: int = int(np.floor(_i))
    _i_plus: int = int((_i_minus + 1) % dem.shape[0])
    _j_minus: int = int(np.floor(_j))
    _j_plus: int = int((_j_minus + 1) % dem.shape[1])

    sample_i_minus_j_minus = dem[_i_minus, _j_minus]
    sample_i_plus_j_minus = dem[_i_plus, _j_minus]
    sample_i_minus_j_plus = dem[_i_minus, _j_plus]
    sample_i_plus_j_plus = dem[_i_plus, _j_plus]
    return 0.25 * (sample_i_minus_j_minus + sample_i_plus_j_minus + sample_i_minus_j_plus + sample_i_plus_j_plus)


if __name__ == "__main__":
    rEarth = 6378136.3
    rMoon = 1737400.0
    rMars = 3396190.0
    rPhobos = 11000.0

    numLon: int = 24
    numLat: int = int(0.5 * numLon - 1)
    
    dem = ReadDEM(moonDEM)
    countLat = dem.shape[0]
    countLon = dem.shape[1]

    invNumLat = 1.0 / (numLat + 1)
    invNumLon = 1.0 / numLon
    
    lats = []
    lons = []
    alts = []
    for i in range(numLat):
        lat = (i + 1) * 180.0 * invNumLat - 90.0
        latIdx = (i + 1) * countLat * invNumLat
        lats.append([])
        lons.append([])
        alts.append([])

        for j in range(numLon):
            lon = j * 360.0 * invNumLon - 180.0
            lonIdx = j * countLon * invNumLon

            altitude_m = 1000.0 * SampleDEM(dem, latIdx, lonIdx)
            # print(f'altitude at lat = {round(lat, 6)}, lon = {round(lon, 6)}: {round(altitude_m, 2)} m')
            
            lats[i].append(lat)
            lons[i].append(lon)
            alts[i].append(altitude_m)
    
    with open("./sampleLatLongs.csv", "w") as samples:
        writer = csv.writer(samples, delimiter=",", quotechar="|", lineterminator="\n")
        for i in range(len(alts)):
            for j in range(len(alts[0])):
                writer.writerow([lats[i][j], lons[i][j], alts[i][j]])
    
    # Visualize:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.imshow(alts, cmap='viridis', aspect='equal')
    plt.colorbar(label='Altitude (m)')

    plt.show()
