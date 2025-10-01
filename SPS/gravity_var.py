from py_src.star.python.transformations import *

from matplotlib import pyplot as plt
import numpy as np
import spiceypy as spice

from pyshtools.gravmag import MakeGravGridPoint

from SPS.SPS_samples import *
from SPS.global_config import *


def angle_between_gravities(real_grav, sphere_grav):
   return arccos_safe(np.dot(normalize(real_grav), normalize(sphere_grav)))


if __name__ == "__main__":
    home = "./py_src/star/"
    spice.furnsh(home + "data/metakernel.txt")
    tNow = '2025 July 4, 00:00:00 UTC'
    etNow = spice.str2et(tNow)

    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel

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
        lat = 90.0 - (i + 1) * 180.0 * invNumLat
        latIdx = (i + 1) * countLat * invNumLat
        lats.append([])
        lons.append([])
        alts.append([])

        for j in range(numLon):
            lon = j * 360.0 * invNumLon - 180.0
            lonIdx = j * countLon * invNumLon

            altitude_m = scaleFactor * SampleDEM(dem, latIdx, lonIdx)
            lats[i].append(lat)
            lons[i].append(lon)
            alts[i].append(altitude_m)

    angle_matrix: list[list[float]] = []
    maxDegree: int = 100
    maxOrder: int = 100
    sampler = GravSampler(gravModel, maxDegree, maxOrder)

    for i in range(len(lats)):
      angle_matrix.append([])
      for j in range(len(lats[i])):
        sphere_grav = -normalize(latlon_to_T(lats[i][j], lons[i][j]).T[0])
        true_grav = sampler.SampleAcceleration(lats[i][j], lons[i][j], planet.radius + alts[i][j], maxDegree,
                                               overrideSphericalHarmonics=False, includeThirdBody=True, et=etNow)
        _angle = angle_between_gravities(true_grav, sphere_grav)
        angle_matrix[i].append(_angle)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    print(f'Max deviation = {np.max(deg_to_arcsec(np.rad2deg(np.asarray(angle_matrix))))}"')

    # cmap options: [RdYlGn_r, rainbow]
    plt.imshow(deg_to_arcsec(np.rad2deg(np.asarray(angle_matrix))), cmap='rainbow', aspect='equal', extent = [-180,180,-90,90])
    plt.colorbar(label='Gravity Vector Difference (arcsec)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Error Between Spherical Harmonic and Pure Spherical Gravity Models")
    # plt.title("Error Between 3rd Body Perturbation and Pure Spherical Gravity Models")
    plt.show()
