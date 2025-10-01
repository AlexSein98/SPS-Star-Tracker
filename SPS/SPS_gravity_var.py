from py_src.star.python.transformations import *

from matplotlib import pyplot as plt
import numpy as np
import spiceypy as spice

from pyshtools.gravmag import MakeGravGridPoint

from SPS.SPS_samples import *
from SPS.global_config import *


def angle_between_gravities(real_grav: np.ndarray[float], sphere_grav: np.ndarray[float]) -> float:
    return arccos_safe(np.dot(normalize(real_grav), normalize(sphere_grav)))


def difference_of_gravities(real_grav: np.ndarray[float], sphere_grav: np.ndarray[float]) -> float:
    sign = 1 if np.linalg.norm(real_grav) >= np.linalg.norm(sphere_grav) else -1
    return sign * np.linalg.norm(real_grav - sphere_grav)


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
            if planet.planetName == "EARTH":
                altitude_m = max(altitude_m, 0.0)
            
            lats[i].append(lat)
            lons[i].append(lon)
            alts[i].append(altitude_m)
    
    angle_matrix: list[list[float]] = []
    diff_matrix: list[list[float]] = []
    maxDegree: int = globalConfig.grav_maxDegree
    maxOrder: int = globalConfig.grav_maxOrder
    sampler = GravSampler(gravModel, maxDegree, maxOrder)

    radiusEquatorial: float = gravModel.radius
    radiusPolar: float = gravModel.polarRadius

    for i in range(len(lats)):
        angle_matrix.append([])
        diff_matrix.append([])
        for j in range(len(lats[i])):
            phi_pg: float = lats[i][j]
            lon: float = lons[i][j]
            h_ellp: float = alts[i][j]
            
            # Planetographic coordinates instead of planetocentric
            xyz = planetographic_to_cartesian(phi_pg, lon, h_ellp, radiusEquatorial, radiusPolar)
            # sphere_grav = -normalize(xyz) * gravModel.mu / (np.linalg.norm(xyz) ** 2)
            sphere_grav = -normalize(xyz) * gravModel.mu / (radiusEquatorial ** 2)

            # Gravity is sampled with planetocentric coordinates though
            phi_pc, _, _ = r_to_latlonalt(xyz, radiusEquatorial)
            # phi_pc = latitude_pg_to_pc(phi_pg, radiusEquatorial, radiusPolar)
            true_grav = sampler.SampleAcceleration_Custom(phi_pc, lon, np.linalg.norm(xyz), maxDegree,
                                                          overrideSphericalHarmonics=False, includeThirdBody=True, et=etNow)
            _angle = angle_between_gravities(true_grav, sphere_grav)
            _diff = difference_of_gravities(true_grav, sphere_grav)
            angle_matrix[i].append(_angle)
            diff_matrix[i].append(_diff)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    print(f'Max deviation = {np.max(deg_to_arcsec(np.rad2deg(np.asarray(angle_matrix))))}"')

    # cmap options: [RdYlGn_r, rainbow]
    plt.sca(ax1)
    plt.imshow(deg_to_arcsec(np.rad2deg(np.asarray(angle_matrix))), cmap='rainbow', aspect='equal', extent = [-180,180,-90,90])
    plt.colorbar(label='Gravity Vector Difference (arcsec)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Error Between Spherical Harmonic and Pure Spherical Gravity Models")
    # plt.title("Error Between 3rd Body Perturbation and Pure Spherical Gravity Models")

    plt.sca(ax2)
    plt.imshow(diff_matrix, cmap='rainbow', aspect='equal', extent = [-180,180,-90,90])
    plt.colorbar(label='Gravity Vector Difference (m/s^2)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Error Between Spherical Harmonic and Pure Spherical Gravity Models")
    # plt.title("Error Between 3rd Body Perturbation and Pure Spherical Gravity Models")

    plt.show()
