from SPS.grav_moon_GRAIL150 import *
from py_src.star.python.transformations import *

import PIL.Image as Image

import sys
import os
import csv
import copy

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import spiceypy as spice

import pyshtools as sh
import pyshtools.gravmag as grav
from pyshtools.gravmag import MakeGravGridPoint


def sample_gravity(moon: grav_moon_GRAIL150, Cilm, lat, lon, r, max_degree):
  mu = moon.mu
  R = moon.radius
  omega = moon.omega

  T = latlon_to_T(lat, lon)
  g_pcpf = T @ MakeGravGridPoint(Cilm, mu, R, r, lat, lon, max_degree, omega)
  return normalize(g_pcpf)




def angle_between_gravities(real_grav, sphere_grav):
  return np.arccos(np.dot(normalize(real_grav), normalize(sphere_grav)))




# Set variables necessary for sample_gravity function
moon = grav_moon_GRAIL150()

max_degree = 128
max_order = 128
Cilm = np.dstack((moon.Clm[:max_degree + 1, :max_order + 1], moon.Slm[:max_degree + 1, :max_order + 1])).transpose((2, 0, 1))
Cilm[0, 0, 0] = 1.0  # add spherical component of gravity



Image.MAX_IMAGE_PIXELS = None
moonDEM = ".\\data\\ldem_64.tif"


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

    numLon: int = 180
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
            lats[i].append(lat)
            lons[i].append(lon)
            alts[i].append(altitude_m)

    angle_matrix = []
    for i in range(len(lats)):
      angle_matrix.append([])
      for j in range(len(lats[i])):
        sphere_grav = -normalize(latlon_to_T(lats[i][j], lons[i][j]).T[0])
        true_grav = sample_gravity(moon,Cilm, lats[i][j], lons[i][j], moon.radius, max_degree)
        _angle = angle_between_gravities(true_grav, sphere_grav)
        angle_matrix[i].append(_angle)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.imshow(deg_to_arcsec(np.rad2deg(np.asarray(angle_matrix))), cmap='RdYlGn_r', aspect='equal', extent = [-180,180,-90,90])
    plt.colorbar(label='Gravity Vector Difference (arcsec)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Error Between Spherical Harmonic and Pure Spherical Gravity Models")
    plt.show()
