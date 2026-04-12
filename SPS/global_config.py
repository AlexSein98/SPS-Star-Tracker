import numpy as np
from SPS.gravity import *


class GlobalConfig:
    def __init__(self, planet: Planet, numLat: int, numLon: int, grav_maxDegree: int, grav_maxOrder: int):
        self.planet = planet
        self.numLat = numLat
        self.numLon = numLon
        self.grav_maxDegree = grav_maxDegree
        self.grav_maxOrder = grav_maxOrder

# Set variables necessary for sample_gravity function
_earthGrav: grav_base = grav_earth_GGM05()
_moonGrav: grav_base = grav_moon_GRAIL150()
_marsGrav: grav_base = grav_mars_MRO110B2()
_phobosGrav: grav_base = grav_phobos_basic()

_earth = Planet("EARTH", "./data/Earth_1arcmin.tif", "m", 6378136.3, "ITRF93", _earthGrav)
_moon = Planet("MOON", "./data/ldem_64.tif", "km", 1737400.0, "MOON_PA", _moonGrav)
_mars = Planet("MARS", "./data/Mars_global_463m.tif", "m", 3396190.0, "IAU_MARS", _marsGrav)
_phobos = Planet("PHOBOS", "./data/Phobos_2ppd.tif", "m", 10993.0, "IAU_PHOBOS", _phobosGrav)
_planets = [_earth, _moon, _mars, _phobos]

##################################
####    SELECT PLANET HERE    ####
# _planetIdx = 0  # Earth
_planetIdx = 1  # Moon
# _planetIdx = 2  # Mars
# _planetIdx = 3  # Phobos
##################################

_planet = _planets[_planetIdx]

_numLon: int = 36
_numLat: int = int(0.5 * _numLon - 1)

_grav_maxDegree: int = 2
_grav_maxOrder: int = 0

globalConfig = GlobalConfig(_planet, _numLat, _numLon, _grav_maxDegree, _grav_maxOrder)
