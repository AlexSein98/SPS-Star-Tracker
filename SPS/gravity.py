from SPS.grav_moon_GRAIL150 import *
from py_src.star.python.transformations import *

import pyshtools as sh
import pyshtools.gravmag as grav
from pyshtools.gravmag import MakeGravGridPoint


class LatLonAlt:
    def __init__(self, lat: float, lon: float, alt: float):
        self.lat = lat
        self.lon = lon
        self.alt = alt


class DeltaLatLon:
    def __init__(self, lla: LatLonAlt, llaPlusX: LatLonAlt, llaMinusX: LatLonAlt, 
                 llaPlusY: LatLonAlt, llaMinusY: LatLonAlt, llaPlusZ: LatLonAlt, llaMinusZ: LatLonAlt):
        self.lla = lla
        self.llaPlusX = llaPlusX
        self.llaMinusX = llaMinusX
        self.llaPlusY = llaPlusY
        self.llaMinusY = llaMinusY
        self.llaPlusZ = llaPlusZ
        self.llaMinusZ = llaMinusZ


moon = grav_moon_GRAIL150()

class GravSampler:
    def __init__(self, maxDegree: int, maxOrder: int):
        self.maxDegree = maxDegree
        self.maxOrder = maxOrder
        self.Cilm = np.dstack((moon.Clm[:maxDegree+1, :maxOrder+1], moon.Slm[:maxDegree+1, :maxOrder+1])).transpose((2, 0, 1))
        self.Cilm[0, 0, 0] = 1.0  # add spherical component of gravity
    
    """
    Latitude and longitude must be in degrees!
    """
    def SampleAcceleration(self, lat: float, lon: float, r: float, maxDegree: float) -> np.ndarray[float]:
        mu = moon.mu
        R = moon.radius
        omega = moon.omega

        T = latlon_to_T(lat, lon)
        g_pcpf = T @ MakeGravGridPoint(self.Cilm, mu, R, r, lat, lon, maxDegree, omega)
        return g_pcpf
    
    """
    Latitude and longitude must be in degrees! dXYZ must be in the same units as R.
    """
    def DeltaXYZ_to_DeltaLatLon(self, lla: LatLonAlt, R: float, dXYZ: float) -> DeltaLatLon:
        lat = lla.lat
        lon = lla.lon
        alt = lla.alt

        # Convert to position vector and apply offsets
        r = (R + alt) * (latlon_to_T(lat, lon).T[0])
        rPlusX = r + np.array([dXYZ, 0.0, 0.0])
        rMinusX = r - np.array([dXYZ, 0.0, 0.0])
        rPlusY = r + np.array([0.0, dXYZ, 0.0])
        rMinusY = r - np.array([0.0, dXYZ, 0.0])
        rPlusZ = r + np.array([0.0, 0.0, dXYZ])
        rMinusZ = r - np.array([0.0, 0.0, dXYZ])

        # Convert back to lat lon alt
        latPlusX, lonPlusX, altPlusX = r_to_latlonalt(rPlusX)
        latMinusX, lonMinusX, altMinusX = r_to_latlonalt(rMinusX)
        latPlusY, lonPlusY, altPlusY = r_to_latlonalt(rPlusY)
        latMinusY, lonMinusY, altMinusY = r_to_latlonalt(rMinusY)
        latPlusZ, lonPlusZ, altPlusZ = r_to_latlonalt(rPlusZ)
        latMinusZ, lonMinusZ, altMinusZ = r_to_latlonalt(rMinusZ)

        # LatLonAlt structs
        llaPlusX = LatLonAlt(latPlusX, lonPlusX, altPlusX)
        llaMinusX = LatLonAlt(latMinusX, lonMinusX, altMinusX)
        llaPlusY = LatLonAlt(latPlusY, lonPlusY, altPlusY)
        llaMinusY = LatLonAlt(latMinusY, lonMinusY, altMinusY)
        llaPlusZ = LatLonAlt(latPlusZ, lonPlusZ, altPlusZ)
        llaMinusZ = LatLonAlt(latMinusZ, lonMinusZ, altMinusZ)

        return DeltaLatLon(lla, llaPlusX, llaMinusX, llaPlusY, llaMinusY, llaPlusZ, llaMinusZ)
    
    """
    Latitude and longitude must be in degrees! dXYZ must be in the same units as r.
    """
    def SampleGradient(self, lat: float, lon: float, r: float, maxDegree: float, dXYZ: float) -> np.ndarray[float]:
        # The gradient matrix is of form: 
        #   | dg/dx[0], dg/dy[0], dg/dz[0] |
        #   | dg/dx[1], dg/dy[1], dg/dz[1] |
        #   | dg/dx[2], dg/dy[2], dg/dz[2] |
        
        alt = r - moon.radius
        deltaLatLon = self.DeltaXYZ_to_DeltaLatLon(LatLonAlt(lat, lon, alt), moon.radius, dXYZ)

        llaPlusX = deltaLatLon.llaPlusX
        llaMinusX = deltaLatLon.llaMinusX
        llaPlusY = deltaLatLon.llaPlusY
        llaMinusY = deltaLatLon.llaMinusY
        llaPlusZ = deltaLatLon.llaPlusZ
        llaMinusZ = deltaLatLon.llaMinusZ

        dg_dx_plus = self.SampleAcceleration(llaPlusX.lat, llaPlusX.lon, llaPlusX.alt + moon.radius, maxDegree)
        dg_dx_minus = self.SampleAcceleration(llaMinusX.lat, llaMinusX.lon, llaMinusX.alt + moon.radius, maxDegree)
        dg_dy_plus = self.SampleAcceleration(llaPlusY.lat, llaPlusY.lon, llaPlusY.alt + moon.radius, maxDegree)
        dg_dy_minus = self.SampleAcceleration(llaMinusY.lat, llaMinusY.lon, llaMinusY.alt + moon.radius, maxDegree)
        dg_dz_plus = self.SampleAcceleration(llaPlusZ.lat, llaPlusZ.lon, llaPlusZ.alt + moon.radius, maxDegree)
        dg_dz_minus = self.SampleAcceleration(llaMinusZ.lat, llaMinusZ.lon, llaMinusZ.alt + moon.radius, maxDegree)

        dXYZ_inv = 1.0 / dXYZ
        dg_dx: np.ndarray[float] = 0.5 * dXYZ_inv * (dg_dx_plus - dg_dx_minus)
        dg_dy: np.ndarray[float] = 0.5 * dXYZ_inv * (dg_dy_plus - dg_dy_minus)
        dg_dz: np.ndarray[float] = 0.5 * dXYZ_inv * (dg_dz_plus - dg_dz_minus)

        return np.array([dg_dx, dg_dy, dg_dz]).T
