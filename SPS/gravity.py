from SPS.grav_moon_GRAIL150 import *
from py_src.star.python.transformations import *

import pyshtools as sh
import pyshtools.gravmag as grav
from pyshtools.gravmag import MakeGravGridPoint, MakeGravGradGridDH

import spiceypy as spice


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
    
    def SampleAcceleration(self, lat: float, lon: float, r: float, maxDegree: float, 
                           includeThirdBody: bool = False, et: float = 0.0) -> np.ndarray[float]:
        """
        Latitude and longitude must be in degrees!
        """
        mu = moon.mu
        R = moon.radius
        omega = moon.omega

        T = latlon_to_T(lat, lon)
        pos_surface = r * T[:, 0]
        g_pcpf = T @ MakeGravGridPoint(self.Cilm, mu, R, r, lat, lon, maxDegree, omega)

        # Third-body perturbations:
        if not includeThirdBody:
            return g_pcpf

        # g_pcpf = -moon.mu * pos_surface / (r ** 3)  # Override spherical harmonics with just spherical gravity
        planetIDs = [10, 199, 299, 399, 499, 599, 699, 799, 899]  # no Pluto hehe >:)
        T_moon = spice.pxform("J2000", "MOON_PA", et)
        for id in planetIDs:
            # Assume SPICE has been furnsh'd
            name: str = spice.bodc2n(id)
            pos_km, _ = spice.spkpos(name, et, "J2000", "NONE", "MOON")
            pos_m = 1000.0 * pos_km

            muResult = spice.bodvrd(name, "GM", 1)
            mu: float = muResult[1][0]
            r_sat_body: np.ndarray[float] = pos_m - T_moon.T @ pos_surface
            g_inertial = mu * (r_sat_body / (np.linalg.norm(r_sat_body) ** 3) - pos_m / (np.linalg.norm(pos_m) ** 3))
            g_pcpf += T_moon @ g_inertial
        return g_pcpf
    
    def DeltaXYZ_to_DeltaLatLon(self, lla: LatLonAlt, R: float, dXYZ: float) -> DeltaLatLon:
        """
        Latitude and longitude must be in degrees! dXYZ must be in the same units as R.
        """
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
        latPlusX, lonPlusX, altPlusX = r_to_latlonalt(rPlusX, moon.radius)
        latMinusX, lonMinusX, altMinusX = r_to_latlonalt(rMinusX, moon.radius)
        latPlusY, lonPlusY, altPlusY = r_to_latlonalt(rPlusY, moon.radius)
        latMinusY, lonMinusY, altMinusY = r_to_latlonalt(rMinusY, moon.radius)
        latPlusZ, lonPlusZ, altPlusZ = r_to_latlonalt(rPlusZ, moon.radius)
        latMinusZ, lonMinusZ, altMinusZ = r_to_latlonalt(rMinusZ, moon.radius)

        # LatLonAlt structs
        llaPlusX = LatLonAlt(latPlusX, lonPlusX, altPlusX)
        llaMinusX = LatLonAlt(latMinusX, lonMinusX, altMinusX)
        llaPlusY = LatLonAlt(latPlusY, lonPlusY, altPlusY)
        llaMinusY = LatLonAlt(latMinusY, lonMinusY, altMinusY)
        llaPlusZ = LatLonAlt(latPlusZ, lonPlusZ, altPlusZ)
        llaMinusZ = LatLonAlt(latMinusZ, lonMinusZ, altMinusZ)

        return DeltaLatLon(lla, llaPlusX, llaMinusX, llaPlusY, llaMinusY, llaPlusZ, llaMinusZ)
    
    def SampleGradient_Numerical(self, lat: float, lon: float, r: float, maxDegree: float, dXYZ: float) -> np.ndarray[float]:
        """
        Latitude and longitude must be in degrees! dXYZ must be in the same units as r.
        """
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
    
    def GetGradientGrid(self, maxDegree: float) -> np.ndarray[float]:
        mu = moon.mu
        R = moon.radius
        f = moon.flattening

        Gxx, Gyy, Gzz, Gxy, Gxz, Gyz = \
            MakeGravGradGridDH(self.Cilm, mu, R, lmax=maxDegree, a=R, f=f, sampling=2, lmax_calc=maxDegree, extend=True)
        return Gxx, Gyy, Gzz, Gxy, Gxz, Gyz
    
    def InterpolateGrid(self, grid: np.ndarray[float], _i_minus: int, _i_plus: int, _j_minus: int, _j_plus: int) -> float:
        """
        Internal use only!
        """
        sample_i_minus_j_minus = grid[_i_minus, _j_minus]
        sample_i_plus_j_minus = grid[_i_plus, _j_minus]
        sample_i_minus_j_plus = grid[_i_minus, _j_plus]
        sample_i_plus_j_plus = grid[_i_plus, _j_plus]
        return 0.25 * (sample_i_minus_j_minus + sample_i_plus_j_minus + sample_i_minus_j_plus + sample_i_plus_j_plus)
    
    def InterpolateGradientGrid(self, lat: float, lon: float, Gxx: np.ndarray[float], 
                                Gyy: np.ndarray[float], Gzz: np.ndarray[float], 
                                Gxy: np.ndarray[float], Gxz: np.ndarray[float], 
                                Gyz: np.ndarray[float]) -> np.ndarray[float]:
        countLat = Gxx.shape[0]
        countLon = Gxx.shape[1]
        _i = countLat * (90.0 - lat) / 180.0
        _j = countLon * (lon) / 360.0

        _i_minus: int = int(np.floor(_i))
        _i_plus: int = int((_i_minus + 1) % countLat)
        _j_minus: int = int(np.floor(_j))
        _j_plus: int = int((_j_minus + 1) % countLon)
        
        Gxx_ij = self.InterpolateGrid(Gxx, _i_minus, _i_plus, _j_minus, _j_plus)
        Gyy_ij = self.InterpolateGrid(Gyy, _i_minus, _i_plus, _j_minus, _j_plus)
        Gzz_ij = self.InterpolateGrid(Gzz, _i_minus, _i_plus, _j_minus, _j_plus)
        Gxy_ij = self.InterpolateGrid(Gxy, _i_minus, _i_plus, _j_minus, _j_plus)
        Gxz_ij = self.InterpolateGrid(Gxz, _i_minus, _i_plus, _j_minus, _j_plus)
        Gyz_ij = self.InterpolateGrid(Gyz, _i_minus, _i_plus, _j_minus, _j_plus)
        
        T: np.ndarray[float] = latlon_to_T(lat, lon)
        G_ij = np.array([[Gzz_ij, Gxz_ij, Gyz_ij], 
                         [Gxz_ij, Gxx_ij, Gxy_ij], 
                         [Gyz_ij, Gxy_ij, Gyy_ij]])
        return T @ G_ij @ T.T
