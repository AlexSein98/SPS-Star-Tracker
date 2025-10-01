from SPS.grav_earth_GGM05 import *
from SPS.grav_moon_GRAIL150 import *
from SPS.grav_mars_MRO110B2 import *
from py_src.star.python.transformations import *

import pyshtools as sh
import pyshtools.gravmag as grav
from pyshtools.gravmag import MakeGravGridPoint, MakeGravGradGridDH

import spiceypy as spice
import math
from decimal import Decimal


class Planet:
    def __init__(self, planetName: str, demName: str, demUnits: str, radius_m: float, planetFrame: str, gravModel: grav_base):
        self.planetName = planetName
        self.demName = demName
        self.demUnits = demUnits
        self.radius = radius_m
        self.planetFrame = planetFrame
        self.gravModel = gravModel


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


class GravSampler:
    def __init__(self, gravModel: grav_base,  maxDegree: int, maxOrder: int):
        self.maxDegree = maxDegree
        self.maxOrder = maxOrder
        self.Cilm = np.dstack((gravModel.Clm[:maxDegree+1, :maxOrder+1], gravModel.Slm[:maxDegree+1, :maxOrder+1])).transpose((2, 0, 1))
        self.Cilm[0, 0, 0] = 1.0  # add spherical component of gravity
        self.Cilm[0, 2, 0] = 0.0  # take out J2 (TODO: don't?)
        self.gravModel = gravModel
    
    def SampleAcceleration(self, lat: float, lon: float, r: float, maxDegree: float, 
                           overrideSphericalHarmonics: bool = False, noRadialTerm: bool = False,
                           includeThirdBody: bool = False, et: float = 0.0) -> np.ndarray[float]:
        """
        Latitude and longitude must be in degrees!
        """
        mu = self.gravModel.mu
        R = self.gravModel.radius
        omega = self.gravModel.omega

        T = latlon_to_T(lat, lon)
        pos_surface = r * T[:, 0]
        g_pcpf = T @ (MakeGravGridPoint(self.Cilm, mu, R, r, lat, lon, maxDegree, omega))
        # g_pcpf = T @ (MakeGravGridPoint(self.Cilm, mu, R, r, lat, lon, maxDegree, 0.0))
        
        # Choose to override radial term (or not); used for gradient calculations
        if noRadialTerm:
            g_pcpf += self.gravModel.mu * pos_surface / (r ** 3)
            return g_pcpf

        # Choose to override spherical harmonics with just spherical gravity (or not):
        if overrideSphericalHarmonics:
            g_pcpf = -self.gravModel.mu * pos_surface / (r ** 3)

        # Choose to do third-body perturbations (or not):
        if not includeThirdBody:
            return g_pcpf
        
        planetIDs = [10, 199, 299, 301, 399, 499, 599, 699, 799, 899]  # no Pluto hehe >:)
        T_centralBody = spice.pxform("J2000", self.gravModel.spiceBodyFrame, et)
        for id in planetIDs:
            # Assume SPICE has been furnsh'd
            name: str = spice.bodc2n(id)
            if name == self.gravModel.name:
                continue  # No third-body gravity if this is the central body

            pos_km, _ = spice.spkpos(name, et, "J2000", "NONE", self.gravModel.name)
            pos_m = 1000.0 * pos_km

            muResult = spice.bodvrd(name, "GM", 1)
            mu: float = muResult[1][0] * 1e9
            r_sat_body: np.ndarray[float] = pos_m - T_centralBody.T @ pos_surface
            g_inertial = mu * (r_sat_body / (np.linalg.norm(r_sat_body) ** 3) - pos_m / (np.linalg.norm(pos_m) ** 3))
            g_pcpf += T_centralBody @ g_inertial
        return g_pcpf
    
    def SampleAcceleration_Custom(self, lat: float, lon: float, r: float, maxDegree: float, 
                                  overrideSphericalHarmonics: bool = False, noRadialTerm: bool = False,
                                  includeThirdBody: bool = False, et: float = 0.0) -> np.ndarray[float]:
        """
        Latitude and longitude must be in degrees!
        """
        mu = self.gravModel.mu
        # R = self.gravModel.radius
        # omega = self.gravModel.omega
        # Omega = np.array([0.0, 0.0, omega])

        T = latlon_to_T(lat, lon)
        posSurface = r * T[:, 0]
        g_pcpf = self.HarmonicAcceleration(posSurface, self.gravModel, lat, lon, maxDegree, maxDegree)

        g_pcpf -= self.gravModel.mu * posSurface / (r ** 3)

        # if includeAngVel:
        #     g_pcpf += np.cross(Omega, np.cross(Omega, posSurface))
        
        # Choose to override radial term (or not); used for gradient calculations
        if noRadialTerm:
            g_pcpf += self.gravModel.mu * posSurface / (r ** 3)
            return g_pcpf

        # Choose to override spherical harmonics with just spherical gravity (or not):
        if overrideSphericalHarmonics:
            g_pcpf = -self.gravModel.mu * posSurface / (r ** 3)

        # Choose to do third-body perturbations (or not):
        if not includeThirdBody:
            return g_pcpf
        
        planetIDs = [10, 199, 299, 301, 399, 499, 599, 699, 799, 899]  # no Pluto hehe >:)
        T_centralBody = spice.pxform("J2000", self.gravModel.spiceBodyFrame, et)
        for id in planetIDs:
            # Assume SPICE has been furnsh'd
            name: str = spice.bodc2n(id)
            if name == self.gravModel.name:
                continue  # No third-body gravity if this is the central body

            pos_km, _ = spice.spkpos(name, et, "J2000", "NONE", self.gravModel.name)
            pos_m = 1000.0 * pos_km

            muResult = spice.bodvrd(name, "GM", 1)
            mu: float = muResult[1][0] * 1e9
            r_sat_body: np.ndarray[float] = pos_m - T_centralBody.T @ posSurface
            g_inertial = mu * (r_sat_body / (np.linalg.norm(r_sat_body) ** 3) - pos_m / (np.linalg.norm(pos_m) ** 3))
            g_pcpf += T_centralBody @ g_inertial
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
        latPlusX, lonPlusX, altPlusX = r_to_latlonalt(rPlusX, self.gravModel.radius)
        latMinusX, lonMinusX, altMinusX = r_to_latlonalt(rMinusX, self.gravModel.radius)
        latPlusY, lonPlusY, altPlusY = r_to_latlonalt(rPlusY, self.gravModel.radius)
        latMinusY, lonMinusY, altMinusY = r_to_latlonalt(rMinusY, self.gravModel.radius)
        latPlusZ, lonPlusZ, altPlusZ = r_to_latlonalt(rPlusZ, self.gravModel.radius)
        latMinusZ, lonMinusZ, altMinusZ = r_to_latlonalt(rMinusZ, self.gravModel.radius)

        # LatLonAlt structs
        llaPlusX = LatLonAlt(latPlusX, lonPlusX, altPlusX)
        llaMinusX = LatLonAlt(latMinusX, lonMinusX, altMinusX)
        llaPlusY = LatLonAlt(latPlusY, lonPlusY, altPlusY)
        llaMinusY = LatLonAlt(latMinusY, lonMinusY, altMinusY)
        llaPlusZ = LatLonAlt(latPlusZ, lonPlusZ, altPlusZ)
        llaMinusZ = LatLonAlt(latMinusZ, lonMinusZ, altMinusZ)

        return DeltaLatLon(lla, llaPlusX, llaMinusX, llaPlusY, llaMinusY, llaPlusZ, llaMinusZ)
    
    def SampleGradient_Numerical(self, xyz: np.ndarray[float], maxDegree: float, dXYZ: float) -> np.ndarray[float]:
        """
        Latitude and longitude must be in degrees! dXYZ must be in the same units as r.
        """
        # The gradient matrix is of form: 
        #   | dg/dx[0], dg/dy[0], dg/dz[0] |
        #   | dg/dx[1], dg/dy[1], dg/dz[1] |
        #   | dg/dx[2], dg/dy[2], dg/dz[2] |

        xyz_pX = xyz + np.array([dXYZ, 0.0, 0.0])
        xyz_mX = xyz - np.array([dXYZ, 0.0, 0.0])
        xyz_pY = xyz + np.array([0.0, dXYZ, 0.0])
        xyz_mY = xyz - np.array([0.0, dXYZ, 0.0])
        xyz_pZ = xyz + np.array([0.0, 0.0, dXYZ])
        xyz_mZ = xyz - np.array([0.0, 0.0, dXYZ])

        lat_pX, lon_pX, alt_pX = r_to_latlonalt(xyz_pX, self.gravModel.radius)
        lat_mX, lon_mX, alt_mX = r_to_latlonalt(xyz_mX, self.gravModel.radius)
        lat_pY, lon_pY, alt_pY = r_to_latlonalt(xyz_pY, self.gravModel.radius)
        lat_mY, lon_mY, alt_mY = r_to_latlonalt(xyz_mY, self.gravModel.radius)
        lat_pZ, lon_pZ, alt_pZ = r_to_latlonalt(xyz_pZ, self.gravModel.radius)
        lat_mZ, lon_mZ, alt_mZ = r_to_latlonalt(xyz_mZ, self.gravModel.radius)

        # phi_pg_pX, _, _ = cartesian_to_planetographic(xyz_pX, self.gravModel.radius, self.gravModel.polarRadius)
        # phi_pg_mX, _, _ = cartesian_to_planetographic(xyz_mX, self.gravModel.radius, self.gravModel.polarRadius)
        # phi_pg_pY, _, _ = cartesian_to_planetographic(xyz_pY, self.gravModel.radius, self.gravModel.polarRadius)
        # phi_pg_mY, _, _ = cartesian_to_planetographic(xyz_mY, self.gravModel.radius, self.gravModel.polarRadius)
        # phi_pg_pZ, _, _ = cartesian_to_planetographic(xyz_pZ, self.gravModel.radius, self.gravModel.polarRadius)
        # phi_pg_mZ, _, _ = cartesian_to_planetographic(xyz_mZ, self.gravModel.radius, self.gravModel.polarRadius)

        dg_dx_p = self.SampleAcceleration_Custom(lat_pX, lon_pX, alt_pX + self.gravModel.radius, maxDegree)
        dg_dx_m = self.SampleAcceleration_Custom(lat_mX, lon_mX, alt_mX + self.gravModel.radius, maxDegree)
        dg_dy_p = self.SampleAcceleration_Custom(lat_pY, lon_pY, alt_pY + self.gravModel.radius, maxDegree)
        dg_dy_m = self.SampleAcceleration_Custom(lat_mY, lon_mY, alt_mY + self.gravModel.radius, maxDegree)
        dg_dz_p = self.SampleAcceleration_Custom(lat_pZ, lon_pZ, alt_pZ + self.gravModel.radius, maxDegree)
        dg_dz_m = self.SampleAcceleration_Custom(lat_mZ, lon_mZ, alt_mZ + self.gravModel.radius, maxDegree)
        
        # phi_pc, lon, _ = r_to_latlonalt(xyz, self.gravModel.radius)
        # phi_pg, _, alt = cartesian_to_planetographic(xyz, self.gravModel.radius, self.gravModel.polarRadius)

        # lat, lon, alt = r_to_latlonalt(xyz, self.gravModel.radius)
        # deltaLatLon = self.DeltaXYZ_to_DeltaLatLon(LatLonAlt(lat, lon, alt), self.gravModel.radius, dXYZ)
        
        # llaPlusX = deltaLatLon.llaPlusX
        # llaMinusX = deltaLatLon.llaMinusX
        # llaPlusY = deltaLatLon.llaPlusY
        # llaMinusY = deltaLatLon.llaMinusY
        # llaPlusZ = deltaLatLon.llaPlusZ
        # llaMinusZ = deltaLatLon.llaMinusZ
        
        # dg_dx_plus = self.SampleAcceleration(llaPlusX.lat, llaPlusX.lon, llaPlusX.alt + self.gravModel.radius, maxDegree, noRadialTerm=False)
        # dg_dx_minus = self.SampleAcceleration(llaMinusX.lat, llaMinusX.lon, llaMinusX.alt + self.gravModel.radius, maxDegree, noRadialTerm=False)
        # dg_dy_plus = self.SampleAcceleration(llaPlusY.lat, llaPlusY.lon, llaPlusY.alt + self.gravModel.radius, maxDegree, noRadialTerm=False)
        # dg_dy_minus = self.SampleAcceleration(llaMinusY.lat, llaMinusY.lon, llaMinusY.alt + self.gravModel.radius, maxDegree, noRadialTerm=False)
        # dg_dz_plus = self.SampleAcceleration(llaPlusZ.lat, llaPlusZ.lon, llaPlusZ.alt + self.gravModel.radius, maxDegree, noRadialTerm=False)
        # dg_dz_minus = self.SampleAcceleration(llaMinusZ.lat, llaMinusZ.lon, llaMinusZ.alt + self.gravModel.radius, maxDegree, noRadialTerm=False)

        dXYZ_inv = 0.5 / dXYZ
        dg_dx: np.ndarray[float] = dXYZ_inv * (dg_dx_p - dg_dx_m)
        dg_dy: np.ndarray[float] = dXYZ_inv * (dg_dy_p - dg_dy_m)
        dg_dz: np.ndarray[float] = dXYZ_inv * (dg_dz_p - dg_dz_m)

        return np.array([dg_dx, dg_dy, dg_dz]).T
    
    def GetGradientGrid(self, maxDegree: float) -> np.ndarray[float]:
        mu = self.gravModel.mu
        R = self.gravModel.radius
        f = self.gravModel.flattening

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
        # G_ij = np.array([[Gzz_ij, Gxz_ij, Gyz_ij], 
        #                  [Gxz_ij, Gxx_ij, Gxy_ij], 
        #                  [Gyz_ij, Gxy_ij, Gyy_ij]])
        G_ij = np.array([[Gzz_ij, -Gyz_ij, Gxz_ij], 
                         [-Gyz_ij, -Gyy_ij, -Gxy_ij], 
                         [Gxz_ij, -Gxy_ij, Gxx_ij]])
        return T @ G_ij @ T.T
    
    def NormalizationCoefficient(self, l: int, m: int):
        k: int = 2
        if m == 0:
            k = 1
        return np.sqrt(float(Decimal(math.factorial(l + m)) / Decimal((math.factorial(l - m) * k * ((2 * l) + 1)))))

    def GetDimensionalZonalHarmonic(self, l, m, Clm):
        PI_lm = self.NormalizationCoefficient(l, m)
        return Clm[l][m] / PI_lm

    def CalculateAccelerationJ2(self, J2: float, gravModel: grav_base, r: np.ndarray[float]) -> np.ndarray[float]:
        x_r: float = r[0]
        y_r: float = r[1]
        z_r: float = r[2]
        r_norm: float = np.linalg.norm(r)
        a_x: float = (-3 * J2 * gravModel.mu * (gravModel.radius ** 2) * x_r) * (1 - ((5 * z_r ** 2) / (r_norm ** 2)) ) / (2 * r_norm ** 5)
        a_y: float = (-3 * J2 * gravModel.mu * (gravModel.radius ** 2) * y_r) * (1 - ((5 * z_r ** 2) / (r_norm ** 2))  ) / (2 * r_norm ** 5)
        a_z: float = (-3 * J2 * gravModel.mu * (gravModel.radius ** 2) * z_r) * (3 - ((5 * z_r ** 2) / (r_norm ** 2))  ) / (2 * r_norm ** 5)
        a = np.array([a_x, a_y, a_z])
        return a

    def HarmonicAcceleration(self, r: np.ndarray[float], gravModel: grav_base, 
                             lat: float, lon: float, degree: int, order: int) -> np.ndarray[float]:
        # initialize P matrix:
        size = degree + 1
        P = np.zeros((size, size))

        lat_r = np.deg2rad(lat)
        lon_r = np.deg2rad(lon)

        R = gravModel.radius
        mu = gravModel.mu

        r_norm = np.linalg.norm(r)
        sinLat = np.sin(lat_r)
        cosLat = np.cos(lat_r)
        tanLat = np.tan(lat_r)
        P[0][0] = 1.0
        P[1][0] = sinLat
        P[1][1] = cosLat

        for l in range(2, size):
            for m in range(0, size):
                if m == 0 and l >= 2:
                    P[l][m] = ((2 * l - 1) * sinLat * P[l - 1][0] - (l - 1) * P[l - 2][0]) / l
                elif m != 0 and m < l:
                    P[l][m] = P[l - 2][m] + (2 * l - 1) * cosLat * P[l - 1][m - 1]
                elif l != 0 and m == l:
                    P[l][m] = (2 * l - 1) * cosLat * P[l - 1][l - 1]

        dUdr = 0.0
        dUdLat = 0.0
        dUdLon = 0.0
        for l in range(2, size):
            for m in range(0, min(l + 1, order + 1)):
                PI_lm = self.NormalizationCoefficient(l, m)
                Clm = gravModel.Clm[l][m] / PI_lm
                Slm = gravModel.Slm[l][m] / PI_lm

                Plm1 = 0.0 ###
                if m < l:
                    Plm1 = P[l][m + 1]

                dUdr += ((R / r_norm) ** l) * (l + 1) * P[l][m] * \
                        (Clm * np.cos(m * lon_r) + Slm * np.sin(m * lon_r))

                dUdLat += ((R / r_norm) ** l) * (Plm1 - m * tanLat * P[l][m]) * \
                        (Clm * np.cos(m * lon_r) + Slm * np.sin(m * lon_r))

                dUdLon += ((R / r_norm) ** l) * m * P[l][m] * \
                        (Slm * np.cos(m * lon_r) - Clm * np.sin(m * lon_r))

        dUdr *= -mu / (r_norm ** 2)
        dUdLat *= mu / r_norm
        dUdLon *= mu / r_norm

        r_squared = r_norm ** 2
        rho_squared = r[0] ** 2 + r[1] ** 2
        rho = np.sqrt(rho_squared)
        a_x = (dUdr / r_norm - r[2] * dUdLat / (r_squared * rho)) * r[0] - (dUdLon / rho_squared) * r[1]
        a_y = (dUdr / r_norm - r[2] * dUdLat / (r_squared * rho)) * r[1] + (dUdLon / rho_squared) * r[0]
        a_z = (dUdr / r_norm) * r[2] + (rho * dUdLat / r_squared)

        return np.array([a_x, a_y, a_z])
