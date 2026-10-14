import copy
import numpy as np
import numpy.typing as npt


def arcsec_to_rad(arcsec: float) -> float:
    return np.deg2rad(arcsec / 3600.0)

'''
Converts planetographic (i.e., geodetic) latitude, longitude, and ellipsoid-referenced 
altitude to an equivalent Cartesian position vector in the planet-fixed frame.

Inputs:
lat:    geodetic latitude [degrees]
lon:    longitude [degrees]
alt:    ellipsoid-referenced altitude [meters]
a:      planet equatorial radius [meters]
b:      planet polar radius [meters]

Outputs:
[x, y, z]:  Cartesian position vector in planet-fixed frame [meters]
'''
def planetographic_to_cartesian(lat: float, lon: float, alt: float, a: float, b: float) -> npt.NDArray:
    lat_rad: float = np.deg2rad(lat)
    lon_rad: float = np.deg2rad(lon)
    N: float = a ** 2 / np.sqrt(a ** 2 * np.cos(lat_rad) ** 2 + b ** 2 * np.sin(lat_rad) ** 2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((b ** 2 / a ** 2) * N + alt) * np.sin(lat_rad)
    return np.array([x, y, z])


'''
Converts a Cartesian position vector in the planet-fixed frame to an equivalent 
planetographic (i.e., geodetic) latitude, longitude, and ellipsoid-referenced altitude.

Outputs:
xyz:    Cartesian position vector in planet-fixed frame [meters]
a:      planet equatorial radius [meters]
b:      planet polar radius [meters]

Outputs:
lat:    geodetic latitude [degrees]
lon:    longitude [degrees]
alt:    ellipsoid-referenced altitude [meters]
'''
def cartesian_to_planetographic(xyz: npt.NDArray, a: float, b: float) -> tuple[float, float, float]:
    # From Vallado, Fundamentals of Astrodynamics and Applications 4th edition, Algorithm 12: "ECEF To LatLon"
    x: float = xyz[0]
    y: float = xyz[1]
    z: float = xyz[2]
    r_delta: float = np.sqrt(x ** 2 + y ** 2)

    alpha: float = np.arctan2(y, x)  # This directly becomes longitude
    phi_pg: float = np.arctan(z / r_delta)
    ecc: float = np.sqrt(1.0 - (b / a) ** 2)
    C: float = a / np.sqrt(1.0 - ecc ** 2 * np.sin(phi_pg) ** 2)
    
    phi_pg_old: float = 2.0 * phi_pg
    while abs(phi_pg - phi_pg_old) > arcsec_to_rad(0.1):
        phi_pg_old = copy.deepcopy(phi_pg)
        C = a / np.sqrt(1.0 - ecc ** 2 * np.sin(phi_pg) ** 2)
        phi_pg = np.arctan((z + C * ecc ** 2 * np.sin(phi_pg)) / r_delta)
    
    h_ellp: float = r_delta / np.cos(phi_pg) - C
    if abs(phi_pg) > np.deg2rad(89.0):
        S: float = a * (1 - ecc ** 2) / np.sqrt(1.0 - ecc ** 2 * np.sin(phi_pg) ** 2)
        h_ellp = z / np.sin(phi_pg) - S
    
    lat: float = np.rad2deg(phi_pg)
    lon: float = np.rad2deg(alpha)
    alt: float = h_ellp
    return lat, lon, alt
