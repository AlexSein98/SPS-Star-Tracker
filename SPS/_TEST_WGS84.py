from py_src.star.python.transformations import *


lat_true: float = 28.626928
lon_true: float = -80.620856
alt_true: float = 4.0  # m
radius: float = 6378137.0  # m
radius_polar: float = 6356752.3  # m

xyz_true = planetographic_to_cartesian(lat_true, lon_true, alt_true, radius, radius_polar)
print(f"xyz_true = {xyz_true}")


def latitude_pc_to_pg(phi_pc: float, a: float, b: float) -> float:
    ecc: float = np.sqrt(1.0 - (b / a) ** 2)
    phi_pg: float = np.arctan(np.tan(np.deg2rad(phi_pc)) / (1.0 - ecc ** 2))
    return np.rad2deg(phi_pg)


def latitude_pg_to_pc(phi_pg: float, a: float, b: float) -> float:
    ecc: float = np.sqrt(1.0 - (b / a) ** 2)
    phi_pc: float = np.arctan(np.tan(np.deg2rad(phi_pg)) * (1.0 - ecc ** 2))
    return np.rad2deg(phi_pc)


lat_pc = latitude_pg_to_pc(lat_true, radius, radius_polar)
print(f"lat_pc = {lat_pc}")
