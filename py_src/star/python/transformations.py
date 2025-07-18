import numpy as np


def T1(angle: float) -> np.ndarray[float]:
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(angle), np.sin(angle)],
                     [0.0, -np.sin(angle), np.cos(angle)]])
                     

def T2(angle: float) -> np.ndarray[float]:
    return np.array([[np.cos(angle), 0.0, -np.sin(angle)],
                     [0.0, 1.0, 0.0],
                     [np.sin(angle), 0.0, np.cos(angle)]])
                     

def T3(angle: float) -> np.ndarray[float]:
    return np.array([[np.cos(angle), np.sin(angle), 0.0],
                     [-np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def R1(angle: float) -> np.ndarray[float]:
    return T1(angle).T


def R2(angle: float) -> np.ndarray[float]:
    return T2(angle).T


def R3(angle: float) -> np.ndarray[float]:
    return T3(angle).T


def normalize(vec: np.ndarray[float]) -> np.ndarray[float]:
    mag = np.linalg.norm(vec)
    if mag > 0.0:
        return vec / np.linalg.norm(vec)
    else:
        return vec


def r_hat_to_ra_dec(r_hat: np.ndarray[float]):
    ra = np.rad2deg(np.arctan2(r_hat[1], r_hat[0]))
    dec = np.rad2deg(np.arcsin(r_hat[2] / np.linalg.norm(r_hat)))
    return ra, dec


def deg_to_hms(deg: float) -> tuple[int, int, float]:
    hours = deg / 15.0
    fracMinSec, hoursWhole = np.modf(hours)
    fracSec, minWhole = np.modf(fracMinSec * 60.0)
    fracSec *= 60.0
    return hoursWhole, minWhole, fracSec


def deg_to_hms_string(deg: float, decimalPlaces: int=3):
    h, m, s = deg_to_hms(deg)
    return f'{h} h {m} m {round(s, decimalPlaces)} s'


def deg_to_dms(deg: float) -> tuple[int, int, float]:
    fracMinSec, degWhole = np.modf(deg)
    fracSec, minWhole = np.modf(fracMinSec * 60.0)
    fracSec *= 60.0
    return degWhole, abs(minWhole), abs(fracSec)


def deg_to_dms_string(deg: float, decimalPlaces: int=3):
    d, m, s = deg_to_dms(deg)
    return f'{d} deg {m} arcmin {round(s, decimalPlaces)} arcsec'


def sec_to_year(sec: float):
    return sec / (86400.0 * 365.25)


def arcsec_to_rad(arcsec: float):
    return np.deg2rad(arcsec / 3600.0)


def marcsec_to_rad(marcsec: float):
    return arcsec_to_rad(marcsec / 1000.0)


def deg_to_arcsec(deg):
    return deg * 3600.0


def rad_to_pixel(rad: float, fieldOfViewU: float, U: int) -> float:
    return float(U) * np.rad2deg(rad) / fieldOfViewU


def camera_to_world(T_worldToCamera: np.ndarray[float], vec: np.ndarray[float]):
    return (T_worldToCamera.T @ vec)


def world_to_camera(T_worldToCamera: np.ndarray[float], vec: np.ndarray[float]):
    return (T_worldToCamera @ np.array([vec]).T).T[0]


def camera_to_uv_centered(fieldOfViewU: float, fieldOfViewV: float, U: float, V: float, vec: np.ndarray[float]) -> np.ndarray[float]:
    u = -0.5 * U * vec[1] / (vec[0] * np.tan(0.5 * np.deg2rad(fieldOfViewU)))
    v = -0.5 * V * vec[2] / (vec[0] * np.tan(0.5 * np.deg2rad(fieldOfViewV)))
    return np.array([u, v, 1.0])


def uv_centered_to_camera(fieldOfViewU: float, fieldOfViewV: float, U: float, V: float, uc: float, vc: float) -> np.ndarray[float]:
    px = 1.0
    py = -2.0 * uc * px * np.tan(0.5 * np.deg2rad(fieldOfViewU)) / U
    pz = -2.0 * vc * px * np.tan(0.5 * np.deg2rad(fieldOfViewV)) / V
    return normalize(np.array([px, py, pz]))


def archaversine(r: float, dec1: float, dec2: float, ra1: float, ra2: float):
    return 2 * r * np.arcsin(np.sqrt(0.5 * (1 - np.cos(dec2 - dec1) + np.cos(dec1) * np.cos(dec2) * (1 - np.cos(ra2 - ra1)))))


def archaversine_unit(dec1: float, dec2: float, ra1: float, ra2: float):
    return archaversine(1.0, dec1, dec2, ra1, ra2)


def clamp(val, low, high):
    return low if val < low else high if val > high else val
