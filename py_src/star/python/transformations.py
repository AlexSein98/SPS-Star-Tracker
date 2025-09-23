import numpy as np
import os


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


def r_hat_to_latlon(r_hat: np.ndarray[float]):
    lat = np.rad2deg(np.arcsin(r_hat[2] / np.linalg.norm(r_hat)))
    lon = np.rad2deg(np.arctan2(r_hat[1], r_hat[0]))
    return lat, lon


def r_to_latlonalt(r: np.ndarray[float], R: float) -> tuple[float, float, float]:
    r_hat = normalize(r)
    lat = np.rad2deg(np.arcsin(r_hat[2] / np.linalg.norm(r_hat)))
    lon = np.rad2deg(np.arctan2(r_hat[1], r_hat[0]))
    alt = np.linalg.norm(r) - R
    return lat, lon, alt


def latlon_to_T(lat: float, lon: float):
    return R3(np.deg2rad(lon)) @ R2(np.deg2rad(-lat))


def T_to_latlon(T: np.ndarray[float]):
    r_hat = T[:, 0]
    return r_hat_to_latlon(r_hat)


def T_angle_axis(angle: float, axis: np.ndarray[float]) -> np.ndarray[float]:
    e_cross = np.array([[0.0, -axis[2], axis[1]], 
                        [axis[2], 0.0, -axis[0]], 
                        [-axis[1], axis[0], 0.0]])
    return np.identity(3) - np.sin(angle) * e_cross + (1.0 - np.cos(angle)) * e_cross @ e_cross


def arccos_safe(arg):
    if arg > 1.0:
        return 0.0
    elif arg < -1.0:
        return np.pi
    else:
        return np.arccos(arg)


def TwoVectors_to_T(v1, v2):
    axis = normalize(np.cross(v1, v2))
    angle = arccos_safe(np.dot(normalize(v1), normalize(v2)))
    return T_angle_axis(angle, axis)


def RADecRoll_to_Camera(RA: float, dec: float, roll: float):
    return R3(np.deg2rad(RA)) @ R2(np.deg2rad(-dec)) @ R1(np.deg2rad(roll))


def AttitudeError(T_true: np.ndarray[float], T_est: np.ndarray[float]) -> float:
    delta_T: np.ndarray[float] = T_true @ T_est.T
    return np.rad2deg(np.arccos(0.5 * (np.trace(delta_T) - 1.0)))


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


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    @classmethod
    def FromMatrix(quat, m: np.ndarray[float]):
        r__ = np.array([m[0, 0], m[0, 1], m[0, 2], 
                        m[1, 0], m[1, 1], m[1, 2], 
                        m[2, 0], m[2, 1], m[2, 2]])
        s = np.zeros(3)

        trace = r__[0] + r__[4] + r__[8]
        mtrace = 1. - trace
        cc4 = trace + 1.
        s114 = mtrace + r__[0] * 2.
        s224 = mtrace + r__[4] * 2.
        s334 = mtrace + r__[8] * 2.

        if (1. <= cc4):
            c__ = np.sqrt(cc4 * .25)
            factor = 1. / (c__ * 4.)
            s[0] = (r__[5] - r__[7]) * factor
            s[1] = (r__[6] - r__[2]) * factor
            s[2] = (r__[1] - r__[3]) * factor
        elif (1. <= s114):
            s[0] = np.sqrt(s114 * .25)
            factor = 1. / (s[0] * 4.)
            c__ = (r__[5] - r__[7]) * factor
            s[1] = (r__[3] + r__[1]) * factor
            s[2] = (r__[6] + r__[2]) * factor
        elif (1. <= s224):
            s[1] = np.sqrt(s224 * .25)
            factor = 1. / (s[1] * 4.)
            c__ = (r__[6] - r__[2]) * factor
            s[0] = (r__[3] + r__[1]) * factor
            s[2] = (r__[7] + r__[5]) * factor
        else:
            s[2] = np.sqrt(s334 * .25)
            factor = 1. / (s[2] * 4.)
            c__ = (r__[1] - r__[3]) * factor
            s[0] = (r__[6] + r__[2]) * factor
            s[1] = (r__[7] + r__[5]) * factor

        q = np.zeros(4)
        l2 = c__ * c__ + s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        if (l2 != 1.):
            polish = 1. / np.sqrt(l2)
            c__ *= polish
            s[0] *= polish
            s[1] *= polish
            s[2] *= polish
        if (c__ > 0.):
            q[0] = c__
            q[1] = s[0]
            q[2] = s[1]
            q[3] = s[2]
        else:
            q[0] = -c__
            q[1] = -s[0]
            q[2] = -s[1]
            q[3] = -s[2]
        
        return quat(q[0], q[1], q[2], q[3])

    def mult(self, other):
        w1 = self.w
        x1 = self.x
        y1 = self.y
        z1 = self.z
        w2 = other.w
        x2 = other.x
        y2 = other.y
        z2 = other.z
        return Quaternion(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    
    def normalize(self):
        mag_1 = 1.0 / np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        return Quaternion(self.w * mag_1, self.x * mag_1, self.y * mag_1, self.z * mag_1)
    
    def conjugate(self):
        return Quaternion(-self.w, self.x, self.y, self.z)
    
    def as_w_first_array(self):
        return np.array([self.w, self.x, self.y, self.z])
    
    def as_w_last_array(self):
        return np.array([self.x, self.y, self.z, self.w])
    
    def to_matrix(self):
        # double l2, q01, q02, q03, q12, q13, q23, sharpn, q1s, q2s, q3s;

        q01 = self.w * self.x
        q02 = self.w * self.y
        q03 = self.w * self.z
        q12 = self.x * self.y
        q13 = self.x * self.z
        q23 = self.y * self.z
        q1s = self.x * self.x
        q2s = self.y * self.y
        q3s = self.z * self.z

        l2 = self.w * self.w + q1s + q2s + q3s
        if l2 != 1.0 and l2 != 0.0: 
            sharpn = 1.0 / l2
            q01 *= sharpn
            q02 *= sharpn
            q03 *= sharpn
            q12 *= sharpn
            q13 *= sharpn
            q23 *= sharpn
            q1s *= sharpn
            q2s *= sharpn
            q3s *= sharpn

        m = np.zeros((3, 3))
        m[0][0] = 1.0 - (q2s + q3s) * 2.0
        m[0][1] = (q12 + q03) * 2.0
        m[0][2] = (q13 - q02) * 2.0
        m[1][0] = (q12 - q03) * 2.0
        m[1][1] = 1.0 - (q1s + q3s) * 2.0
        m[1][2] = (q23 + q01) * 2.0
        m[2][0] = (q13 + q02) * 2.0
        m[2][1] = (q23 - q01) * 2.0
        m[2][2] = 1.0 - (q1s + q2s) * 2.0

        return m


if __name__ == "__main__":
    os.system('cls')

    r = np.array([2193075.113333, 743921.623579, -2485679.376025])
    R_mars = 3396190.0
    lat, lon, alt = r_to_latlonalt(r, R_mars)
    T = latlon_to_T(lat, lon) @ T2(np.pi / 2.0)
    q: Quaternion = Quaternion.FromMatrix(T)

    print(f'lat = {round(lat, 6)}, lon = {round(lon, 6)}\n')
    print(f'T = {T}\n')
    print(f'q = [{round(q.w, 6)}, {round(q.x, 6)}, {round(q.y, 6)}, {round(q.z, 6)}]')
