from SPS.grav_moon_GRAIL150 import *
from py_src.star.python.transformations import *

import sys
import csv
import copy

from matplotlib import pyplot as plt
import matplotlib
import spiceypy as spice

import pyshtools as sh
import pyshtools.gravmag as grav
from pyshtools.gravmag import MakeGravGridPoint


def read_csv(path: str, ignore: list=[], hasHeader=False):
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"', lineterminator='\n')
        data = []

        headerBool = False
        if hasHeader:
            headerBool = True
        for row in reader:
            if headerBool:
                headerBool = False
                continue
            data.append([float(row[i]) for i in range(len(row)) if i not in ignore])
        return data


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
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


def matrix_to_angleaxis(m: np.ndarray[float]):
    # Angle in radians
    phi_r = np.arccos(0.5 * (np.trace(m) - 1.0))

    # Axis components
    sin_phi_2 = 0.5 / np.sin(phi_r)
    e0 = (m[1][2] - m[2][1]) * sin_phi_2
    e1 = (m[2][0] - m[0][2]) * sin_phi_2
    e2 = (m[0][1] - m[1][0]) * sin_phi_2

    # Return angle (in degrees) + axis
    phi_d = np.rad2deg(phi_r)
    e = np.array([e0, e1, e2])
    return phi_d, e


def deg_to_arcsec(angle):
    return angle * 3600.0


def arcsec_to_rad(angle):
    return np.deg2rad(angle / 3600.0)


def plot_errors(truthDataPath: str, estDataPath: str, reject: float=0, planetRadius: float=6378136.3):
    truthData = read_csv(truthDataPath)
    estData = read_csv(estDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
    
    # Very basic error handling if datasets are not the same length
    if len(truthData) != len(estData):
        return

    # Compare data
    errorsArcsec = []
    for i in range(len(truthData)):
        truth_i = truthData[i]
        est_i = estData[i]
        
        if est_i[0] == 999 or est_i[1] == 999 or est_i[2] == 999 or est_i[3] == 999:
            continue
        
        q_real = Quaternion(truth_i[0], truth_i[1], truth_i[2], truth_i[3]).normalize()
        q_est = Quaternion(est_i[0], est_i[1], est_i[2], est_i[3]).normalize()
        q_err = q_est.conjugate().mult(q_real).normalize()
        m = spice.q2m(q_err.as_w_first_array())
        phi_d, _ = matrix_to_angleaxis(m)
        phi_arcsec = deg_to_arcsec(phi_d)
        
        if reject != 0 and phi_arcsec > reject:
            continue
        errorsArcsec.append(phi_arcsec)
    
    # Calculate statistics
    mean = np.mean(errorsArcsec)
    median = np.median(errorsArcsec)
    std = np.std(errorsArcsec)
    projectedSPSMeanError = arcsec_to_rad(mean) * planetRadius

    # Print statistics
    print(f'Mean                        = {round(mean, 3)} arcseconds')
    print(f'Median                      = {round(median, 3)} arcseconds')
    print(f'Standard Deviation          = {round(std, 3)} arcseconds')
    print(f'Projected SPS Mean Error    = {round(projectedSPSMeanError, 1)} meters')

    # Plot histogram and box plot of errors
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.hist(errorsArcsec, bins=200, color='skyblue', edgecolor='black')
    mpl_v = matplotlib.__version__.split(".")
    mpl_v_maj = float(mpl_v[0])
    mpl_v_min = float(mpl_v[1])
    # mpl_v_patch = float(mpl_v[2])

    # If matplotlib version is too low, the "orientation" kwarg is invalid. Need to use "vert: bool" for matplotlib < 3.10
    if mpl_v_maj < 3 or mpl_v_min < 10:
        ax2.boxplot(errorsArcsec, vert=False, showfliers=False)
    else:
        ax2.boxplot(errorsArcsec, orientation='horizontal', showfliers=False)

    # Nice plot stuff
    extraPrint = f' (errors > {reject} arcseconds rejected)'
    if reject == 0:
        extraPrint = f' (no errors rejected)'
    ax1.set_title(f'Attitude Error Counts for {len(errorsArcsec)}/{len(truthData)} Samples' + extraPrint)
    ax1.set_xlabel('Error (arcseconds)')
    ax1.set_ylabel('Count')
    ax1.grid()
    ax2.set_title(f'Attitude Error Statistics for {len(errorsArcsec)}/{len(truthData)} Samples' + extraPrint)
    ax2.set_xlabel('Error (arcseconds)')
    ax2.grid()

    # Show plot
    plt.show()


def estimate_position(truthDataPath: str, estDataPath: str, latLonDataPath: str):
    truthData = read_csv(truthDataPath)
    estData = read_csv(estDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
    latLonData = read_csv(latLonDataPath)
    
    # Very basic error handling if datasets are not the same length
    if not (len(truthData) == len(estData) == len(latLonData)):
        return

    # Compare data
    errorsArcsec = []
    moon = grav_moon_GRAIL150()

    max_degree = 32
    max_order = 32
    Cilm = np.dstack((moon.Clm[:max_degree + 1, :max_order + 1], moon.Slm[:max_degree + 1, :max_order + 1])).transpose((2, 0, 1))
    # Cilm = np.zeros((2, max_degree + 1, max_order + 1))  # testing spherical gravity
    Cilm[0, 0, 0] = 1.0  # add spherical component of gravity

    for i in range(len(truthData)):
        truth_i = truthData[i]
        est_i = estData[i]
        latLon_i = latLonDataPath[i]
        
        if est_i[0] == 999 or est_i[1] == 999 or est_i[2] == 999 or est_i[3] == 999:
            continue
        
        q_real = Quaternion(truth_i[0], truth_i[1], truth_i[2], truth_i[3]).normalize()
        q_est = Quaternion(est_i[0], est_i[1], est_i[2], est_i[3]).normalize()
        
        lat = 0.0 
        lon = 0.0
        
        T_g_c = np.identity(3)  # transformation from gravity to camera frame
        
        T_i_b = np.identity(3)  # transformation from inertial to planet frame
        
        T_s_g = np.identity(3)  # transformation from surface to gravity frame
        T_s_g_old = T_angle_axis(np.deg2rad(5.0), np.array([0, 0, 1]))
        
        T_i_c = q_est.to_matrix()
        
        i: int = 0
        tol_angle_arcsec = 1.0
        tol_angle_deg = tol_angle_arcsec / 3600.0
        while abs(AttitudeError(T_s_g, T_s_g_old)) > tol_angle_deg:
            i += 1
            T_s_g_old = copy.deepcopy(T_s_g)
            
            T_b_s = T_s_g.T @ T_g_c.T @ T_i_c @ T_i_b.T  # transformation from planet to surface frame
            lat, lon = T_to_latlon(T_b_s)
            
            g = sample_gravity(moon, Cilm, lat, lon, moon.radius, max_degree)
            p_hat_gc = T_b_s[:, 0]
            p_hat_as = -g / np.linalg.norm(g)
            T_s_g = TwoVectors_to_T(p_hat_as, p_hat_gc)
            
            print(f'Run {i}:')
            print(f'err = {round(AttitudeError(T_s_g, T_s_g_old) * 3600.0, 1)}"')
            print(f'lat = {lat} deg')
            print(f'lon = {lon} deg\n')
        
        latTruth = latLon_i[0]
        lonTruth = latLon_i[1]
        distance_err = archaversine(moon.radius, np.deg2rad(latTruth), np.deg2rad(lat), np.deg2rad(lonTruth), np.deg2rad(lon))
        print(f'distance_err = {round(distance_err * 1000.0, 1)} m')


def sample_gravity(moon: grav_moon_GRAIL150, Cilm, lat, lon, r, max_degree):
    mu = moon.mu
    R = moon.radius
    omega = moon.omega

    T = latlon_to_T(lat, lon)
    g_pcpf = T @ MakeGravGridPoint(Cilm, mu, R, r, lat, lon, max_degree, omega)
    return g_pcpf


if __name__ == "__main__":
    rEarth = 6378136.3
    rMoon = 1737400.0
    rMars = 3396190.0
    rPhobos = 11000.0

    # Star tracker "measurements" file
    n = len(sys.argv)
    measurements = ".\\output.csv"
    if n > 1:
        measurements = sys.argv[1]
    
    # Truth source directory
    truthSourceDir = ".\\py_src\\star\\"
    if n > 2:
        truthSourceDir = sys.argv[2]
    
    latLonDataPath = ".\\sampleLatLongs.csv"
    if n > 3:
        latLonDataPath = sys.argv[3]
    
    # # Cutoff for "good measurements" in arcseconds
    # reject = 0.0
    # if n > 3:
    #     reject = float(sys.argv[3])
    
    # # Planet radius for evaluating location error
    # radius = 0.0
    # if n > 4:
    #     radius = float(sys.argv[4])
    # else:
    #     radius = rEarth

    estimate_position(truthSourceDir + "truth_data.csv", measurements, latLonDataPath)
    pass
