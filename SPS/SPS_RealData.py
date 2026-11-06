import os, sys, time, datetime, traceback
import spaceteams as st
import scipy.special as special
from scipy.optimize import curve_fit
# Can import other modules here, but any code that actually executes before st.connect_to_sim() 
# will have less helpful logging and error handling. SimGlobals and logger functions are not yet available.

def before_init():
    pass # Replace this to run code during the Sync state that needs to finish before Init starts anywhere in the sim.
    # Sync state is where Sim Editing is done, so code in here shouldn't change parameters that would
    # make saving and reloading the sim problematic.

def before_runtime():
    pass # Replace this to run code during the Init state that needs to finish before Runtime starts anywhere in the sim.

#################################################################
# DON'T CHANGE ANY OF THE BELOW; NECESSARY FOR JOINING SIMULATION
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
    exit(1)
sys.excepthook = custom_exception_handler
st.BeforeInit(before_init)
st.BeforeRuntime(before_runtime)
st.connect_to_sim(sys.argv)
# DON'T CHANGE ANY OF THE ABOVE; NECESSARY FOR JOINING SIMULATION
#################################################################

thisRepo = os.path.join(str(st.path_utils.GetLocalAssetsDir()), "Repos", "SPS-Star-Tracker")
localOutput = str(st.path_utils.GetLocalOutputDir())
sys.path.append(thisRepo)
sys.path.append(localOutput)

from py_src.star_tracker.star_tracker import main
from py_src.star_tracker.star_tracker.cam_matrix import *
from py_src.star_tracker.star_tracker.array_transformations import *

# import spiceypy as spice
import copy
import psutil
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib

import numpy as np
import numpy.typing as npt
import shutil
import csv
import astropy.time as astrotime

from dataclasses import dataclass, field
import pyexiv2 as xmp

# from SPS.global_config import globalConfig
from py_src.star.python.transformations import latlon_to_T, T_to_latlon, r_to_latlonalt, r_hat_to_ra_dec, normalize

np.seterr(all='raise')


def T1(angle: float) -> npt.NDArray:
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(angle), np.sin(angle)],
                     [0.0, -np.sin(angle), np.cos(angle)]])
                     

def T2(angle: float) -> npt.NDArray:
    return np.array([[np.cos(angle), 0.0, -np.sin(angle)],
                     [0.0, 1.0, 0.0],
                     [np.sin(angle), 0.0, np.cos(angle)]])
                     

def T3(angle: float) -> npt.NDArray:
    return np.array([[np.cos(angle), np.sin(angle), 0.0],
                     [-np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def R1(angle: float) -> npt.NDArray:
    return T1(angle).T


def R2(angle: float) -> npt.NDArray:
    return T2(angle).T


def R3(angle: float) -> npt.NDArray:
    return T3(angle).T


def ra_dec_to_rot(rightAscension: float, declination: float):
    return R3(np.deg2rad(rightAscension)) @ R2(np.deg2rad(-declination))


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


def write_csv(filepath: str, data: list[np.ndarray[float]]):
    with open(filepath, "w") as dataCSV:
        writer = csv.writer(dataCSV, delimiter=',', quotechar='"', lineterminator='\n')
        for line in data:
            writer.writerow(line)


def positivize_quat(q: npt.NDArray) -> npt.NDArray:
    if q[3] > 0.0:
        return q
    else:
        return np.array([-q[0], -q[1], -q[2], -q[3]])


# For w-last quaternions
def skew_symmetric_quat(q: npt.NDArray) -> npt.NDArray:
    # return np.array([[q[3], q[2], -q[1], q[0]],
    #                  [-q[2], q[3], q[0], q[1]],
    #                  [q[1], -q[0], q[3], q[2]],
    #                  [-q[0], -q[1], -q[2], q[3]]])
    return np.array([[q[3], -q[2], q[1], q[0]],
                     [q[2], q[3], -q[0], q[1]],
                     [-q[1], q[0], q[3], q[2]],
                     [-q[0], -q[1], -q[2], q[3]]])


def quat_mult(q1: npt.NDArray, q2: npt.NDArray):
    x = q1[0]
    y = q1[1]
    z = q1[2]
    w = q1[3]

    x2 = q2[0]
    y2 = q2[1]
    z2 = q2[2]
    w2 = q2[3]

    q_product = np.array([w * x2 + x * w2 + y * z2 - z * y2,
                          w * y2 - x * z2 + y * w2 + z * x2,
                          w * z2 + x * y2 - y * x2 + z * w2,
                          w * w2 - x * x2 - y * y2 - z * z2])
    return positivize_quat(q_product)


def quat_inverse(q: npt.NDArray):
    if q[3] > 0.0:
        q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
        return q_inv
    else:
        q_inv = np.array([q[0], q[1], q[2], -q[3]])
        return q_inv


def read_quats(filename: str) -> list[npt.NDArray]:
    quats = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("CPU"):
                continue
            curr = line.strip().split(",")
            w = float(curr[4])
            x = float(curr[5])
            y = float(curr[6])
            z = float(curr[7])
            if w > 0.0:
                quats.append(np.array([x, y, z, w]))
            else:
                quats.append(np.array([-x, -y, -z, -w]))
    return quats


def xi_transpose(q: npt.NDArray, active: bool) -> npt.NDArray:
    qx = copy.deepcopy(q[0])
    qy = copy.deepcopy(q[1])
    qz = copy.deepcopy(q[2])
    qw = copy.deepcopy(q[3])

    if active:
        xi = np.array([[qw, -qz, qy],
                    [qz, qw, -qx],
                    [-qy, qx, qw],
                    [-qx, -qy, -qz]])
        return xi.T
    else:
        xi = np.array([[qw, qz, -qy],
                       [-qz, qw, qx],
                       [qy, -qx, qw],
                       [-qx, -qy, -qz]])
        return xi.T


def estimate_omega(quats: list[npt.NDArray], deltaT: float, T_unrotate: list[npt.NDArray]) -> npt.NDArray:
    """
    IMPORTANT:
    q_dot is kept as a raw 4-vector and is NOT normalized.
    """
    # quats = align_quaternion_signs(quats)

    N = len(quats) - 1
    omega_sum = np.zeros(3)
    omega_history: list[npt.NDArray] = []

    for i in range(1, N - 1):
        q_prev = st.math.DCM_to_Quat(T_unrotate[i - 1] @ st.math.Quat_to_DCM(normalize(quats[i - 1])))
        q_curr = st.math.DCM_to_Quat(T_unrotate[i] @ st.math.Quat_to_DCM(normalize(quats[i])))
        q_next = st.math.DCM_to_Quat(T_unrotate[i + 1] @ st.math.Quat_to_DCM(normalize(quats[i + 1])))

        q_dot = np.array([[(q_next[0] - q_prev[0]) / (2.0 * deltaT),
                           (q_next[1] - q_prev[1]) / (2.0 * deltaT),
                           (q_next[2] - q_prev[2]) / (2.0 * deltaT),
                           (q_next[3] - q_prev[3]) / (2.0 * deltaT)]]).T
        
        T_extra = np.array([[0, 0, 1], 
                            [1, 0, 0], 
                            [0, 1, 0]])
        
        omega_i = (2.0 * T_extra @ xi_transpose(q_curr, False) @ q_dot).T[0]
        omega_history.append(omega_i)
        omega_sum += omega_i

    return omega_sum / float(len(omega_history)), omega_history


def estimate_omega_EnrightForm(quats: list[npt.NDArray], deltaT: float, T_unrotate: list[npt.NDArray]) -> npt.NDArray:
    """
    IMPORTANT:
    q_dot is kept as a raw 4-vector and is NOT normalized.
    """
    # quats = align_quaternion_signs(quats)

    N = len(quats) - 1
    omega_sum = np.zeros(3)
    omega_history: list[npt.NDArray] = []

    for i in range(1, N-1):
        # q_prev = st.math.DCM_to_Quat(st.math.Quat_to_DCM(normalize(quats[i - 1])) @ T_unrotate[i - 1])
        # q_curr = st.math.DCM_to_Quat(st.math.Quat_to_DCM(normalize(quats[i])) @ T_unrotate[i])
        q_prev = quats[i - 1]
        q_curr = quats[i]
        q_next = quats[i + 1]
        # q_next = st.math.DCM_to_Quat(T_unrotate[i + 1] @ st.math.Quat_to_DCM(normalize(quats[i + 1])))

        delta_q = normalize(quat_mult(q_next, quat_inverse(q_prev)))
        angle: float = 2.0 * np.arccos(np.clip(delta_q[3], -1.0, 1.0))
        axis: npt.NDArray = np.zeros(3)
        if angle != 0.0:
            axis: npt.NDArray = delta_q[0:3] / np.sin(0.5 * angle)

        # q_unrotate_prev = st.math.DCM_to_Quat(T_unrotate[i - 1])
        # q_unrotate_curr = st.math.DCM_to_Quat(T_unrotate[i])
        # T_unrotate_avg = st.math.Quat_to_DCM(normalize(0.5 * (q_unrotate_prev + q_unrotate_curr)))
        # omega_i = T_unrotate_avg @ (angle * axis / deltaT)

        omega_i = T_unrotate[i] @ (-angle * axis / (2.0 * deltaT))

        # omega_i = angle * axis / deltaT
        omega_history.append(omega_i)
        omega_sum += omega_i

    return omega_sum / float(len(omega_history)), omega_history


def average_quats(quats: list[npt.NDArray]) -> npt.NDArray:
    N = len(quats)
    M = np.zeros((4, 4))
    w = 1.0 / float(N)
    for i in range(N):
        M += w * np.outer(quats[i], quats[i])

    eigvals, eigvecs = np.linalg.eig(M)
    eigvalArgMax = np.argmax(eigvals)
    eigvecMax = eigvecs[:, eigvalArgMax]
    return positivize_quat(normalize(eigvecMax))


"""
Following this QuateRA paper:

Recursive Star-Identification Algorithm Using an Adaptive Singular-Value-Decomposition-Based Angular-Velocity Estimator
Hunter Johnston, Carl Leake, Marcelino M. de Almeida, and Daniele Mortari
Journal of Spacecraft and Rockets 2021 58:4, 1138-1148

"""
def estimate_omega_QuateRA(quats: list[npt.NDArray], times: list[float]) -> npt.NDArray:
    N = len(quats)

    # Step 1, following Equation (4)
    Q_k_n = np.zeros((4, N))
    for k in range(N):
        Q_k_n[:, k] = quats[k]

    # Step 2
    result = np.linalg.svd(Q_k_n @ Q_k_n.T)
    U = result.U
    u1 = U[:, 0]
    u2 = U[:, 1]

    # Step 3
    omega_opt = skew_symmetric_quat(u2) @ quat_inverse(u1)

    # Steps 4 and 5
    Phi = np.zeros(N)
    weights = np.zeros(N)
    for i in range(N):
        # Optimal quaternions
        q_proj_u1: npt.NDArray = quats[i].dot(u1)
        q_proj_u2: npt.NDArray = quats[i].dot(u2)
        q_i: npt.NDArray = (q_proj_u1 * u1 + q_proj_u2 * u2) / np.sqrt(q_proj_u1 ** 2 + q_proj_u2 ** 2)
        Phi[i] = 2.0 * np.atan2(q_i.dot(u2), q_i.dot(u1))

        # Weights
        quat_deviation: npt.NDArray = normalize(skew_symmetric_quat(q_i) @ quat_inverse(quats[i]))
        angle_deviation: float = 2.0 * np.arccos(np.clip(quat_deviation[3], -1.0, 1.0))
        weights[i] = 1.0 / (angle_deviation ** 2)

    Phi = np.unwrap(Phi, discont=2.0 * np.pi)
    st.OnScreenLogMessage(f"Phi = {Phi}", "SPSTraverse", st.Severity.Info)

    # W = np.diag(weights)
    W = np.identity(N)

    # Step 6
    H = np.ones((N, 2))
    for k in range(N):
        H[k, 1] = times[k]
    X_opt = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ Phi
    Omega_opt = X_opt[1]

    st.OnScreenLogMessage(f"omega_opt = {omega_opt}", "SPSTraverse", st.Severity.Info)
    st.OnScreenLogMessage(f"Omega_opt = {Omega_opt}", "SPSTraverse", st.Severity.Info)

    # Return angular velocity
    angVel: npt.NDArray = Omega_opt * omega_opt
    return angVel[0:3]


os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

##################################
####    PLANET DATA IMPORT    ####
##################################

planetData = st.ProcPlanet.DataStore()

earthGlobalData = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Earth/Global/Topography/gebco_08_rev_elev_A1_grey_geo")
args = st.ProcPlanet.GeoBin_Extra_Args()
args.cubicInterp = True

planetData.AddGeoBinAltimetryLayer(1.0, earthGlobalData, args)

#####################################
####    CLASSES AND FUNCTIONS    ####
#####################################

def rad_to_arcsec(rad: float) -> float:
    return 3600.0 * np.rad2deg(rad)


def is_dir_empty(path):
    # Returns True if empty, False otherwise
    return not any(Path(path).iterdir())


@dataclass
class DataCapture:
    filepath: Path | None = None
    capture_time: datetime.datetime | None = None
    acceleration: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @classmethod
    def from_img(cls, filepath: str | Path) -> 'DataCapture':
        filepath = Path(filepath)
        with xmp.Image(str(filepath)) as img:
            metadata = img.read_xmp()

        return cls(
            filepath=filepath,
            capture_time=datetime.datetime.fromisoformat(metadata['Xmp.sps.capture_time_UTC']),
            acceleration=[
                float(metadata['Xmp.sps.ax']),
                float(metadata['Xmp.sps.ay']),
                float(metadata['Xmp.sps.az']),
            ],
        )


def arcsec_to_rad(arcsec: float) -> float:
    return np.deg2rad(arcsec / 3600.0)
    #This is not arcsecant. This is an arcsecond. An arcsecond is 1/3600 of a degree.
    # 1 arcsecond * (1/3600) Degrees / Arcsecond = 1/3600 degree
    # Also we use np.deg2rad to convert it to radians, multiplying by pi / 180


'''
Converts planetographic (i.e., geodetic) latitude, longitude, and ellipsoid-referenced 
altitude to an equivalent Cartesian position vector in the planet-fixed frame.

Inputs:
lat:    geodetic latitude [degrees]; latitude = north/south measurement angle, aka 'spicy polar angle'
lon:    longitude [degrees]; longitude = east/west measurement angle, aka 'azimuthal angle'
alt:    ellipsoid-referenced altitude [meters]
a:      planet equatorial radius [meters]
b:      planet polar radius [meters]

Outputs:
[x, y, z]:  Cartesian position vector in planet-fixed frame [meters]
'''
def planetographic_to_cartesian(lat: float, lon: float, alt: float, a: float, b: float) -> npt.NDArray:
    lat_rad: float = np.deg2rad(lat) # Similar to polar angle in spherical coordinates, but measured from equatorial plane rather than top of sphere
    lon_rad: float = np.deg2rad(lon) # Similar to azimuthal angle in spherical coordinates

    # From above, looking down into the xy plane, the elliptical earth just looks like a regular sphere
    # Therefore, the conversions become trivial: rcos(lat) projects onto the xy plane, and cos or sin(lon) projects onto the x or y axes
    # Except instead of r, we use N + alt so the xy projection line starts from the z-axis
    # N is a fancy number representing the distance from a point on the surface to the z-axis, along the surface normal vector
    # N depends on a, b, and latitude (spicy polar angle)
    # N + alt is therefore this coordinate system's version of regular spherical coordinate radius
    # Also, the z-coordinate (simply r * sin(lat) if it were a sphere) requires a scaling factor based on the radii of the ellipse

    N: float = a ** 2 / np.sqrt(a ** 2 * np.cos(lat_rad) ** 2 + b ** 2 * np.sin(lat_rad) ** 2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((b ** 2 / a ** 2) * N + alt) * np.sin(lat_rad)
    return np.array([x, y, z])

    # The spicy-spherical --> Cartesian conversions are as follows:
    # x = (N + alt)cos(lat)cos(lon)
    # y = (N + alt)cos(lat)sin(lon)
    # z = ((b/a)^2 * N + alt)sin(lat)

    # Spherical or spicy spherical, getting TO Cartesian coordinates is generally easy


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

    # First we simply extract the xyz coordinates from the NDArray.
    # We also calculate r_delta, which is the projection of the true radius into the xy plane, aka distance from z-axis

    alpha: float = np.arctan2(y, x)  # This directly becomes longitude. np.arctan2() is needed because longitude is not bounded by -pi/2, pi/2
    phi_pg: float = np.arctan(z / r_delta) # This is an 'estimate' of latitude, using z and r_delta. It will be refined over time
    ecc: float = np.sqrt(1.0 - (b / a) ** 2)
    C: float = a / np.sqrt(1.0 - ecc ** 2 * np.sin(phi_pg) ** 2)

    # Because the top-down view of the ellipsoid earth in the xy plane looks just like a regular circle...
    # Finding the longitude (azimuthal angle) is trivial!
    # But the latitude (spicy polar angle) is trickier. First we make a guess using arctan(z / r_delta) as if it were a perfect sphere
    # We also calculate eccentricity using the equatorial and polar radii: e = sqrt(1 - (b/a)^2)
    # Eccentricity is a value between 0 and 1 that measures the similarity of an ellipse to a sphere
    # Eccentricity of 0 means a perfect sphere. Eccentricity of 1 means infinitely fat
    # Eccentricity is also defined as e = c/a, the ratio of 'c' (the distance from the center to a focus) to the semi-major axis
    # And c, the focus distance, is found with c^2 + b^2 = a^2, drawing a right triangle in the ellipse where the LENGTH of a is the hypotenuse
    # Thus, c = sqrt(a^2 - b^2), and e = c/a...combine these and you get e = sqrt(1 - (b/a)^2)
    # This means that if you decrease b or increase a, the eccentricity (stretchiness) will increase
    # C is just another name for N, and it can also be calculated using eccentricity
    
    phi_pg_old: float = 2.0 * phi_pg
    while abs(phi_pg - phi_pg_old) > arcsec_to_rad(0.1):
        phi_pg_old = copy.deepcopy(phi_pg)
        C = a / np.sqrt(1.0 - ecc ** 2 * np.sin(phi_pg) ** 2)
        phi_pg = np.arctan((z + C * ecc ** 2 * np.sin(phi_pg)) / r_delta)

    # This loop is designed to iteratively improve the latitude guess for as long as the gap between the old and new guess is > 0.1 arcseconds
    # We start off by using 2 * 'regular latitude' as our 'past guess', just so that the code begins.
    # The 'new' past guess becomes a copy of the 'current' guess, prior to its modification.
    # It calculates C using the latitude, and then plugs that version back into the formula for latitude
    # Since the formula for latitude uses the actual, known values of z and r_delta, it will produce a more accurate result
    # Basically, C --> Latitude improves the value of latitude, which gives us a C of similar accuracy, which lets us find an even more accurate latitude

    h_ellp = 0.0
    if abs(phi_pg) > np.deg2rad(89.0):
        S: float = a * (1 - ecc ** 2) / np.sqrt(1.0 - ecc ** 2 * np.sin(phi_pg) ** 2)
        h_ellp = z / np.sin(phi_pg) - S
    else:
        h_ellp: float = r_delta / np.cos(phi_pg) - C #This is the line that I moved down.

    # This calculates our altitude, now that we have a good value for longitude, latitude, and C
    # In the 'else' block, r_delta * sec(lat) projects r_delta along the line of C
    # Since that value includes C + our altitude along the surface normal, we just subtract C to get back the altitude
    # HOWEVER, if our latitude angle is close to 90 degrees, then r_delta and cos(lat) will both approach 0, which can cause errors
    # So in that case, we use S = (b/a)^2 * N, like from the weird z-coordinate formula from before.
    # Except we use eccentricity again by convention, evn though that makes it more complicated.
    # Therefore, z = ((b/a)^2 * N + alt)sin(lat) becomes z = (S + alt)sin(lat), and thus, alt = z / sin(lat) - S.
    
    lat: float = np.rad2deg(phi_pg)
    lon: float = np.rad2deg(alpha)
    alt: float = h_ellp
    return lat, lon, alt

    # We have gone from x, y, and z (and the semi-major/semi-minor axes) to spicy latitude, longitude, and altitude!


np.set_printoptions(suppress=True)

#################################
####    GLOBAL PARAMETERS    ####
#################################

reprocess_star_tracker: bool    = True
doAlignment: bool               = False
doAngVel: bool                  = True
globalDoPrint: bool             = True

alignmentCutoff: int = 100
endpoint: int = alignmentCutoff + 200

numImages: int = alignmentCutoff + 200

#####################################
####    DATA SOURCE SELECTION    ####
#####################################

this = st.GetThisSystem()

"""
Expected directory structure (e.g., on Windows)

Notes:
- We're on Earth for all near-future field tests, so the "Earth" directory should be used for all test runs
- Directories with custom names are indicated in [brackets]
- The System param "DataSource" needs to indicate a valid directory under the "Earth" directory

C:/Users/[user]/AppData/Local/SimDynamX/STPro/Output/Earth
    [DataSource]
        -> Images
            -> StarImage0001.png
            -> StarImage0002.png
            ...
            -> StarImageXXXX.png
        -> Calibration
            -> Calibration.json
            -> DarkFrame.png
            -> LightFrame.png
        
        measurements_Earth.csv      (generated by this script)
        omega_i_b_Earth.csv         (generated by this script)
        q_i_b_Earth.csv             (generated by this script)
        true_gravities_Earth.csv    (generated by this script)
"""

dataSource: str = this.GetParam(st.VarType.string, "DataSource")

# Expects latitude and longitude in degrees, altitude in meters
trueLoc: npt.NDArray = this.GetParam(st.VarType.doubleV3, "TrueLocationLLA")
phi_pg_0: float = trueLoc[0]
lon_pg_0: float = trueLoc[1]
h_ellp_0: float = trueLoc[2]

# phi_pg_0: float = 30.643609
# lon_pg_0: float = -96.337302
# h_ellp_0: float = 86.281612

#################################
####    PLANET PARAMETERS    ####
#################################

J2000: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "J2000Frame")
J2000Frame = J2000.GetBodyFixedFrame()
planetEntity: st.Entity = this.GetParam(st.VarType.entityRef, "Planet")
planetFixedFrame = planetEntity.GetBodyFixedFrame()
planetName: str = planetEntity.getName()

cameraEntity: st.Entity = this.GetParam(st.VarType.entityRef, "Camera")

imageDirLocal = os.path.join("Local", "..", "Output", planetName, dataSource, "Images")
outputDir = os.path.join(localOutput, planetName, dataSource)
imageDir = os.path.join(outputDir, "Images")

truthDataPath = os.path.join(outputDir, "truth_data_" + planetName + ".csv")
attitudeEstDataPath = os.path.join(outputDir, "attitudes_" + planetName + ".csv")
gravTruthDataPath = os.path.join(outputDir, "true_gravities_" + planetName + ".csv")
gravEstDataPath = os.path.join(outputDir, "measurements_" + planetName + ".csv")
q_i_b_DataPath = os.path.join(outputDir, "q_i_b_" + planetName + ".csv")
omega_i_b_DataPath = os.path.join(outputDir, "omega_i_b_" + planetName + ".csv")

# TODO: ellipsoid for non-Moon testing
# cameraPosPlanetFixed = st.PlanetUtils.LLA_to_PCPF(st.PlanetUtils.LatLonAlt(np.deg2rad(phi_pg_0), np.deg2rad(lon_pg_0), h_ellp_0), radiusEquatorial)
# cameraPosPlanetFixed, _ = st.ProcPlanet.SampleGround(planetData, cameraPosPlanetFixed, radiusEquatorial, 0.0, 20)
# llaResult = st.PlanetUtils.PCPF_to_LLA(cameraPosPlanetFixed, radiusEquatorial)
# lat_pc = np.rad2deg(llaResult.lat())
# lon_pc = np.rad2deg(llaResult.lon())
# h_pc = llaResult.alt()

# This should work for ellipsoid-referenced planets
radiusEquatorial: float = planetEntity.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])
radiusPolar: float = planetEntity.GetParam(st.VarType.double, ["#Planet", "General", "RadiusPolar_m"])
cameraPosPlanetFixed = planetographic_to_cartesian(phi_pg_0, lon_pg_0, h_ellp_0, radiusEquatorial, radiusPolar)

###############################
####    READ IMAGE DATA    ####
###############################

# Start getting true data and storing measurements

true_accelerations: list[npt.NDArray] = []
measured_accelerations: list[npt.NDArray] = []
q_i_b_data: list[npt.NDArray] = []
omega_i_b_data: list[npt.NDArray] = []

mu: float = 1e9 * planetEntity.GetParam(st.VarType.double, ["Dynamics", "GravitationalParameter_km3_s2"])
Omega = planetEntity.getAngVelocity().WRT(J2000Frame).ExprIn(planetFixedFrame)
st.OnScreenLogMessage(f"Planet angular velocity (Omega) = {np.rad2deg(Omega)} deg/s", "SPSTraverse", st.Severity.Info)
# Omega = np.array([0.0, 0.0, 2.66e-6])  # Expressed in the planet-fixed frame

times: list[float] = []

planetStateDummy = st.frames.FramedLocVelAcc(st.frames.rva_struct(np.zeros(3), np.zeros(3), np.zeros(3)), planetFixedFrame)
realImageCounter: int = 0

t0_datetime = datetime.datetime(2026, 1, 1)  # dummy date, replaced with first image's datetime

for realImageFile in Path(imageDir).iterdir():
    imageData = DataCapture.from_img(realImageFile)
    t0_datetime = imageData.capture_time

    # Initial time
    st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(t0_datetime))
    time.sleep(0.1)
    break  # break after one iteration just to get the first image

for realImageFile in Path(imageDir).iterdir():
    realImageCounter += 1
    st.OnScreenLogMessage(f"Processing real image {realImageCounter}", "RealSPS", st.Severity.Info)

    imageData = DataCapture.from_img(realImageFile)
    t_thisImage = imageData.capture_time
    acc_thisImage = imageData.acceleration

    dt: datetime.timedelta = t_thisImage - t0_datetime
    times.append(dt.total_seconds())

    # Let STP reset the time (don't hot loop)
    st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(t_thisImage))
    time.sleep(0.1)

    # Get true accel
    stateNow = st.frames.FramedLocVelAcc(st.frames.rva_struct(cameraPosPlanetFixed, np.zeros(3), np.zeros(3)), planetFixedFrame)
    g = st.SimGlobals.SampleVectorField("Gravity", stateNow).ExprIn(planetFixedFrame)

    g_thirdBody = st.SimGlobals.SampleVectorField("ThirdBodyGravity", stateNow).ExprIn(planetFixedFrame)
    g_thirdBody -= st.SimGlobals.SampleVectorField("ThirdBodyGravity", planetStateDummy).ExprIn(planetFixedFrame)
    # st.OnScreenLogMessage(f"True third body gravity = {g_thirdBody}", "SPSTraverse", st.Severity.Info)

    g += g_thirdBody - np.cross(Omega, np.cross(Omega, cameraPosPlanetFixed))
    g_true = copy.deepcopy(g)
    true_accelerations.append(g_true)

    # Get planet attitude and ang vel at this time
    planetRot = planetEntity.getRotation().DCM_WRT(J2000Frame)  # Passive, planet attitude WRT J2000
    planetAngVel = planetEntity.getAngVelocity().WRT(J2000Frame).ExprIn(planetFixedFrame)
    _q_i_b = st.math.DCM_to_Quat(planetRot)

    # Append data
    measured_accelerations.append(np.asarray(acc_thisImage))
    q_i_b_data.append(_q_i_b)
    omega_i_b_data.append(planetAngVel)

write_csv(gravTruthDataPath, true_accelerations)
write_csv(gravEstDataPath, measured_accelerations)
write_csv(q_i_b_DataPath, q_i_b_data)
write_csv(omega_i_b_DataPath, omega_i_b_data)

#########################################
####    OBTAIN ATTITUDE ESTIMATES    ####
#########################################

if reprocess_star_tracker:
    Path(attitudeEstDataPath).unlink(missing_ok=True)

if not Path(attitudeEstDataPath).is_file():
    ##########################
    ####    User Input    ####
    ##########################
    
    nmatch = 8 # minimum number of stars to match
    starMatchPixelTol = 1 # pixel match tolerance
    min_star_area = 12 # minimum pixel area for a star
    max_star_area = 200 # maximum pixel area for a star
    max_num_stars_to_process = 40 # maximum number of centroids to attempt to match per image

    low_thresh_pxl_intensity = None
    hi_thresh_pxl_intensity = None

    VERBOSE = True # set True for prints on results
    graphics = False # set True for graphics throughout the solve process

    data_path = os.path.join(thisRepo, 'data') # full path to your data
    # cam_config_file_path = os.path.join(thisRepo, 'data', 'cam_config', 'Custom_cam.json') # full path (including filename) of your cam config file
    cam_config_file_path = os.path.join(thisRepo, 'CameraCalibration', 'canoneosrebelt7_calibration_10_16.json') # full path (including filename) of your cam config file
    darkframe_file_path = os.path.join(thisRepo, 'Images', 'darkframes', 'darkframe.png') # full path (including filename) of your darkframe file
    image_extension = ".jpg" # the image extension to search for in the data_path directory
    cat_prefix ='' # if the catalog has a prefix, define it here

    #################################
    ####    Support Functions    ####
    #################################

    st.OnScreenLogMessage(f'imgSourceDir = {imageDir}', "SPSTraverse", st.Severity.Info)

    ###################################
    ####    Process Star Images    ####
    ###################################

    # Load star tracker and catalog data
    if darkframe_file_path == '': darkframe_file_path = None
    if darkframe_file_path is not None:
        if not os.path.exists(darkframe_file_path):
            darkframe_file_path = None
            st.OnScreenLogMessage("unable to find provided darkframe file, proceeding without one...", "SPSTraverse", st.Severity.Info)
        else:    st.OnScreenLogMessage("darkframe file: " + darkframe_file_path, "SPSTraverse", st.Severity.Info)
    else:    st.OnScreenLogMessage("no darkframe file provided, proceeding without one...", "SPSTraverse", st.Severity.Info)

    k = np.load(os.path.join(data_path, cat_prefix+'k.npy'))
    m = np.load(os.path.join(data_path, cat_prefix+'m.npy'))
    q = np.load(os.path.join(data_path, cat_prefix+'q.npy'))
    x_cat = np.load(os.path.join(data_path, cat_prefix+'u.npy'))
    indexed_star_pairs = np.load(os.path.join(data_path, cat_prefix+'indexed_star_pairs.npy'))

    cam_file = cam_config_file_path
    camera_matrix, _, _ = read_cam_json(cam_file)
    dx = camera_matrix[0, 0]
    isa_thresh = starMatchPixelTol*(1/dx)

    st.OnScreenLogMessage(f"isa_thresh = {isa_thresh}", "RealSPS", st.Severity.Info)

    # Define structures for data capture
    image_name = []
    ttime = []
    stemp = []
    sram  = []
    scpu  = []
    solve_time = []
    qs = []
    qv0 = []
    qv1 = []
    qv2 = []

    # Create list of all images in target dir
    total_start = time.time()

    dir_contents = os.listdir(imageDir)
    for i in range(len(dir_contents)):
        dir_contents[i] = imageDir + "/" + dir_contents[i]
    dir_contents.sort()

    image_names = []

    for item in dir_contents:
        if image_extension in item:
            image_names+=[os.path.abspath(item)]

    idx: int = 0
    for image_filename in image_names:
        image_name += [image_filename]
        # st.OnScreenLogMessage("===================================================", "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(image_filename, "SPSTraverse", st.Severity.Info)

        #run star tracker
        solve_start_time = time.time()

        q_est, idmatch, nmatches, x_obs, rtrnd_img = main.star_tracker(
                image_filename, cam_file, m=m, q=q, x_cat=x_cat, k=k, indexed_star_pairs=indexed_star_pairs, darkframe_file=darkframe_file_path, 
                min_star_area=min_star_area, max_star_area=max_star_area, isa_thresh=isa_thresh, nmatch=nmatch, n_stars=max_num_stars_to_process,
                low_thresh_pxl_intensity=low_thresh_pxl_intensity,hi_thresh_pxl_intensity=hi_thresh_pxl_intensity,graphics=graphics,verbose=VERBOSE, watchdog=5)

        solve_time += [time.time()-solve_start_time]

        # Collect data
        try:
            assert not np.any(np.isnan(q_est))
            if VERBOSE:
                st.OnScreenLogMessage('est q: ' + str(q_est)+'\n', "SPSTraverse", st.Severity.Info)
            q_rotate = np.array([0.5, -0.5, 0.5, 0.5])  # w-last quaternion
            q_est = quat_mult(q_est, q_rotate)  # w-last quaternion
            qs += [q_est[3]]
            qv0 += [-q_est[0]]
            qv1 += [-q_est[1]]
            qv2 += [-q_est[2]]
        except AssertionError:
            if VERBOSE:
                st.OnScreenLogMessage('NO VALID STARS FOUND\n', "SPSTraverse", st.Severity.Info)
            qs += [999]
            qv0 += [999]
            qv1 += [999]
            qv2 += [999]

        ttime += [time.time()]
        sram  += [psutil.virtual_memory().percent]
        #scpu  += [psutil.cpu_percent(2)]
        scpu  += [psutil.cpu_percent()]

        st.OnScreenLogMessage(f'Completed image {idx} ({round(float(idx) / float(len(image_names)) * 100.0, 2)} %)', "SPSTraverse", st.Severity.Info)
        idx += 1

    data = {'image name':image_name,'time':ttime,'RAM':sram,'CPU':scpu,'image solve time (s)':solve_time, 'qs':qs,'qv0':qv0,'qv1':qv1,'qv2':qv2}

    now = str(datetime.datetime.now())
    now = now.split('.')
    now = now[0]
    now = now.replace(' ','_')
    now = now.replace(':','-')

    #write stuff
    keys=sorted(data.keys())

    with open(attitudeEstDataPath,'w', newline='') as csv_file:
        writer=csv.writer(csv_file)
        writer.writerow(keys)
        writer.writerows(zip(*[data[key] for  key in keys]))

    st.OnScreenLogMessage("\n\n took " + str(time.time()-total_start) + " seconds to complete \n\n", "SPSTraverse", st.Severity.Info)
    st.OnScreenLogMessage("data saved to: " + attitudeEstDataPath, "SPSTraverse", st.Severity.Info)

else:
    st.OnScreenLogMessage("Quaternion measurements already processed; skipping processing step...\n", "SPSTraverse", st.Severity.Info)

# Get data from files
attitudeEstData = read_csv(attitudeEstDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
gravTruthData = read_csv(gravTruthDataPath)
gravEstData = read_csv(gravEstDataPath)
q_i_b_list = read_csv(q_i_b_DataPath)
omega_i_b_list = read_csv(omega_i_b_DataPath)

# Very basic error handling if datasets are not the same length
if not (len(attitudeEstData) == len(gravEstData)):
    st.OnScreenLogMessage(f'Warning: early exit due to dataset length mismatch; attitudeEstData length = {len(attitudeEstData)}, and gravEstData length = {len(gravEstData)}.', "SPSTraverse", st.Severity.Info)
    exit(0)

# Initialize all inertial-to-planet attitude matrices
T_i_b_list: list[npt.NDArray] = []
angvel_i_b_list: list[npt.NDArray] = []
T_i_c_list: list[npt.NDArray] = []
g_est_list: list[npt.NDArray] = []
for i in range(len(times)):
    T_i_b_list.append(st.math.Quat_to_DCM(normalize(q_i_b_list[i])))
    angvel_i_b_list.append(np.array([omega_i_b_list[i][0], omega_i_b_list[i][1], omega_i_b_list[i][2]]))
    q_i_c = np.array([attitudeEstData[i][1], attitudeEstData[i][2], attitudeEstData[i][3], attitudeEstData[i][0]])
    T_i_c_list.append(st.math.Quat_to_DCM(normalize(q_i_c)))

    # lat_pcpf_meas, lon_pcpf_meas = T_to_latlon((T_i_c_list[-1] @ T_i_b_list[-1].T).T)
    # st.OnScreenLogMessage(f"Latitude = {lat_pcpf_meas}, Longitude = {lon_pcpf_meas}", "SPSTraverse", st.Severity.Info)
    # st.OnScreenLogMessage(f"q_i_b = {q_i_b_list[i]}", "SPSTraverse", st.Severity.Info)
    # st.OnScreenLogMessage(f"q_i_c = {q_i_c}", "SPSTraverse", st.Severity.Info)
    g_est_list.append(gravEstData[i])

###############################
####    SPS ALIGNMENT    ####
###############################

eps: float = 1.0e-4
T_alignment = np.identity(3)
if doAlignment:
    # phi_pc, lon_pc, _ = r_to_latlonalt(cameraPosPlanetFixed, radiusEquatorial)

    def SampleTrueGravity(pos_SPS_PCPF: npt.NDArray, j: int, _gravTruthData: list[npt.NDArray]) -> npt.NDArray:
        return _gravTruthData[j]
    
    # def SampleTrueGravity(pos_SPS_PCPF: npt.NDArray, j: int, _gravTruthData: list[npt.NDArray]) -> npt.NDArray:
    #     rva = st.frames.rva_struct(pos_SPS_PCPF, np.zeros(3), np.zeros(3))
    #     framedGrav = st.SimGlobals.SampleVectorField("Gravity", st.frames.FramedLocVelAcc(rva, planetFixedFrame))
    #     return framedGrav.ExprIn(planetFixedFrame)

    SampleTrueGravity_Wrapped = lambda pos, j : SampleTrueGravity(pos, j, gravTruthData)
    T_alignment = st.ProcPlanet.SPS.CalculateAlignment(alignmentCutoff, cameraPosPlanetFixed, 
        T_i_c_list, T_i_b_list, g_est_list, SampleTrueGravity_Wrapped, eps)

    # gravityDiffs: list[npt.NDArray] = [np.asarray(g_est_list[j]) - np.asarray(gravTruthData[j]) for j in range(alignmentCutoff)]
    # gravityDiffs_x: list[float] = [gravityDiffs[j][0] for j in range(alignmentCutoff)]
    # gravityDiffs_y: list[float] = [gravityDiffs[j][1] for j in range(alignmentCutoff)]
    # gravityDiffs_z: list[float] = [gravityDiffs[j][2] for j in range(alignmentCutoff)]

    # numBins: int = int(alignmentCutoff / 2)
    # counts_x, bin_edges_x = np.histogram(gravityDiffs_x, bins=numBins, density=True)
    # counts_y, bin_edges_y = np.histogram(gravityDiffs_x, bins=numBins, density=True)
    # counts_z, bin_edges_z = np.histogram(gravityDiffs_x, bins=numBins, density=True)

    # bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2.0
    # bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2.0
    # bin_centers_z = (bin_edges_z[:-1] + bin_edges_z[1:]) / 2.0

    # p_x, cov_x = curve_fit(pdf_mix, bin_centers_x, counts_x)
    # p_y, cov_y = curve_fit(pdf_mix, bin_centers_y, counts_y)
    # p_z, cov_z = curve_fit(pdf_mix, bin_centers_z, counts_z)

    # st.OnScreenAlert(f"p_x = {p_x}", "SPSGravityStatistics", st.Severity.Warning)
    # st.OnScreenAlert(f"p_y = {p_y}", "SPSGravityStatistics", st.Severity.Warning)
    # st.OnScreenAlert(f"p_z = {p_z}", "SPSGravityStatistics", st.Severity.Warning)

if doAngVel:
    st.OnScreenAlert(f"Made it to ang vel processing!", "SPS", st.Severity.Warning)

    T_unrotate: list[npt.NDArray] = []
    Q_unrotate: list[npt.NDArray] = []
    T_planet: list[npt.NDArray] = []
    q_planet: list[npt.NDArray] = []
    
    quatEstimates = read_quats(attitudeEstDataPath)[:alignmentCutoff]
    for i in range(alignmentCutoff):
        _T_planet = T_i_b_list[i]
        _T_i_c = st.math.Quat_to_DCM(normalize(quatEstimates[i]))
        _T_unrotate = _T_planet @ _T_i_c.T
        T_unrotate.append(_T_unrotate)
        T_planet.append(_T_planet)

        _q_planet = st.math.DCM_to_Quat(_T_planet)
        if _q_planet[3] < 0.0:
            for qq in range(4):
                _q_planet[qq] = -_q_planet[qq]
        q_planet.append(_q_planet)

        _q_unrotate = positivize_quat(st.math.DCM_to_Quat(_T_unrotate))
        Q_unrotate.append(_q_unrotate)
        
    st.OnScreenAlert(f"Made it to ang vel bulk estimation!", "SPS", st.Severity.Warning)

    # omega_est, omega_hist = estimate_omega_EnrightForm(quatEstimates, alignmentTimeStep_s, T_unrotate)

    q_unrotate_avg = average_quats(Q_unrotate)
    T_unrotate_avg = st.math.Quat_to_DCM(normalize(q_unrotate_avg))

    omega_est = -estimate_omega_QuateRA(quatEstimates, times[:alignmentCutoff])
    omega_est_QuateRA = T_unrotate_avg @ omega_est

    st.OnScreenLogMessage(f"omega_true           = {np.rad2deg(Omega)} deg/s", "SPSTraverse", st.Severity.Info)
    st.OnScreenLogMessage(f"omega_est  (QuateRA) = {np.rad2deg(omega_est_QuateRA)} deg/s", "SPSTraverse", st.Severity.Info)
    st.OnScreenLogMessage(f"omega diff (QuateRA) = {100.0 * (omega_est_QuateRA - Omega) / Omega} %", "SPSTraverse", st.Severity.Info)

# exit(0)

st.leave_sim()
