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


# Statistics
def pdf_gauss(x: float | npt.NDArray, mean: float, std: float):
    return (1.0 / np.sqrt(2.0 * np.pi * std ** 2)) * np.exp(-((x - mean) ** 2) / (std ** 2))


def pdf_cauchy(x: float | npt.NDArray, median: float, gamma: float):
    return (1.0 / np.pi) * (gamma / ((x - median) ** 2  + gamma ** 2))


def pdf_mix(x: float | npt.NDArray, x0: float, gamma: float, factor: float):
    return factor * pdf_cauchy(x, x0, gamma) + (1.0 - factor) * pdf_gauss(x, x0, gamma)


def positivize_quat(q: npt.NDArray) -> npt.NDArray:
    if q[3] > 0.0:
        return q
    else:
        return np.array([-q[0], -q[1], -q[2], -q[3]])


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


os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

##################################
####    PLANET DATA IMPORT    ####
##################################

planetData = st.ProcPlanet.DataStore()

moon_Global_Data = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014")
moon_LSP75_Data = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m")
moon_LSP875_Data = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m")
moon_Apollo15_Data = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Local/Apollo15/LRO_NAC_DEM_Apollo_15_26N004E_150cmp")
moon_Apollo17_Data = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Local/Apollo17/APOLLO17_DTM_150CM")
args = st.ProcPlanet.GeoBin_Extra_Args()
args.cubicInterp = True

planetData.AddGeoBinAltimetryLayer(1.0, moon_Global_Data, args)
planetData.AddGeoBinAltimetryLayer(1.0, moon_LSP75_Data, args)
planetData.AddGeoBinAltimetryLayer(1.0, moon_LSP875_Data, args)

#####################################
####    CLASSES AND FUNCTIONS    ####
#####################################

def rad_to_arcsec(rad: float) -> float:
    return 3600.0 * np.rad2deg(rad)


def is_dir_empty(path):
    # Returns True if empty, False otherwise
    return not any(Path(path).iterdir())


np.set_printoptions(suppress=True)

#################################
####    GLOBAL PARAMETERS    ####
#################################

alignmentCutoff: int = 300
endpoint: int = alignmentCutoff + 200

numImages: int = alignmentCutoff + 200

##############################
####    SITE SELECTION    ####
##############################

# site: str = "Ideal"
# site: str = "Apollo11"
# site: str = "Apollo15"
# site: str = "Apollo17"
site: str = "ConnectingRidge"
# site: str = "NobileRim1"

hash_object = hashlib.sha256(site.encode('utf-8'))
int_seed = int(hash_object.hexdigest(), 16)

#################################
####    PLANET PARAMETERS    ####
#################################

this = st.GetThisSystem()

J2000: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "J2000Frame")
J2000Frame = J2000.GetBodyFixedFrame()

planetEntity: st.Entity = this.GetParam(st.VarType.entityRef, "Planet")
planetFixedFrame = planetEntity.GetBodyFixedFrame()
planetName: str = planetEntity.getName()
planetRadius: float = planetEntity.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])

cameraEntity: st.Entity = this.GetParam(st.VarType.entityRef, "Camera")

renderDirLocal = os.path.join("Local", "..", "Output", planetName, site, "Renders")
outputDir = os.path.join(localOutput, planetName, site)
renderDir = os.path.join(outputDir, "Renders")

cameraTraversePositions_path = os.path.join(outputDir, "cameraTraversePositions_" + planetName + ".csv")
cameraEstimatedPositions_path = os.path.join(outputDir, "cameraEstimatedPositions_" + planetName + ".csv")

cameraTraversePositions = read_csv(cameraTraversePositions_path)
cameraEstimatedPositions = read_csv(cameraEstimatedPositions_path)

cameraTraversePoints: list[npt.NDArray] = []
cameraEstimatedPoints: list[npt.NDArray] = []
for i in range(len(cameraTraversePositions)):
    cameraTraversePoints.append(0.001 * st.ProcPlanet.SampleGround(planetData, cameraTraversePositions[i], planetRadius, 2.0, 16)[0])
    cameraEstimatedPoints.append(0.001 * st.ProcPlanet.SampleGround(planetData, cameraEstimatedPositions[i], planetRadius, 2.0, 16)[0])

curve = st.vis.curves.DrawCurve(planetEntity, color = [0.05, 0.95, 0.1, 1.0], closed = False, points = cameraTraversePoints, thickness = 5000.0)
curve = st.vis.curves.DrawCurve(planetEntity, color = [0.05, 0.1, 0.95, 1.0], closed = False, points = cameraEstimatedPoints, thickness = 5000.0)

exit_flag: bool = False
while not exit_flag:
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

st.leave_sim()
