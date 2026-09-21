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
sys.path.append(thisRepo)

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

    for i in range(1, N):
        # q_prev = st.math.DCM_to_Quat(st.math.Quat_to_DCM(normalize(quats[i - 1])) @ T_unrotate[i - 1])
        # q_curr = st.math.DCM_to_Quat(st.math.Quat_to_DCM(normalize(quats[i])) @ T_unrotate[i])
        q_prev = quats[i - 1]
        q_curr = quats[i]
        # q_next = st.math.DCM_to_Quat(T_unrotate[i + 1] @ st.math.Quat_to_DCM(normalize(quats[i + 1])))

        delta_q = quat_mult(q_curr, quat_inverse(q_prev))
        angle: float = 2.0 * np.arccos(delta_q[3])
        axis: npt.NDArray = delta_q[0:3] / np.sin(0.5 * angle)

        # q_unrotate_prev = st.math.DCM_to_Quat(T_unrotate[i - 1])
        # q_unrotate_curr = st.math.DCM_to_Quat(T_unrotate[i])
        # T_unrotate_avg = st.math.Quat_to_DCM(normalize(0.5 * (q_unrotate_prev + q_unrotate_curr)))
        # omega_i = T_unrotate_avg @ (angle * axis / deltaT)

        omega_i = T_unrotate[i] @ (-angle * axis / deltaT)

        # omega_i = angle * axis / deltaT
        omega_history.append(omega_i)
        omega_sum += omega_i

    return omega_sum / float(len(omega_history)), omega_history


os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

##################################
####    PLANET DATA IMPORT    ####
##################################

planetData = st.ProcPlanet.DataStore()

moonGlobalData = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014")
args = st.ProcPlanet.GeoBin_Extra_Args()
args.cubicInterp = True

planetData.AddGeoBinAltimetryLayer(1.0, moonGlobalData, args)

# time.sleep(5.0)


# Wait for Eridani to load (TODO: probably don't need this because 
# it's guaranteed to only start after "init" is done on all systems?)

# st.OnScreenLogMessage("Waiting for Eridani to load...", "SPSTraverse", st.Severity.Info)
# st.GetThisSystem().AddOrSetParam(st.VarType.bool, "Ready", False)
# def SetIsReady(paramMap: st.ParamMap, timeNow: st.timestamp):
#     st.GetThisSystem().SetParam(st.VarType.bool, "Ready", True)
# st.SimGlobals.Subscribe("EridaniLoadingComplete", SetIsReady)
# while not st.GetThisSystem().GetParam(st.VarType.bool, "Ready"):
#     pass

# st.OnScreenLogMessage("Got past the initial wait time!", "SPSTraverse", st.Severity.Info)

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

regenerateStarCatalog: bool     = False  # DO NOT USE THIS; CURRENTLY BROKEN
delete_old: bool                = True
reprocess_star_tracker: bool    = True
doAlignment: bool               = True
doAngVel: bool                  = True
globalDoPrint: bool             = True

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

phi_pg_0: float = 0.0
lon_pg_0: float = 0.0
h_ellp_0: float = 0.0

if site == "Ideal":
    # Something with low gravity variation (ideal)
    phi_pg_0 = 4.0
    lon_pg_0 = -29.4
    h_ellp_0 = 0.0

elif site == "Apollo11":
    # Something with low gravity variation (ideal)
    phi_pg_0 = 0.67
    lon_pg_0 = 23.47
    h_ellp_0 = -1500.0

elif site == "Apollo15":
    # Something like Apollo 15
    phi_pg_0 = 26.13
    lon_pg_0 = 3.63
    h_ellp_0 = -1931.0

elif site == "Apollo17":
    # Something like Apollo 17
    phi_pg_0 = 20.19
    lon_pg_0 = 30.77
    h_ellp_0 = -2500.0

elif site == "ConnectingRidge":
    # Something like Connecting Ridge
    phi_pg_0 = -89.45
    lon_pg_0 = -137.2
    h_ellp_0 = 1960.0

elif site == "NobileRim1":
    # Something like Nobile Rim 1
    phi_pg_0 = -85.4
    lon_pg_0 = 35.3
    h_ellp_0 = 1743.0

#################################
####    PLANET PARAMETERS    ####
#################################

J2000: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "J2000Frame")
J2000Frame = J2000.GetBodyFixedFrame()
planetEntity: st.Entity = st.GetThisSystem().GetParam(st.VarType.entityRef, "Planet")
planetFixedFrame = planetEntity.GetBodyFixedFrame()
planetName: str = planetEntity.getName()

cameraEntity: st.Entity = st.GetThisSystem().GetParam(st.VarType.entityRef, "Camera")

renderDirLocal = os.path.join("Local", "Repos", "SPS-Star-Tracker", "output", planetName, site, "Renders")
outputDir = os.path.join(thisRepo, "output", planetName, site)
renderDir = os.path.join(outputDir, "Renders")

truthDataPath = os.path.join(outputDir, "truth_data_" + planetName + ".csv")
attitudeEstDataPath = os.path.join(outputDir, "attitudes_" + planetName + ".csv")
gravTruthDataPath = os.path.join(outputDir, "true_gravities_" + planetName + ".csv")
gravEstDataPath = os.path.join(outputDir, "measurements_" + planetName + ".csv")
q_i_b_DataPath = os.path.join(outputDir, "q_i_b_" + planetName + ".csv")
omega_i_b_DataPath = os.path.join(outputDir, "omega_i_b_" + planetName + ".csv")

#############################
####    ERROR SOURCES    ####
#############################

# Random number generator
rng = np.random.default_rng(int_seed + 200)

addMeasurementBias: bool = False
addMeasurementNoise: bool = True

sigma_2 = np.array([[1e-8, 0.0, 0.0],
                    [0.0, 1e-8, 0.0],
                    [0.0, 0.0, 1e-8]])  # Worst case for the BMA220 IMU (once averaged out)
biasSigma_2 = np.array([[0.15 ** 2, 0.0, 0.0],
                        [0.0, 0.15 ** 2, 0.0],
                        [0.0, 0.0, 0.15 ** 2]])
bias = np.linalg.cholesky(biasSigma_2) @ rng.normal(0.0, 1.0, size=3)

############################
####    RENDER SETUP    ####
############################

# Delete all old images
if delete_old:
    if os.path.exists(renderDir):
        shutil.rmtree(renderDir)
    Path(renderDir).mkdir(parents=True)

# Time
t0: st.timestamp = st.timestamp.from_datetime(datetime.datetime(year=2026, month=5, day=22, hour=16))
st.SimGlobals.SimClock.ResetTo(t0)

time.sleep(0.1)

alignmentTimeStep_s: float = 1.0
traverseTimeStep_s: float = 100.0

true_data: list[npt.NDArray] = []
true_accelerations: list[npt.NDArray] = []
measured_accelerations: list[npt.NDArray] = []
q_i_b_data: list[npt.NDArray] = []
omega_i_b_data: list[npt.NDArray] = []

times_alignment = alignmentTimeStep_s * np.linspace(0.0, alignmentCutoff - 1, alignmentCutoff)
times_traverse = times_alignment[-1] + traverseTimeStep_s + traverseTimeStep_s * np.linspace(
    0.0, endpoint - alignmentCutoff - 1, endpoint - alignmentCutoff)
times = np.concat((times_alignment, times_traverse))

mu: float = 1e9 * planetEntity.GetParam(st.VarType.double, ["Dynamics", "GravitationalParameter_km3_s2"])
radiusEquatorial: float = planetEntity.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])
Omega = planetEntity.getAngVelocity().WRT(J2000Frame).ExprIn(planetFixedFrame)

st.OnScreenLogMessage(f"Planet angular velocity (Omega) = {np.rad2deg(Omega)} deg/s", "SPSTraverse", st.Severity.Info)
# Omega = np.array([0.0, 0.0, 2.66e-6])  # Expressed in the planet-fixed frame

startTime = time.perf_counter()
elapsedSeconds: float = 0.0
printInterval: int = 10

# TODO: ellipsoid for non-Moon testing
cameraPosPlanetFixed = st.PlanetUtils.LLA_to_PCPF(st.PlanetUtils.LatLonAlt(np.deg2rad(phi_pg_0), np.deg2rad(lon_pg_0), h_ellp_0), radiusEquatorial)
cameraPosPlanetFixed, _ = st.ProcPlanet.SampleGround(planetData, cameraPosPlanetFixed, radiusEquatorial, 0.0, 20)
llaResult = st.PlanetUtils.PCPF_to_LLA(cameraPosPlanetFixed, radiusEquatorial)
lat_pc = np.rad2deg(llaResult.lat())
lon_pc = np.rad2deg(llaResult.lon())
h_pc = llaResult.alt()

###################################
####    TRAVERSE PARAMETERS    ####
###################################

# Assume crew takes SPS measurements every X meters. To find X, assume 0.5 m/s walking 
# speed based on Apollo estimates, science objectives slowing down the crew, etc.
traverseStep_m: float = 50.0

# Assume SPS measurement time per stop is 60 seconds
# TODO: Unimplemented for now
traverseMeasurementWaitTime: float = 60.0

# Assume crew deviates from planned traverse by some Gaussian noise with a 1-sigma of 5 meters per 50-meter step
traverseFollowingError_m_1sigma: float = 5.0

# Traverse direction (roughly horizontal at location)
cameraPosNWU = st.PlanetUtils.NorthWestUpFromLocation(cameraPosPlanetFixed, radiusEquatorial)
cameraPosFLU = st.PlanetUtils.ForwardLeftUpFromAzimuth(cameraPosPlanetFixed, 
                                                        rng.uniform(0.0, 2.0 * np.pi), 
                                                        radiusEquatorial)
traverseDirection: npt.NDArray = cameraPosFLU.forward()

################################
####    EXECUTE TRAVERSE    ####
################################

positions = []
pos_i = copy.deepcopy(cameraPosPlanetFixed)
for i in range(numImages):
    positions.append(copy.deepcopy(pos_i))
    if i >= alignmentCutoff:
        pos_i += traverseDirection * traverseStep_m + rng.normal(0.0, traverseFollowingError_m_1sigma, size=3)
        pos_i, _ = st.ProcPlanet.SampleGround(planetData, pos_i, radiusEquatorial, 0.0, 20)

#######################################
####    REGENERATE STAR CATALOG    ####
#######################################

if regenerateStarCatalog:
    import py_src.star_tracker.star_tracker.ground as ground
    import py_src.star_tracker.star_tracker.cam_matrix as cam_matrix

    b_thresh = 6.0
    excess_rows = [53, 54]
    # column (0-indexing) containing the Hipparcos ID number
    index_col = 2
    repo_dir = st.path_utils.AssetPathToReal(st.AssetType.Generic, "Local/Repos/SPS-Star-Tracker")
    starcat_file = os.path.join(repo_dir, 'data', 'starcat.tsv')
    cam_config_dir = os.path.join(repo_dir, 'data', 'cam_config')
    cam_config_file = 'perfect_cam.json' # the name of the camera config file in /data
    cam_config_file = os.path.join(cam_config_dir, cam_config_file)
    camera_matrix, cam_resolution, dist_coefs = cam_matrix.read_cam_json(cam_config_file)

    save_vals = True
    save_dir = os.path.join(repo_dir, 'data')

    nrow = cam_resolution[1]
    ncol = cam_resolution[0]
    fov = cam_matrix.cam2fov(cam_matrix.cam_matrix_inv(camera_matrix), nrow, ncol)

    simTimeNow: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    simTimeAstropy = astrotime.Time(simTimeNow, format='datetime')
    planetLoc = planetEntity.getLocation().WRT_ExprIn(J2000Frame)

    st.OnScreenLogMessage("Got to star catalog creation!", "SPSStarCatalog", st.Severity.Info)

    ground.create_star_catalog(starcat_file=starcat_file, brightness_thresh=b_thresh,
                               excess_rows=excess_rows, index_col=index_col, fov=fov,
                               save_vals=save_vals, rB=np.array([0.001*planetLoc]).T, save_dir=save_dir, t=simTimeAstropy)
    
    st.OnScreenLogMessage("Got past star catalog creation!", "SPSStarCatalog", st.Severity.Info)

#############################
####    RENDER IMAGES    ####
#############################

if not os.path.exists(renderDir):
    Path(renderDir).mkdir(parents=True)

if is_dir_empty(renderDir):
    # locs: list[npt.NDArray] = []
    # vels: list[npt.NDArray] = []
    # rots: list[npt.NDArray] = []
    # names: list[str] = []
    
    planetStateDummy = st.frames.FramedLocVelAcc(st.frames.rva_struct(np.zeros(3), np.zeros(3), np.zeros(3)), planetFixedFrame)
    for i in range(numImages):
        doPrint: bool = i % printInterval == 0 and globalDoPrint

        # Step time forward by the correct dt
        tNow: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()

        if i < alignmentCutoff:
            tNow += datetime.timedelta(seconds=alignmentTimeStep_s)
        else:
            tNow += datetime.timedelta(seconds=traverseTimeStep_s)

        st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tNow))
        time.sleep(0.1)

        # Sample gravity vector
        positionNow = positions[i]
        stateNow = st.frames.FramedLocVelAcc(st.frames.rva_struct(positionNow, np.zeros(3), np.zeros(3)), planetFixedFrame)
        g = st.SimGlobals.SampleVectorField("Gravity", stateNow).ExprIn(planetFixedFrame)

        g_thirdBody = st.SimGlobals.SampleVectorField("ThirdBodyGravity", stateNow).ExprIn(planetFixedFrame)
        g_thirdBody -= st.SimGlobals.SampleVectorField("ThirdBodyGravity", planetStateDummy).ExprIn(planetFixedFrame)
        st.OnScreenLogMessage(f"True third body gravity = {g_thirdBody}", "SPSTraverse", st.Severity.Info)

        g += g_thirdBody - np.cross(Omega, np.cross(Omega, positionNow))
        g_true = copy.deepcopy(g)
        g_true_framed = st.frames.FramedVector(g_true, planetFixedFrame)
        
        lat_pc, lon_pc, h_pc = r_to_latlonalt(positionNow, radiusEquatorial)
        T_P_G = latlon_to_T(lat_pc, lon_pc).T
        g_IMU_frame = (T_P_G @ np.array([g]).T).T[0]

        if addMeasurementBias:
            g_IMU_frame += bias

        if addMeasurementNoise:
            g_IMU_frame += np.linalg.cholesky(sigma_2) @ rng.normal(0.0, 1.0, size=3)
        
        measured_accelerations.append(g_IMU_frame)

        planetRot = planetEntity.getRotation().DCM_WRT(J2000Frame)  # Passive, planet attitude WRT J2000
        planetAngVel = planetEntity.getAngVelocity().WRT(J2000Frame).ExprIn(planetFixedFrame)
        _q_i_b = st.math.DCM_to_Quat(planetRot)
        q_i_b_data.append(_q_i_b)
        omega_i_b_data.append(planetAngVel)

        gInertial_true = g_true_framed.ExprIn(J2000Frame)
        # gInertial_true = (planetRot.T @ np.array([g_true]).T).T[0]
        true_accelerations.append(g_true)

        g_measured_planetFixed = (T_P_G.T @ np.array([g_IMU_frame]).T).T[0]
        g_measured_framed = st.frames.FramedVector(g_measured_planetFixed, planetFixedFrame)
        gInertial = g_measured_framed.ExprIn(J2000Frame)

        # st.OnScreenLogMessage(f'g_true                 = {g_true}', "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f'gInertial_true         = {gInertial_true}', "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f'g_measured_planetFixed = {g_measured_planetFixed}', "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f'gInertial              = {gInertial}', "SPSTraverse", st.Severity.Info)

        # gInertial = (planetRot.T @ T_P_G.T @ np.array([g_IMU_frame]).T).T[0]
        ra_true_i, de_true_i = r_hat_to_ra_dec(-normalize(gInertial_true))
        ra, de = r_hat_to_ra_dec(-normalize(gInertial))
        ra_pcpf, de_pcpf = r_hat_to_ra_dec(-normalize(g_true))
        ra_meas_pcpf, de_meas_pcpf = r_hat_to_ra_dec(-normalize(g_measured_planetFixed))
        
        # st.OnScreenLogMessage(f'_q_i_b       = {_q_i_b}', "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f'ra_true_i    = {ra_true_i}, de_true_i    = {de_true_i}', "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f'ra           = {ra}, de           = {de}', "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f'ra_pcpf      = {ra_pcpf}, de_pcpf      = {de_pcpf}', "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f'ra_meas_pcpf = {ra_meas_pcpf}, de_meas_pcpf = {de_meas_pcpf}', "SPSTraverse", st.Severity.Info)

        # Passive transform from inertial frame to surface (grav vector) frame (x axis along -grav)
        # We get this by converting a pointing vec in inertial frame to ra/dec, then converting
        # ra/dec to rotation matrix (Euler 321 with negative dec, 0 roll), then transposing.
        T_I_S_true = ra_dec_to_rot(ra_true_i, de_true_i).T
        q_I_S_true_passive = st.math.DCM_to_Quat(T_I_S_true)

        T_PCPF_S_true = ra_dec_to_rot(ra_pcpf, de_pcpf).T
        q_PCPF_S_true_passive = st.math.DCM_to_Quat(T_PCPF_S_true)
        # q_c_b_true = st.math.DCM_to_Quat(rotMat_pcpf.T)
        # st.OnScreenLogMessage(f"Latitude = {de_pcpf}, Longitude = {ra_pcpf}", "SPSTraverse", st.Severity.Info)

        # Render (old version)
        pos_framed = st.frames.FramedLoc(positionNow, planetFixedFrame)
        vel_framed = st.frames.FramedLocVel(st.frames.rv_struct(positionNow, np.zeros(3)), planetFixedFrame)
        # rot_framed = st.frames.FramedRot(rotMat_pcpf, planetFixedFrame)

        # rotQuatInertial = rot_framed.Quat_WRT(J2000Frame)
        # st.OnScreenLogMessage(f"True inertial attitude = {q_I_S_true_passive}", "SPSTraverse", st.Severity.Info)
        # st.OnScreenLogMessage(f"True q_c_b             = {q_PCPF_S_true_passive}", "SPSTraverse", st.Severity.Info)

        cameraEntity.setRotation(st.frames.FramedRot(T_I_S_true, J2000Frame))

        EridaniRenderPayload = st.ParamMap()
        EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Loc", pos_framed.WRT_ExprIn(J2000Frame))
        EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Vel", vel_framed.vel_WRT_ExprIn(J2000Frame))
        EridaniRenderPayload.AddParam(st.VarType.doubleV4, "Rot", q_I_S_true_passive)
        EridaniRenderPayload.AddParam(st.VarType.entityRef, "Frame", J2000)
        EridaniRenderPayload.AddParam(st.VarType.string, "NameOverride", "SPSRender" + str(i).zfill(5))
        EridaniRenderPayload.AddParam(st.VarType.string, "OutputPathOverride", str(renderDirLocal))
        payload = st.SimGlobals.Request("EridaniSingleRender", EridaniRenderPayload, timeout=datetime.timedelta(seconds=60.0))

        # Append properties for batch render
        # locs.append(positionNow)
        # vels.append(np.zeros(3))
        # rots.append(st.math.DCM_to_Quat(planetRot @ ra_dec_to_rot(ra, de)))
        # names.append("SPSRender" + str(i).zfill(5))

        if doPrint:
            st.OnScreenLogMessage(f'Rendering image {i} of {numImages} ({round(100.0 * float(i) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}', "SPSTraverse", st.Severity.Info)
            # st.OnScreenLogMessage(f'Recording data for image {i} of {numImages} ({round(100.0 * float(i) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}', "SPSTraverse", st.Severity.Info)

        # true_data.append(st.math.DCM_to_Quat(ra_dec_to_rot(ra_true_i, de_true)))
        true_data.append(q_I_S_true_passive)

        endTime = time.perf_counter()
        elapsedSeconds = endTime - startTime
        elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

        if doPrint:
            projectedRemainingSeconds: float = elapsedSeconds * float(numImages - i + 1) / float(i + 1)
            projectedRemainingTime = datetime.timedelta(seconds=round(projectedRemainingSeconds))
            st.OnScreenLogMessage(f'Elapsed time: {elapsedTime}. Remaining time estimate: {projectedRemainingTime}\n', "SPSTraverse", st.Severity.Info)

    # Batch render (new version)
    # EridaniBatchRenderPayload = st.ParamMap()
    # EridaniBatchRenderPayload.AddParamArray(st.VarType.doubleV3, "Locs", locs)
    # EridaniBatchRenderPayload.AddParamArray(st.VarType.doubleV3, "Vels", vels)
    # EridaniBatchRenderPayload.AddParamArray(st.VarType.doubleV4, "Rots", rots)
    # EridaniBatchRenderPayload.AddParam(st.VarType.entityRef, "Frame", planetEntity)
    # EridaniBatchRenderPayload.AddParamArray(st.VarType.string, "NameOverrides", names)
    # payload = st.SimGlobals.Request("EridaniBatchRender", EridaniBatchRenderPayload, timeout=datetime.timedelta(seconds=600.0))

    write_csv(truthDataPath, true_data)
    write_csv(gravTruthDataPath, true_accelerations)
    write_csv(gravEstDataPath, measured_accelerations)
    write_csv(q_i_b_DataPath, q_i_b_data)
    write_csv(omega_i_b_DataPath, omega_i_b_data)

else:
    st.OnScreenLogMessage("Render directory not empty; skipping render step...\n", "SPSTraverse", st.Severity.Info)


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
    min_star_area = 3 # minimum pixel area for a star
    max_star_area = 200 # maximum pixel area for a star
    max_num_stars_to_process = 40 # maximum number of centroids to attempt to match per image

    low_thresh_pxl_intensity = None
    hi_thresh_pxl_intensity = None

    VERBOSE = False # set True for prints on results
    graphics = False # set True for graphics throughout the solve process

    data_path = os.path.join(thisRepo, 'data') # full path to your data
    cam_config_file_path = os.path.join(thisRepo, 'data', 'cam_config', 'Custom_cam.json') # full path (including filename) of your cam config file
    darkframe_file_path = os.path.join(thisRepo, 'Images', 'darkframes', 'darkframe.png') # full path (including filename) of your darkframe file
    image_extension = ".png" # the image extension to search for in the data_path directory
    cat_prefix ='' # if the catalog has a prefix, define it here

    #################################
    ####    Support Functions    ####
    #################################

    st.OnScreenLogMessage(f'imgSourceDir = {renderDir}', "SPSTraverse", st.Severity.Info)

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

    dir_contents = os.listdir(renderDir)
    for i in range(len(dir_contents)):
        dir_contents[i] = renderDir + "/" + dir_contents[i]
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
truthData = read_csv(truthDataPath)
attitudeEstData = read_csv(attitudeEstDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
gravTruthData = read_csv(gravTruthDataPath)
gravEstData = read_csv(gravEstDataPath)
q_i_b_list = read_csv(q_i_b_DataPath)
omega_i_b_list = read_csv(omega_i_b_DataPath)

# Very basic error handling if datasets are not the same length
if not (len(truthData) == len(attitudeEstData) == len(gravEstData)):
    st.OnScreenLogMessage(f'Warning: early exit due to dataset length mismatch; truthData length = {len(truthData)}, attitudeEstData length = {len(attitudeEstData)}, and gravEstData length = {len(gravEstData)}.', "SPSTraverse", st.Severity.Info)
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

eps: float = 1.0e-6
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
    T_unrotate: list[npt.NDArray] = []
    T_planet: list[npt.NDArray] = []
    q_planet: list[npt.NDArray] = []
    for i in range(alignmentCutoff):
        _T_planet = T_i_b_list[i]
        # _T_unrotate = st.math.Quat_to_DCM(np.array([0.5, -0.5, 0.5, 0.5])) @ _T_planet
        # _T_unrotate = _T_planet.T
        _T_unrotate = _T_planet
        # _T_unrotate = np.identity(3)
        T_unrotate.append(_T_unrotate)
        T_planet.append(_T_planet)

        _q_planet = st.math.DCM_to_Quat(_T_planet)
        if _q_planet[3] < 0.0:
            for qq in range(4):
                _q_planet[qq] = -_q_planet[qq]
        q_planet.append(_q_planet)

    quatEstimates = read_quats(attitudeEstDataPath)[:alignmentCutoff]
    omega_est, omega_hist = estimate_omega_EnrightForm(quatEstimates, alignmentTimeStep_s, T_unrotate)
    est = np.rad2deg(np.linalg.norm(omega_est))
    exp = np.rad2deg(np.linalg.norm(angvel_i_b_list[0])) 

    st.OnScreenLogMessage(f"Estimated omega (deg/s): {np.round(np.rad2deg(omega_est), 6)}", "SPSAngVelEstimate", st.Severity.Info)
    st.OnScreenLogMessage(f"Expected omega (deg/s): {np.round(np.rad2deg(angvel_i_b_list[0]), 6)}", "SPSAngVelEstimate", st.Severity.Info)
    st.OnScreenLogMessage(f"Estimated magnitude (deg/s): {est:.6f}", "SPSAngVelEstimate", st.Severity.Info)
    st.OnScreenLogMessage(f"Expected magnitude (deg/s): {exp:.6f}", "SPSAngVelEstimate", st.Severity.Info)

# exit(0)

#################################
####    SPS KALMAN FILTER    ####
#################################

# Position error logging
# position_errors: list[npt.NDArray] = []
# distanceErrors_m: list[float] = []
# distanceErrors_km: list[float] = []

# Tolerances and scale factors
tol: float = 10.0  # m
gradientWalkFactor: float = 1.0

# You can't handle the truth!
latTruth = copy.deepcopy(phi_pg_0)
lonTruth = copy.deepcopy(lon_pg_0)
altTruth = copy.deepcopy(h_ellp_0)

startTime = time.perf_counter()
elapsedSeconds: float = 0.0
printInterval: int = 100

# Mean and covariance initialization for Kalman filter
mx_0: npt.NDArray = copy.deepcopy(cameraPosPlanetFixed)
Pxx_0: npt.NDArray = np.diag(np.array([1.0, 1.0, 1.0])) * 3e2 ** 2
Pww: npt.NDArray = np.diag(np.array([1.0, 1.0, 1.0])) * 5e1 ** 2
Pvv: npt.NDArray = np.diag(np.array([1.0, 1.0, 1.0])) * 1e2 ** 2

Hx: npt.NDArray = np.identity(3)

mx_plus = copy.deepcopy(mx_0)
Pxx_plus = copy.deepcopy(Pxx_0)

z_history: list[npt.NDArray] = []
mx_history: list[npt.NDArray] = []
Pxx_history: list[npt.NDArray] = []

alpha_underweight = 3.0
gamma_underweight = 0.3
# T_g_c = np.identity(3)  # transformation from gravity to camera frame
T_g_c = T_alignment.T  # transformation from gravity to camera frame

estimatedPositions: list[npt.NDArray] = []

tStart = st.timestamp.from_datetime(t0.as_datetime() + datetime.timedelta(seconds=alignmentCutoff * alignmentTimeStep_s))
st.SimGlobals.SimClock.ResetTo(tStart)

for j in range(len(times[alignmentCutoff:endpoint])):

    ######################################
    ####    Measurement Processing    ####
    ######################################

    tRightNow: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    if j < alignmentCutoff:
        tRightNow += datetime.timedelta(seconds=alignmentTimeStep_s)
    else:
        tRightNow += datetime.timedelta(seconds=traverseTimeStep_s)
    st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tRightNow))
    
    doPrint: bool = j % printInterval == 0

    T_i_b: npt.NDArray = T_i_b_list[j + alignmentCutoff]
    # R_i_b: npt.NDArray = T_i_b_list[j + alignmentCutoff].T
    truth_j = truthData[j + alignmentCutoff]
    attitudeEst_j = attitudeEstData[j + alignmentCutoff]
    gravEst_j = gravEstData[j + alignmentCutoff]
    
    if attitudeEst_j[0] == 999 or attitudeEst_j[1] == 999 or attitudeEst_j[2] == 999 or attitudeEst_j[3] == 999:
        st.OnScreenLogMessage(f'Warning: skipped measurement at index {j} (invalid quaternion).', "SPSTraverse", st.Severity.Info)
        continue
    
    q_i_c = np.array([attitudeEst_j[1], attitudeEst_j[2], attitudeEst_j[3], attitudeEst_j[0]])
    T_i_c = st.math.Quat_to_DCM(normalize(q_i_c))

    g_sensorFrame = np.array([gravEst_j[0], gravEst_j[1], gravEst_j[2]])

    # Coarse estimates
    r_coarse_1 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, mu, 
                                                  Omega, planetData, planetFixedFrame, False, np.zeros(3), 0.0, 20)
    r_coarse_2 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, mu, 
                                                  Omega, planetData, planetFixedFrame, True, r_coarse_1, 0.0, 20)
    
    if doPrint:
        st.OnScreenLogMessage(f'Sample point {j}:', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage(f'r_expected = {positions[j]}', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage(f'r_coarse_1 = {r_coarse_1}', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage(f'r_coarse_2 = {r_coarse_2}', "SPSTraverse", st.Severity.Info)
    
    fineOutputs = st.ProcPlanet.SPS.FineEstimate(r_coarse_2, T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, 
                                                 Omega, planetData, planetFixedFrame, gradientWalkFactor, tol, 
                                                 doPrint, j + alignmentCutoff, 0.0, 20)

    q_c_b = st.math.DCM_to_Quat(T_i_b @ T_i_c.T)
    # st.OnScreenLogMessage(f'KF q_c_b = {q_c_b}', "SPSTraverse", st.Severity.Info)

    r_bestEstimate = fineOutputs.pos
    phi_pg = fineOutputs.phi_pg
    lon = fineOutputs.lon
    alt = fineOutputs.alt
    i = fineOutputs.iterations

    estimatedPositions.append(r_bestEstimate)
    
    if doPrint:
        st.OnScreenLogMessage(f'r_bestEstimate = {r_bestEstimate}', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage(f'Estimated lat = {round(phi_pg, 6)} deg', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage(f'Estimated lon = {round(lon, 6)} deg', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage(f'True lat = {round(latTruth, 6)} deg', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage(f'True lon = {round(lonTruth, 6)} deg\n', "SPSTraverse", st.Severity.Info)
    
    ##################################
    ####    Filter Propagation    ####
    ##################################

    # TODO: pretty sure we can't just add error here because that breaks the Kalman Filter?
    # mx_minus = mx_plus + traverseDirection * traverseStep_m + rng.normal(0.0, traverseFollowingError_m_1sigma, size=3)
    mx_minus = mx_plus + traverseDirection * traverseStep_m
    mx_minus, _ = st.ProcPlanet.SampleGround(planetData, mx_minus, radiusEquatorial, 0.0, 20)
    Pxx_minus = Pxx_plus + Pww
    
    ########################################
    ####    Cauchy Measurement Noise    ####
    ########################################

    # gk = np.zeros(3)
    # Gk = np.identity(3)

    # gamma = 100.0  # Allegedly the "standard deviation" but that's only for Gaussian
    # Wk = Hx @ Pxx_minus @ Hx.T
    # innovation = r_bestEstimate - Hx @ mx_minus

    # st.OnScreenLogMessage(f"innovation = {innovation}", "SPSCauchyFilter", st.Severity.Info)
    # st.OnScreenLogMessage(f"Wk = {np.diag(Wk)}", "SPSCauchyFilter", st.Severity.Info)

    # for idx in range(3):
    #     a = 0.5 * Wk[idx, idx]
    #     sqrt_a = np.sqrt(a)
    #     A = np.complex64(gamma / sqrt_a, innovation[idx] / sqrt_a)
    #     phi = np.exp(0.25 * A * A) * special.erfc(0.5 * A)
    #     gk_idx = 1.0 / (2.0 * sqrt_a) * np.imag(A * phi) / np.real(phi)
    #     Gk_idx = gk_idx * gk_idx + np.real((0.5 * A * A + 1.0) * phi - A / np.sqrt(np.pi)) / (2.0 * a * np.real(phi))

    #     gk[idx] = gk_idx if np.isfinite(gk_idx) else 0.0
    #     Gk[idx] = Gk_idx if np.isfinite(Gk_idx) else 1.0

    # st.OnScreenLogMessage(f"gk = {gk}", "SPSCauchyFilter", st.Severity.Info)
    # st.OnScreenLogMessage(f"Gk = {np.diag(Gk)}", "SPSCauchyFilter", st.Severity.Info)
    
    #############################
    ####    Filter Update    ####
    #############################

    # Measurement editing: only keep if less than 6-sigma from mean
    if (np.abs(r_bestEstimate - mx_minus) < 6.0 * np.sqrt(np.diag(Pxx_minus))).all():
        mz_minus = Hx @ mx_minus
        Pxz_minus = Pxx_minus @ Hx.T
        Pzz_minus = Hx @ Pxx_minus @ Hx.T + alpha_underweight * Pvv
        K = Pxz_minus @ np.linalg.inv(Pzz_minus)

        mx_plus = mx_minus + gamma_underweight * K @ (r_bestEstimate - mz_minus)
        Pxx_plus = Pxx_minus - Pxz_minus @ K.T - K @ Pxz_minus.T + K @ Pzz_minus @ K.T

        # mx_plus = mx_minus + Pxx_minus @ Hx.T @ gk
        # Pxx_plus = Pxx_minus - Pxx_minus @ Hx.T @ Gk @ Hx @ Pxx_minus
    else:
        st.OnScreenLogMessage(f"Measurement at index {j} not processed; exceed 6-sigma distance to mean.", "SPSTraverse", st.Severity.Info)
        mx_plus = copy.deepcopy(mx_minus)
        Pxx_plus = copy.deepcopy(Pxx_minus)

    mx_plus, _ = st.ProcPlanet.SampleGround(planetData, mx_plus, radiusEquatorial, 0.0, 20)

    z_history.append(r_bestEstimate)
    mx_history.append(mx_plus)
    Pxx_history.append(Pxx_plus)
    
    #####################################
    ####    Clean-up and Printing    ####
    #####################################

    percentComplete = round(100.0 * float(j) / float(len(times[alignmentCutoff:endpoint])), 3)
    
    endTime = time.perf_counter()
    elapsedSeconds = endTime - startTime
    elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

    if doPrint:
        st.OnScreenLogMessage(f'Elapsed time: {elapsedTime}', "SPSTraverse", st.Severity.Info)
        st.OnScreenLogMessage("------------------------------------------------------------------------------------------------------\n", "SPSTraverse", st.Severity.Info)

########################
####    PLOTTING    ####
########################

cameraTraversePositions: list[npt.NDArray] = positions[alignmentCutoff:endpoint]

fig1 = plt.figure(layout='constrained')
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)

z_x = np.array([z[0] for z in z_history])
z_y = np.array([z[1] for z in z_history])
z_z = np.array([z[2] for z in z_history])

mx_x = np.array([mx[0] for mx in mx_history])
mx_y = np.array([mx[1] for mx in mx_history])
mx_z = np.array([mx[2] for mx in mx_history])

Pxx_x = np.array([Pxx[0, 0] for Pxx in Pxx_history])
Pxx_y = np.array([Pxx[1, 1] for Pxx in Pxx_history])
Pxx_z = np.array([Pxx[2, 2] for Pxx in Pxx_history])

camPos_x = np.array([camPos[0] for camPos in cameraTraversePositions])
camPos_y = np.array([camPos[1] for camPos in cameraTraversePositions])
camPos_z = np.array([camPos[2] for camPos in cameraTraversePositions])

subsample: int = 1
ax1.scatter(times[alignmentCutoff:endpoint][::subsample], z_x[::subsample] - camPos_x[::subsample], label=r'$z(0)$', color='purple')
ax1.plot(times[alignmentCutoff:endpoint], mx_x - camPos_x, label=r"$m_{x}(0)$", color='blue')
ax1.plot(times[alignmentCutoff:endpoint], -3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r', label=r"$P_{xx}(0,0)$")
ax1.plot(times[alignmentCutoff:endpoint], 3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Position error (m)")
ax1.set_title(r"Error in $m_{x}(0)$ over Time")
ax1.grid()
ax1.legend()

ax2.scatter(times[alignmentCutoff:endpoint][::subsample], z_y[::subsample] - camPos_y[::subsample], label=r'$z(1)$', color='purple')
ax2.plot(times[alignmentCutoff:endpoint], mx_y - camPos_y, label=r"$m_{x}(1)$", color='blue')
ax2.plot(times[alignmentCutoff:endpoint], -3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r', label=r"$P_{xx}(1,1)$")
ax2.plot(times[alignmentCutoff:endpoint], 3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position error (m)")
ax2.set_title(r"Error in $m_{x}(1)$ over Time")
ax2.grid()
ax2.legend()

ax3.scatter(times[alignmentCutoff:endpoint][::subsample], z_z[::subsample] - camPos_z[::subsample], label=r'$z(2)$', color='purple')
ax3.plot(times[alignmentCutoff:endpoint], mx_z - camPos_z, label=r"$m_{x}(2)$", color='blue')
ax3.plot(times[alignmentCutoff:endpoint], -3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r', label=r"$P_{xx}(2,2)$")
ax3.plot(times[alignmentCutoff:endpoint], 3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r')
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Position error (m)")
ax3.set_title(r"Error in $m_{x}(2)$ over Time")
ax3.grid()
ax3.legend()

local_x = -cameraPosNWU.west()
local_y = cameraPosNWU.north()

trueTraverse_x = [np.dot(camPos - cameraPosPlanetFixed, local_x) for camPos in cameraTraversePositions]
trueTraverse_y = [np.dot(camPos - cameraPosPlanetFixed, local_y) for camPos in cameraTraversePositions]

estTraverse_x = [np.dot(mx - cameraPosPlanetFixed, local_x) for mx in mx_history]
estTraverse_y = [np.dot(mx - cameraPosPlanetFixed, local_y) for mx in mx_history]

positionEstimates_x = [np.dot(estPos - cameraPosPlanetFixed, local_x) for estPos in estimatedPositions]
positionEstimates_y = [np.dot(estPos - cameraPosPlanetFixed, local_y) for estPos in estimatedPositions]

fig2 = plt.figure(layout='constrained')
ax4 = fig2.add_subplot(111)

ax4.plot(trueTraverse_x, trueTraverse_y, color='green', label='True Traverse')
ax4.plot(estTraverse_x, estTraverse_y, color='blue', label='Estimated Traverse')
ax4.scatter(positionEstimates_x, positionEstimates_y, color='purple', marker='x', label='Position Estimates')
ax4.scatter(0.0, 0.0, color='red', marker='*', s=100, label=f'Origin ({phi_pg_0}°N, {lon_pg_0}°E)', zorder=2)

ax4.set_xlabel('East Position (m)')
ax4.set_ylabel('North Position (m)')
ax4.axis('scaled')
ax4.set_box_aspect(1)
ax4.grid()
ax4.legend()

plt.show()

st.leave_sim()
