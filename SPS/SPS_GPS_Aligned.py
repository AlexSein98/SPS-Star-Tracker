import os, sys, time, datetime, traceback
import spaceteams as st
import scipy.special as special
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
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

from SPS.grav_vesta_DAWN import grav_vesta_DAWN, grav_base

# import spiceypy as spice
import copy
import psutil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import hashlib

import numpy as np
import numpy.typing as npt
import shutil
import csv
import astropy.time as astrotime
import math
from decimal import Decimal

# from SPS.global_config import globalConfig
from py_src.star.python.transformations import latlon_to_T, T_to_latlon, r_to_latlonalt, r_hat_to_ra_dec, normalize


class GravSampler:
    def __init__(self, gravModel: grav_base,  maxDegree: int, maxOrder: int):
        self.maxDegree = maxDegree
        self.maxOrder = maxOrder
        self.Cilm = np.dstack((gravModel.Clm[:maxDegree+1, :maxOrder+1], gravModel.Slm[:maxDegree+1, :maxOrder+1])).transpose((2, 0, 1))
        self.Cilm[0, 0, 0] = 1.0  # add spherical component of gravity
        self.Cilm[0, 2, 0] = 0.0  # take out J2 (TODO: don't?)
        self.gravModel = gravModel

    def NormalizationCoefficient(self, l: int, m: int):
        k: int = 2
        if m == 0:
            k = 1
        return np.sqrt(float(Decimal(math.factorial(l + m)) / Decimal((math.factorial(l - m) * k * ((2 * l) + 1)))))

    def GetDimensionalZonalHarmonic(self, l, m, Clm):
        PI_lm = self.NormalizationCoefficient(l, m)
        return Clm[l][m] / PI_lm

    def HarmonicAcceleration(self, r: npt.NDArray, gravModel: grav_base, 
                             lat: float, lon: float, degree: int, order: int) -> npt.NDArray:
        # initialize P matrix:
        size = degree + 1
        P = np.zeros((size, size))

        lat_r = np.deg2rad(lat)
        lon_r = np.deg2rad(lon)

        R = gravModel.radius
        mu = gravModel.mu

        r_norm = np.linalg.norm(r)
        r_norm_inv = 1.0 / r_norm
        r_norm_inv_2 = r_norm_inv * r_norm_inv
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
            R_r_l = (R * r_norm_inv) ** l
            l_1 = l + 1.0
            for m in range(0, min(l + 1, order + 1)):
                PI_lm = self.NormalizationCoefficient(l, m)
                Clm = gravModel.Clm[l][m] / PI_lm
                Slm = gravModel.Slm[l][m] / PI_lm

                Plm1 = 0.0 ###
                if m < l:
                    Plm1 = P[l][m + 1]

                dUdr += R_r_l * l_1 * P[l][m] * (Clm * np.cos(m * lon_r) + Slm * np.sin(m * lon_r))
                dUdLat += R_r_l * (Plm1 - m * tanLat * P[l][m]) * (Clm * np.cos(m * lon_r) + Slm * np.sin(m * lon_r))
                dUdLon += R_r_l * m * P[l][m] * (Slm * np.cos(m * lon_r) - Clm * np.sin(m * lon_r))

        dUdr *= -mu * r_norm_inv_2
        dUdLat *= mu * r_norm_inv
        dUdLon *= mu * r_norm_inv

        r_squared = r_norm ** 2
        rho_squared = r[0] ** 2 + r[1] ** 2
        rho = np.sqrt(rho_squared)
        a_x = (dUdr * r_norm_inv - r[2] * dUdLat / (r_squared * rho)) * r[0] - (dUdLon / rho_squared) * r[1]
        a_y = (dUdr * r_norm_inv - r[2] * dUdLat / (r_squared * rho)) * r[1] + (dUdLon / rho_squared) * r[0]
        a_z = (dUdr * r_norm_inv) * r[2] + (rho * dUdLat * r_norm_inv_2)

        return np.array([a_x, a_y, a_z])


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


os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
np.seterr(all='raise')

##################################
####    PLANET DATA IMPORT    ####
##################################

planetData = st.ProcPlanet.DataStore()

# moonGlobalData = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014")
vestaGlobalData = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Vesta/Global/Vesta_Dawn_HAMO_DTM_DLR_Global_48ppd_Altitude")
args = st.ProcPlanet.GeoBin_Extra_Args()
args.cubicInterp = True

planetData.AddGeoBinAltimetryLayer(1.0, vestaGlobalData, args)

# time.sleep(5.0)


# Wait for Eridani to load (TODO: probably don't need this because 
# it's guaranteed to only start after "init" is done on all systems?)

# st.OnScreenLogMessage("Waiting for Eridani to load...", "SPSGPS", st.Severity.Info)
# st.GetThisSystem().AddOrSetParam(st.VarType.bool, "Ready", False)
# def SetIsReady(paramMap: st.ParamMap, timeNow: st.timestamp):
#     st.GetThisSystem().SetParam(st.VarType.bool, "Ready", True)
# st.SimGlobals.Subscribe("EridaniLoadingComplete", SetIsReady)
# while not st.GetThisSystem().GetParam(st.VarType.bool, "Ready"):
#     pass

# st.OnScreenLogMessage("Got past the initial wait time!", "SPSGPS", st.Severity.Info)

#####################################
####    CLASSES AND FUNCTIONS    ####
#####################################

def rad_to_arcsec(rad: float) -> float:
    return 3600.0 * np.rad2deg(rad)


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
    N = len(quats) - 1
    omega_sum = np.zeros(3)
    omega_history: list[npt.NDArray] = []

    for i in range(1, N-1):
        q_prev = quats[i - 1]
        q_curr = quats[i]
        q_next = quats[i + 1]

        delta_q = normalize(quat_mult(q_next, quat_inverse(q_prev)))
        angle: float = 2.0 * np.arccos(np.clip(delta_q[3], -1.0, 1.0))
        axis: npt.NDArray = np.zeros(3)
        if angle != 0.0:
            axis: npt.NDArray = delta_q[0:3] / np.sin(0.5 * angle)

        omega_i = T_unrotate[i] @ (-angle * axis / (2.0 * deltaT))

        # omega_i = angle * axis / deltaT
        omega_history.append(omega_i)
        omega_sum += omega_i

    return omega_sum / float(len(omega_history)), omega_history


def is_dir_empty(path):
    # Returns True if empty, False otherwise
    return not any(Path(path).iterdir())


np.set_printoptions(suppress=True)

#################################
####    GLOBAL PARAMETERS    ####
#################################

delete_old: bool                = False
reprocess_star_tracker: bool    = False
globalDoPrint: bool             = True
incrementalRendering: bool      = False

# Warning: setting Incremental Rendering to "True" this carries the assumption that the only change between 
# runs was increasing the number of images. All previously-rendered images will be kept as is, carrying with
# them all baked-in assumptions about the times and locations at which they were captured. Unfortunately, it
# is necessary to reprocess all star tracker measurements in this case.
if incrementalRendering:
    delete_old = False
    reprocess_star_tracker = True

numImages: int = 400

##############################
####    INITIALIZATION    ####
##############################

# site = "Vesta"
site = "VestaLongTerm"

hash_object = hashlib.sha256(site.encode('utf-8'))
int_seed = int(hash_object.hexdigest(), 16)

#################################
####    PLANET PARAMETERS    ####
#################################

J2000: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "J2000Frame")
J2000Frame = J2000.GetBodyFixedFrame()
planetEntity: st.Entity = st.GetThisSystem().GetParam(st.VarType.entityRef, "Planet")
planetFixedFrame = planetEntity.GetBodyFixedFrame()
# planetName: str = planetEntity.getName()
planetName = site

cameraEntities: list[st.Entity] = st.GetThisSystem().GetParamArray(st.VarType.entityRef, "SPSCameras")
numSPS = len(cameraEntities)
orbiter: st.Entity = st.GetThisSystem().GetParam(st.VarType.entityRef, "Orbiter")

outputDir = os.path.join(localOutput, planetName)
renderDirsLocal: list[str] = []
renderDirs: list[str] = []
for i in range(numSPS):
    renderDirsLocal.append(os.path.join("Local", "..", "Output", planetName, "Renders", str(i).zfill(3)))
    renderDirs.append(os.path.join(outputDir, "Renders", str(i).zfill(3)))

truthDataPaths: list[str] = []
attitudeEstDataPaths: list[str] = []
gravTruthDataPaths: list[str] = []
gravEstDataPaths: list[str] = []
q_i_b_DataPaths: list[str] = []
omega_i_b_DataPaths: list[str] = []

for i in range(numSPS):
    truthDataPaths.append(os.path.join(outputDir, "truth_data_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    attitudeEstDataPaths.append(os.path.join(outputDir, "attitudes_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    gravTruthDataPaths.append(os.path.join(outputDir, "true_gravities_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    gravEstDataPaths.append(os.path.join(outputDir, "measurements_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    q_i_b_DataPaths.append(os.path.join(outputDir, "q_i_b_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    omega_i_b_DataPaths.append(os.path.join(outputDir, "omega_i_b_" + planetName + "_" + str(i).zfill(3) + ".csv"))

#############################
####    ERROR SOURCES    ####
#############################

# Random number generator
rng = np.random.default_rng(int_seed + 100)

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
    for i in range(numSPS):
        if os.path.exists(renderDirs[i]):
            shutil.rmtree(renderDirs[i])
        Path(renderDirs[i]).mkdir(parents=True)

# Time
t0: st.timestamp = st.timestamp.from_datetime(datetime.datetime(year=2026, month=5, day=22, hour=16))
st.SimGlobals.SimClock.ResetTo(t0)

st.OnScreenLogMessage(f"t0 = {t0.as_utc_string()}", "SPSGPS", st.Severity.Info)

time.sleep(0.1)

timeStep_s: float = 900.0
propagationTimeStep_s = 60.0

times = timeStep_s * np.linspace(0.0, numImages - 1, numImages)
times_days: npt.NDArray = times / 86400.0

# mu: float = 1e9 * planetEntity.GetParam(st.VarType.double, ["Dynamics", "GravitationalParameter_km3_s2"])
radiusEquatorial: float = planetEntity.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])
Omega = planetEntity.getAngVelocity().WRT(J2000Frame).ExprIn(planetFixedFrame)

st.OnScreenLogMessage(f"Planet angular velocity (Omega) = {np.rad2deg(Omega)} deg/s", "SPSGPS", st.Severity.Info)
# Omega = np.array([0.0, 0.0, 2.66e-6])  # Expressed in the planet-fixed frame

startTime = time.perf_counter()
elapsedSeconds: float = 0.0
printInterval: int = 10

cameraTruePositions: list[npt.NDArray] = []
for i in range(numSPS):
    cameraPosPlanetFixed = cameraEntities[i].GetParam(st.VarType.doubleV3, "Location")
    cameraPosPlanetFixed, _ = st.ProcPlanet.SampleGround(planetData, cameraPosPlanetFixed, radiusEquatorial, 0.0, 20)
    cameraTruePositions.append(cameraPosPlanetFixed)

    llaResult = st.PlanetUtils.PCPF_to_LLA(cameraPosPlanetFixed, radiusEquatorial)
    lat_pc = np.rad2deg(llaResult.lat())
    lon_pc = np.rad2deg(llaResult.lon())
    h_pc = llaResult.alt()

#############################
####    RENDER IMAGES    ####
#############################

for ee in range(numSPS):
    if not os.path.exists(renderDirs[ee]):
        Path(renderDirs[ee]).mkdir(parents=True)

    if is_dir_empty(renderDirs[ee]) or incrementalRendering:
        # locs: list[npt.NDArray] = []
        # vels: list[npt.NDArray] = []
        # rots: list[npt.NDArray] = []
        # names: list[str] = []

        true_data: list[npt.NDArray] = []
        true_accelerations: list[npt.NDArray] = []
        measured_accelerations: list[npt.NDArray] = []
        q_i_b_data: list[npt.NDArray] = []
        omega_i_b_data: list[npt.NDArray] = []
        
        planetStateDummy = st.frames.FramedLocVelAcc(st.frames.rva_struct(np.zeros(3), np.zeros(3), np.zeros(3)), planetFixedFrame)
        tNow: datetime.datetime = t0.as_datetime()
        st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tNow))
        for i in range(numImages):
            renderName = "SPSRender" + str(i).zfill(5)
            doPrint: bool = i % printInterval == 0 and globalDoPrint

            # Sample gravity vector
            stateNow = st.frames.FramedLocVelAcc(st.frames.rva_struct(cameraTruePositions[ee], np.zeros(3), np.zeros(3)), planetFixedFrame)
            g = st.SimGlobals.SampleVectorField("Gravity", stateNow).ExprIn(planetFixedFrame)

            g_thirdBody = st.SimGlobals.SampleVectorField("ThirdBodyGravity", stateNow).ExprIn(planetFixedFrame)
            g_thirdBody -= st.SimGlobals.SampleVectorField("ThirdBodyGravity", planetStateDummy).ExprIn(planetFixedFrame)
            # st.OnScreenLogMessage(f"True third body gravity = {g_thirdBody}", "SPSGPS", st.Severity.Info)

            g += g_thirdBody - np.cross(Omega, np.cross(Omega, cameraTruePositions[ee]))
            g_true = copy.deepcopy(g)
            g_true_framed = st.frames.FramedVector(g_true, planetFixedFrame)
            
            lat_pc, lon_pc, h_pc = r_to_latlonalt(cameraTruePositions[ee], radiusEquatorial)
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
            true_accelerations.append(g_true)

            g_measured_planetFixed = (T_P_G.T @ np.array([g_IMU_frame]).T).T[0]
            g_measured_framed = st.frames.FramedVector(g_measured_planetFixed, planetFixedFrame)
            gInertial = g_measured_framed.ExprIn(J2000Frame)

            ra_true_i, de_true_i = r_hat_to_ra_dec(-normalize(gInertial_true))
            ra, de = r_hat_to_ra_dec(-normalize(gInertial))
            ra_pcpf, de_pcpf = r_hat_to_ra_dec(-normalize(g_true))
            ra_meas_pcpf, de_meas_pcpf = r_hat_to_ra_dec(-normalize(g_measured_planetFixed))

            # Passive transform from inertial frame to surface (grav vector) frame (x axis along -grav)
            # We get this by converting a pointing vec in inertial frame to ra/dec, then converting
            # ra/dec to rotation matrix (Euler 321 with negative dec, 0 roll), then transposing.
            T_I_S_true = ra_dec_to_rot(ra_true_i, de_true_i).T
            q_I_S_true_passive = st.math.DCM_to_Quat(T_I_S_true)
            true_data.append(q_I_S_true_passive)

            T_PCPF_S_true = ra_dec_to_rot(ra_pcpf, de_pcpf).T
            q_PCPF_S_true_passive = st.math.DCM_to_Quat(T_PCPF_S_true)

            # Render (old version)
            pos_framed = st.frames.FramedLoc(cameraTruePositions[ee], planetFixedFrame)
            vel_framed = st.frames.FramedLocVel(st.frames.rv_struct(cameraTruePositions[ee], np.zeros(3)), planetFixedFrame)

            cameraEntities[ee].setRotation(st.frames.FramedRot(T_I_S_true, J2000Frame))

            if os.path.isfile(os.path.join(renderDirs[ee], renderName + ".png")):
                if doPrint:
                    st.OnScreenLogMessage(f"Incremental Rendering: skipped existing image {renderName}", "SPSGPS", st.Severity.Info)

                # Step time forward by the correct dt
                tNow += datetime.timedelta(seconds=timeStep_s)
                
                st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tNow))
                time.sleep(0.1)

                continue

            EridaniRenderPayload = st.ParamMap()
            EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Loc", pos_framed.WRT_ExprIn(J2000Frame))
            EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Vel", vel_framed.vel_WRT_ExprIn(J2000Frame))
            EridaniRenderPayload.AddParam(st.VarType.doubleV4, "Rot", q_I_S_true_passive)
            EridaniRenderPayload.AddParam(st.VarType.entityRef, "Frame", J2000)
            EridaniRenderPayload.AddParam(st.VarType.string, "NameOverride", renderName)
            EridaniRenderPayload.AddParam(st.VarType.string, "OutputPathOverride", str(renderDirsLocal[ee]))
            payload = st.SimGlobals.Request("EridaniSingleRender", EridaniRenderPayload, timeout=datetime.timedelta(seconds=60.0))

            if doPrint:
                st.OnScreenLogMessage(f'Rendering image {i} of {numImages} ({round(100.0 * float(i) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}', "SPSGPS", st.Severity.Info)

            endTime = time.perf_counter()
            elapsedSeconds = endTime - startTime
            elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

            if doPrint:
                projectedRemainingSeconds: float = elapsedSeconds * float(numImages - i + 1) / float(i + 1)
                projectedRemainingTime = datetime.timedelta(seconds=round(projectedRemainingSeconds))
                st.OnScreenLogMessage(f'Elapsed time: {elapsedTime}. Remaining time estimate: {projectedRemainingTime}\n', "SPSGPS", st.Severity.Info)

            # Step time forward by the correct dt
            tNow += datetime.timedelta(seconds=timeStep_s)
            
            st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tNow))
            time.sleep(0.1)

        write_csv(truthDataPaths[ee], true_data)
        write_csv(gravTruthDataPaths[ee], true_accelerations)
        write_csv(gravEstDataPaths[ee], measured_accelerations)
        write_csv(q_i_b_DataPaths[ee], q_i_b_data)
        write_csv(omega_i_b_DataPaths[ee], omega_i_b_data)

    else:
        st.OnScreenLogMessage(f"Render directory {ee} not empty; skipping render step...\n", "SPSGPS", st.Severity.Info)

#########################################
####    OBTAIN ATTITUDE ESTIMATES    ####
#########################################

for ee in range(numSPS):
    if reprocess_star_tracker:
        Path(attitudeEstDataPaths[ee]).unlink(missing_ok=True)

    if not Path(attitudeEstDataPaths[ee]).is_file():
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

        st.OnScreenLogMessage(f'imgSourceDir = {renderDirs[ee]}', "SPSGPS", st.Severity.Info)

        ###################################
        ####    Process Star Images    ####
        ###################################

        # Load star tracker and catalog data
        if darkframe_file_path == '': darkframe_file_path = None
        if darkframe_file_path is not None:
            if not os.path.exists(darkframe_file_path):
                darkframe_file_path = None
                st.OnScreenLogMessage("unable to find provided darkframe file, proceeding without one...", "SPSGPS", st.Severity.Info)
            else:    st.OnScreenLogMessage("darkframe file: " + darkframe_file_path, "SPSGPS", st.Severity.Info)
        else:    st.OnScreenLogMessage("no darkframe file provided, proceeding without one...", "SPSGPS", st.Severity.Info)

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

        dir_contents = os.listdir(renderDirs[ee])
        for i in range(len(dir_contents)):
            dir_contents[i] = renderDirs[ee] + "/" + dir_contents[i]
        dir_contents.sort()

        image_names = []

        for item in dir_contents:
            if image_extension in item:
                image_names+=[os.path.abspath(item)]

        idx: int = 0
        for image_filename in image_names:
            image_name += [image_filename]
            # st.OnScreenLogMessage("===================================================", "SPSGPS", st.Severity.Info)
            # st.OnScreenLogMessage(image_filename, "SPSGPS", st.Severity.Info)

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
                    st.OnScreenLogMessage('est q: ' + str(q_est)+'\n', "SPSGPS", st.Severity.Info)
                q_rotate = np.array([0.5, -0.5, 0.5, 0.5])  # w-last quaternion
                q_est = quat_mult(q_est, q_rotate)  # w-last quaternion
                qs += [q_est[3]]
                qv0 += [-q_est[0]]
                qv1 += [-q_est[1]]
                qv2 += [-q_est[2]]
            except AssertionError:
                if VERBOSE:
                    st.OnScreenLogMessage('NO VALID STARS FOUND\n', "SPSGPS", st.Severity.Info)
                qs += [999]
                qv0 += [999]
                qv1 += [999]
                qv2 += [999]

            ttime += [time.time()]
            sram  += [psutil.virtual_memory().percent]
            #scpu  += [psutil.cpu_percent(2)]
            scpu  += [psutil.cpu_percent()]

            st.OnScreenLogMessage(f'Completed image {idx} ({round(float(idx) / float(len(image_names)) * 100.0, 2)} %)', "SPSGPS", st.Severity.Info)
            idx += 1

        data = {'image name':image_name,'time':ttime,'RAM':sram,'CPU':scpu,'image solve time (s)':solve_time, 'qs':qs,'qv0':qv0,'qv1':qv1,'qv2':qv2}

        now = str(datetime.datetime.now())
        now = now.split('.')
        now = now[0]
        now = now.replace(' ','_')
        now = now.replace(':','-')

        #write stuff
        keys=sorted(data.keys())

        with open(attitudeEstDataPaths[ee],'w', newline='') as csv_file:
            writer=csv.writer(csv_file)
            writer.writerow(keys)
            writer.writerows(zip(*[data[key] for  key in keys]))

        st.OnScreenLogMessage("\n\n took " + str(time.time()-total_start) + " seconds to complete \n\n", "SPSGPS", st.Severity.Info)
        st.OnScreenLogMessage("data saved to: " + attitudeEstDataPaths[ee], "SPSGPS", st.Severity.Info)

    else:
        st.OnScreenLogMessage("Quaternion measurements already processed; skipping processing step...\n", "SPSGPS", st.Severity.Info)

# Get data from files
truthData = []
attitudeEstData = []
gravTruthData = []
gravEstData = []
q_i_b_list = []
omega_i_b_list = []

T_i_b_list: list[list[npt.NDArray]]         = []
angvel_i_b_list: list[list[npt.NDArray]]    = []
T_i_c_list: list[list[npt.NDArray]]         = []
g_est_list: list[list[npt.NDArray]]         = []
omega_est_list: list[list[npt.NDArray]]     = []

for ee in range(numSPS):
    truthData.append(read_csv(truthDataPaths[ee]))
    attitudeEstData.append(read_csv(attitudeEstDataPaths[ee], ignore=[0, 1, 2, 3], hasHeader=True))
    gravTruthData.append(read_csv(gravTruthDataPaths[ee]))
    gravEstData.append(read_csv(gravEstDataPaths[ee]))
    q_i_b_list.append(read_csv(q_i_b_DataPaths[ee]))
    omega_i_b_list.append(read_csv(omega_i_b_DataPaths[ee]))

    # Very basic error handling if datasets are not the same length
    if not (len(truthData) == len(attitudeEstData) == len(gravEstData)):
        st.OnScreenLogMessage(f'Warning: early exit due to dataset length mismatch; truthData length = {len(truthData)}, attitudeEstData length = {len(attitudeEstData)}, and gravEstData length = {len(gravEstData)}.', "SPSGPS", st.Severity.Info)
        exit(0)

    # Initialize all inertial-to-planet attitude matrices
    T_i_b_list.append([])
    angvel_i_b_list.append([])
    T_i_c_list.append([])
    g_est_list.append([])
    for i in range(len(times)):
        T_i_b_list[ee].append(st.math.Quat_to_DCM(normalize(q_i_b_list[ee][i])))
        angvel_i_b_list[ee].append(np.array([omega_i_b_list[ee][i][0], omega_i_b_list[ee][i][1], omega_i_b_list[ee][i][2]]))
        q_i_c = np.array([attitudeEstData[ee][i][1], attitudeEstData[ee][i][2], attitudeEstData[ee][i][3], attitudeEstData[ee][i][0]])
        T_i_c_list[ee].append(st.math.Quat_to_DCM(normalize(q_i_c)))
        g_est_list[ee].append(gravEstData[ee][i])

    T_unrotate: list[npt.NDArray] = []
    T_planet: list[npt.NDArray] = []
    q_planet: list[npt.NDArray] = []

    quatEstimates = read_quats(attitudeEstDataPaths[ee])
    for i in range(len(times)):
        _T_planet = T_i_b_list[ee][i]
        _T_i_c = st.math.Quat_to_DCM(normalize(quatEstimates[i]))
        _T_unrotate = _T_planet @ _T_i_c.T
        T_unrotate.append(_T_unrotate)
        T_planet.append(_T_planet)

        _q_planet = st.math.DCM_to_Quat(_T_planet)
        if _q_planet[3] < 0.0:
            for qq in range(4):
                _q_planet[qq] = -_q_planet[qq]
        q_planet.append(_q_planet)

    omega_est, omega_hist = estimate_omega_EnrightForm(quatEstimates, timeStep_s, T_unrotate)
    omega_est_list.append(omega_hist)

#################################
####    SPS KALMAN FILTER    ####
#################################

st.SimGlobals.SimClock.ResetTo(t0)

# Tolerances and scale factors
tol: float = 10.0  # m
gradientWalkFactor: float = 1.0

startTime = time.perf_counter()
elapsedSeconds: float = 0.0
printInterval: int = 10

sc_pos = orbiter.GetParam(st.VarType.doubleV3, "Location")
sc_vel = orbiter.GetParam(st.VarType.doubleV3, "Velocity") - np.cross(Omega, sc_pos)

st.OnScreenLogMessage(f"sc_pos = {sc_pos}", "SPSGPS", st.Severity.Info)
st.OnScreenLogMessage(f"sc_vel = {sc_vel}", "SPSGPS", st.Severity.Info)

# Setting up the two gravity models
estDegree: int = 4  # = estOrder, keep them the same for simplicity
gravModel_true: grav_base = grav_vesta_DAWN()
gravModel_est: grav_base = grav_vesta_DAWN()
gravModel_est.Clm = np.zeros((estDegree + 1, estDegree + 1))
gravModel_est.Slm = np.zeros((estDegree + 1, estDegree + 1))

trueGravSampler = GravSampler(gravModel_true, 20, 20)
customGravSampler = GravSampler(gravModel_est, estDegree, estDegree)

mu_true = gravModel_true.mu
Clm_true = gravModel_true.Clm
Slm_true = gravModel_true.Slm

#############################################
####    Run the filter in km and km/s    ####
#############################################

# Mean initialization for Kalman filter
#################################################################################################
# Example for 4 SPS devices and a 4x4 estimated spherical harmonic gravity model
# mx_0 = [r0(3), v0(3),             Indexing: 0:6       Inclusive indices: [0,  5 ]
#         rSPS1(3), bSPS1(3),       Indexing: 6:12      Inclusive indices: [6,  11]
#         rSPS2(3), bSPS2(3),       Indexing: 12:18     Inclusive indices: [12, 17]
#         rSPS3(3), bSPS3(3),       Indexing: 18:24     Inclusive indices: [18, 23]
#         rSPS4(3), bSPS4(3),       Indexing: 24:30     Inclusive indices: [24, 29]
#         C(2,0), ..., C(4,4),      Indexing: 30:42     Inclusive indices: [30, 41]
#         S(2,1), ..., S(4,4),      Indexing: 42:51     Inclusive indices: [42, 50]
#         mu]                       Indexing: 51        Inclusive indices: [51    ]
#################################################################################################

# Number of states:
numClm: int = int(0.5 * (estDegree - 1) * (estDegree + 4))
numSlm: int = int(0.5 * (estDegree - 1) * (estDegree + 4) - (estDegree - 1))
numSphericalHarmonicStates: int = numClm + numSlm
numStates: int = 6 + 1 + numSphericalHarmonicStates + 6
numMeasurements: int = 7 * numSPS
st.OnScreenLogMessage(f"Number of states       = {numStates}", "SPSGPS_KF_Initialization", st.Severity.Info)
st.OnScreenLogMessage(f"Number of measurements = {numMeasurements}", "SPSGPS_KF_Initialization", st.Severity.Info)

# Indices:
idx_rsc_start: int          = 0
idx_vsc_start: int          = 3
idx_mu: int                 = 6
idx_Clm_start: int          = 7
idx_Slm_start: int          = idx_Clm_start + numClm
idx_sigma_start: int        = idx_Slm_start + numSlm
idx_omega_start: int        = idx_sigma_start + 3

# Error 1-sigma magnitudes
err_range_km: float         = 0.01
err_rangeRate_km_s: float   = 0.0005
err_angle_rad: float        = 0.0001

# Spacecraft position and velocity
mx_0: npt.NDArray = np.concat((1e-3 * sc_pos, 1e-3 * sc_vel))

# Gravitational parameters (including GM at the start)
mx_0 = np.concat((mx_0, np.zeros(1), np.zeros(numSphericalHarmonicStates)))

# Attitude and angular velocity
mx_0 = np.concat((mx_0, st.math.DCM_to_MRP(T_i_b_list[0][0]).ToVector(), angvel_i_b_list[0][0]))

# Set initial GM estimate
mx_0[idx_mu] = 1e-9 * gravModel_true.mu

# Set C[2,0] term to correct value to mitigate filter periodicity
mx_0[idx_Clm_start] = gravModel_true.Clm[2, 0]

st.OnScreenLogMessage(f'mx_0 = {mx_0}', "SPSGPS", st.Severity.Info)

# Covariance initialization for Kalman filter
Pxx_0: npt.NDArray = np.zeros((numStates, numStates))

# Spacecraft position covariance
Pxx_0[idx_rsc_start:idx_rsc_start + 3, 
      idx_rsc_start:idx_rsc_start + 3] = np.identity(3) * 0.2 ** 2  # km

# Spacecraft velocity covariance
Pxx_0[idx_vsc_start:idx_vsc_start + 3, 
      idx_vsc_start:idx_vsc_start + 3] = np.identity(3) * 0.002 ** 2  # km/s

# Spherical harmonic coefficient covariance
for sh in range(numSphericalHarmonicStates):
    Pxx_0[idx_Clm_start + sh, 
          idx_Clm_start + sh] = 0.002 ** 2

# GM covariance
Pxx_0[idx_mu, idx_mu] = 0.01 ** 2

# Override C[2,0] covariance
Pxx_0[idx_Clm_start, 
      idx_Clm_start] = 0.0002 ** 2

# Attitude and angular velocity covariances
Pxx_0[idx_sigma_start:idx_sigma_start + 3, 
      idx_sigma_start:idx_sigma_start + 3] = np.identity(3) * 1e-1 ** 2
Pxx_0[idx_omega_start:idx_omega_start + 3,
      idx_omega_start:idx_omega_start + 3] = np.identity(3) * 1e-5 ** 2

Pww: npt.NDArray = Pxx_0 * 0.1 ** 2
Pww[idx_mu, idx_mu] = 0.000001 ** 2
for sh in range(numSphericalHarmonicStates):
    Pww[idx_Clm_start + sh, 
        idx_Clm_start + sh] = 0.00002 ** 2

Pww[idx_sigma_start:idx_sigma_start + 3, 
    idx_sigma_start:idx_sigma_start + 3] = np.identity(3) * 1e-4 ** 2

Pvv: npt.NDArray = np.zeros((numMeasurements, numMeasurements))
for ee in range(numSPS):
    posIndex: int = 7 * ee
    Pvv[posIndex, posIndex] = (1.0 * err_range_km) ** 2
    Pvv[posIndex + 1, posIndex + 1] = (1.0 * err_rangeRate_km_s) ** 2
    Pvv[posIndex + 2, posIndex + 2] = (1.0 * err_angle_rad) ** 2
    Pvv[posIndex + 3, posIndex + 3] = (1.0 * err_angle_rad) ** 2
    Pvv[posIndex + 4:posIndex + 7, posIndex + 4:posIndex + 7] = np.identity(3) * 4e-5 ** 2

# Add a bit of scatter to initial mean based on Pww
# st.OnScreenLogMessage(f"mx_0 before Pww = {mx_0}", "SPSGPS", st.Severity.Info)
mx_0 += np.linalg.cholesky(Pww) @ rng.normal(0.0, 1.0, numStates)
# st.OnScreenLogMessage(f"mx_0 after Pww  = {mx_0}", "SPSGPS", st.Severity.Info)

indices_lowerTriangular_Clm = np.tril_indices(estDegree + 1)
indices_lowerTriangular_Slm = np.tril_indices(estDegree)


def SetCoeffsInGravityModel(x: npt.NDArray):
    mu_new = 1e9 * x[idx_mu]
    Clm_linear = np.concat((np.zeros(3), x[idx_Clm_start:idx_Slm_start]))
    Slm_linear_noFirstColumn = np.concat((np.zeros(1), x[idx_Slm_start:idx_mu]))

    Clm_new = np.zeros((estDegree + 1, estDegree + 1))
    Slm_new = np.zeros((estDegree, estDegree))

    Clm_new[indices_lowerTriangular_Clm] = Clm_linear
    Slm_new[indices_lowerTriangular_Slm] = Slm_linear_noFirstColumn

    gravModel_est.mu = mu_new
    gravModel_est.Clm = Clm_new
    gravModel_est.Slm[1:estDegree + 1, 1:estDegree + 1] = Slm_new


def HarmonicGravity(pos_pcpf: npt.NDArray):
    llaResult = st.PlanetUtils.PCPF_to_LLA(pos_pcpf, radiusEquatorial)
    lat_pc = np.rad2deg(llaResult.lat())
    lon_pc = np.rad2deg(llaResult.lon())

    g_pcpf = customGravSampler.HarmonicAcceleration(pos_pcpf, gravModel_est, lat_pc, lon_pc, estDegree, estDegree)
    g_pcpf -= gravModel_est.mu * pos_pcpf / (np.linalg.norm(pos_pcpf) ** 3)
    return g_pcpf


def HarmonicGravity_True(pos_pcpf: npt.NDArray):
    llaResult = st.PlanetUtils.PCPF_to_LLA(pos_pcpf, radiusEquatorial)
    lat_pc = np.rad2deg(llaResult.lat())
    lon_pc = np.rad2deg(llaResult.lon())

    g_pcpf = customGravSampler.HarmonicAcceleration(pos_pcpf, gravModel_true, lat_pc, lon_pc, 20, 20)
    g_pcpf -= gravModel_true.mu * pos_pcpf / (np.linalg.norm(pos_pcpf) ** 3)
    return g_pcpf


def HarmonicGravityForMeasurements(state: st.frames.FramedLocVelAcc):
    return st.frames.FramedVector(HarmonicGravity(state.loc_WRT_ExprIn(planetFixedFrame)), planetFixedFrame)


# TODO: Not zero third-body gravity
def ThirdBodyGravityForMeasurements(state: st.frames.FramedLocVelAcc):
    return st.frames.FramedVector(np.zeros(3), planetFixedFrame)


def Skew(vec: npt.NDArray):
    return np.array([[0.0, -vec[2], vec[1]],
                    [vec[2], 0.0, -vec[0]],
                    [-vec[1], vec[0], 0.0]])


def HandleShadowParams(mx: npt.NDArray) -> npt.NDArray:
    if np.linalg.norm(mx[idx_sigma_start:idx_sigma_start + 3]) >= 1.0:
        mx[idx_sigma_start:idx_sigma_start + 3] = st.math.ModifiedRodriguesParameters(
            mx[idx_sigma_start:idx_sigma_start + 3]).shadow().ToVector()
        return mx
    else:
        return mx


def MRPDynamics(mrp: npt.NDArray, angVel: npt.NDArray) -> npt.NDArray:
    return 0.5 * (0.5 * (1.0 - np.dot(mrp, mrp)) * np.identity(3) - Skew(mrp) + np.outer(mrp, mrp)) @ angVel


def EquationsOfMotion(t: float, x: npt.NDArray):
    r_sc_km = x[idx_rsc_start:idx_rsc_start + 3]
    v_sc_km_s = x[idx_vsc_start:idx_vsc_start + 3]
    sigma_planet = x[idx_sigma_start:idx_sigma_start + 3]
    omega_planet = x[idx_omega_start:idx_omega_start + 3]

    x_dot = np.zeros(len(x))
    x_dot[idx_rsc_start:idx_rsc_start + 3] = copy.deepcopy(v_sc_km_s)
    x_dot[idx_vsc_start:idx_vsc_start + 3] = 1e-3 * HarmonicGravity(1e3 * r_sc_km) - \
        np.cross(omega_planet, np.cross(omega_planet, r_sc_km)) - 2.0 * np.cross(omega_planet, v_sc_km_s)
    
    x_dot[idx_sigma_start:idx_sigma_start + 3] = MRPDynamics(sigma_planet, omega_planet)
    return x_dot


def EquationsOfMotion_True(t: float, x: npt.NDArray):
    r_sc_m = x[0:3]
    v_sc_m_s = x[3:6]

    x_dot = np.zeros(len(x))
    x_dot[0:3] = v_sc_m_s
    x_dot[3:6] = HarmonicGravity_True(r_sc_m) - np.cross(Omega, np.cross(Omega, r_sc_m)) - 2.0 * np.cross(Omega, v_sc_m_s)
    return x_dot


def MeasurementModel(x: npt.NDArray, T_i_c_SPS: npt.NDArray):
    r_sc = x[0:3]
    v_sc = x[3:6]
    z_exp = np.zeros(7 * numSPS)
    for ee in range(numSPS):
        sigma_p_i = x[idx_sigma_start:idx_sigma_start + 3]
        omega_p_i = x[idx_omega_start:idx_omega_start + 3]
        
        r_SPS = cameraTruePositions[ee]

        r_sc_SPS = r_sc - r_SPS
        rho = np.linalg.norm(r_sc_SPS)
        rho_dot = np.dot(v_sc, r_sc_SPS / rho)
        alpha = np.arctan2(r_sc_SPS[1], r_sc_SPS[0])
        dec = np.arccos(r_sc_SPS[2] / rho)

        posIndex: int = 7 * ee
        z_exp[posIndex] = rho
        z_exp[posIndex + 1] = rho_dot
        z_exp[posIndex + 2] = alpha
        z_exp[posIndex + 3] = dec
        z_exp[posIndex + 4:posIndex + 7] = T_i_c_SPS[ee] @ \
            st.math.MRP_to_DCM(st.math.ModifiedRodriguesParameters(sigma_p_i)).T @ omega_p_i
    return z_exp


def IntegrateRK4(eom_func, t_span: tuple[float, float], x: npt.NDArray, t_step: float):
    out_x = copy.deepcopy(x)
    t_curr = t_span[0]
    while t_curr < t_span[1]:
        dt = min(t_step, t_span[1] - t_curr)
        k1 = dt * eom_func(t_curr, out_x)
        k2 = dt * eom_func(t_curr + 0.5 * dt, out_x + 0.5 * HandleShadowParams(k1))
        k3 = dt * eom_func(t_curr + 0.5 * dt, out_x + 0.5 * HandleShadowParams(k2))
        k4 = dt * eom_func(t_curr + dt, out_x + HandleShadowParams(k3))
        out_x += (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + HandleShadowParams(k4))
        t_curr += dt
    return HandleShadowParams(out_x)


def ApplyMask(vec: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    return vec * mask


mx_plus = copy.deepcopy(mx_0)
Pxx_plus = copy.deepcopy(Pxx_0)

sc_true_history: list[npt.NDArray] = []
z_history: list[npt.NDArray] = []
mx_history: list[npt.NDArray] = []
Pxx_history: list[npt.NDArray] = []

# alpha_underweight = 3.0
# gamma_underweight = 0.3
alpha_underweight = 1.0
gamma_underweight = 1.0
# T_g_c = np.identity(3)  # transformation from gravity to camera frame (no alignment)

# UKF params
alpha_ukf: float = 1.0
beta_ukf: float = 0.0
kappa_ukf: float = max(3.0 - numStates, 0.0)
lam_ukf: float = alpha_ukf * alpha_ukf * (numStates + kappa_ukf) - numStates

w0_m = lam_ukf / (lam_ukf + numStates)
wi_m = 1.0 / (2.0 * (lam_ukf + numStates))
w0_c = lam_ukf / (lam_ukf + numStates) + (1.0 - alpha_ukf * alpha_ukf + beta_ukf)
wi_c = 1.0 / (2.0 * (lam_ukf + numStates))
sqrt_n_lam = np.sqrt(numStates + lam_ukf)

SetCoeffsInGravityModel(mx_0)

# Spacecraft propagation
x_sc_true = np.concat((sc_pos, sc_vel))

# Default masks
mask_noAngVel = np.ones(numMeasurements)
for ee in range(numSPS):
    posIndex: int = 7 * ee
    mask_noAngVel[posIndex + 4:posIndex + 7] = np.zeros(3)

cutoff = 20
times = times[:cutoff]
times_days = times_days[:cutoff]
for j in range(len(times)):

    ######################################
    ####    Measurement Processing    ####
    ######################################

    angVelOutOfBounds: bool = True
    # angVelOutOfBounds: bool = j == 0 or j == len(times) - 1

    # tRightNow: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    # tRightNow += datetime.timedelta(seconds=timeStep_s)
    # st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tRightNow))
    
    doPrint: bool = j % printInterval == 0

    #################################################
    ####    Accumulate SPS Position Estimates    ####
    #################################################
    
    estimatedPositions: list[npt.NDArray] = []
    T_i_c_SPS: list[npt.NDArray] = []
    for ee in range(numSPS):
        # T_i_b: npt.NDArray = T_i_b_list[ee][j]
        # truth_j = truthData[ee][j]
        attitudeEst_j = attitudeEstData[ee][j]
        # gravEst_j = gravEstData[ee][j]
        
        if attitudeEst_j[0] == 999 or attitudeEst_j[1] == 999 or attitudeEst_j[2] == 999 or attitudeEst_j[3] == 999:
            st.OnScreenLogMessage(f'Warning: skipped measurement at index {j} (invalid quaternion).', "SPSGPS", st.Severity.Info)
            continue
        
        q_i_c = np.array([attitudeEst_j[1], attitudeEst_j[2], attitudeEst_j[3], attitudeEst_j[0]])
        T_i_c = st.math.Quat_to_DCM(normalize(q_i_c))
        T_i_c_SPS.append(T_i_c)

        # g_sensorFrame = np.array([gravEst_j[0], gravEst_j[1], gravEst_j[2]])

        # Coarse estimates
        # r_coarse_1 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, gravModel_est.mu, 
        #                                               Omega, planetData, planetFixedFrame, False, np.zeros(3), 0.0, 
        #                                               20, ThirdBodyGravityForMeasurements)
        # r_coarse_2 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, gravModel_est.mu, 
        #                                               Omega, planetData, planetFixedFrame, True, r_coarse_1, 0.0, 
        #                                               20, ThirdBodyGravityForMeasurements)
        
        # if doPrint:
            # st.OnScreenLogMessage(f'Sample point {j}:', "SPSGPS", st.Severity.Info)
            # st.OnScreenLogMessage(f'r_expected = {cameraTruePositions[ee]}', "SPSGPS", st.Severity.Info)
            # st.OnScreenLogMessage(f'r_coarse_1 = {r_coarse_1}', "SPSGPS", st.Severity.Info)
            # st.OnScreenLogMessage(f'r_coarse_2 = {r_coarse_2}', "SPSGPS", st.Severity.Info)
        
        # fineOutputs = st.ProcPlanet.SPS.FineEstimate(r_coarse_2, T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, Omega, 
        #                                              planetData, planetFixedFrame, gradientWalkFactor, tol, doPrint, j, 0.0, 
        #                                              20, HarmonicGravityForMeasurements, ThirdBodyGravityForMeasurements)

        # q_c_b = st.math.DCM_to_Quat(T_i_b @ T_i_c.T)

        # r_bestEstimate = fineOutputs.pos
        # estimatedPositions.append(r_bestEstimate)
    
    ########################################
    ####    Save States for Plotting    ####
    ########################################

    mx_plus = HandleShadowParams(mx_plus)

    sc_true_history.append(copy.deepcopy(x_sc_true))
    mx_history.append(copy.deepcopy(mx_plus))
    Pxx_history.append(copy.deepcopy(Pxx_plus))

    ######################################
    ####    Spacecraft Propagation    ####
    ######################################

    x_sc_true = IntegrateRK4(EquationsOfMotion_True, (0.0, timeStep_s), x_sc_true, propagationTimeStep_s)
    
    ##################################
    ####    Filter Propagation    ####
    ##################################

    st.OnScreenLogMessage(f'Filter iteration {j}', "SPSGPS", st.Severity.Info)

    # Sigma points:
    # SetCoeffsInGravityModel(mx_plus)
    chi_0 = IntegrateRK4(EquationsOfMotion, (0.0, timeStep_s), mx_plus, propagationTimeStep_s)
    mx_minus: npt.NDArray = w0_m * chi_0
    Sxx_plus = np.linalg.cholesky(Pxx_plus)
    chi_list_prop_1 = []
    chi_list_prop_2 = []
    for ii in range(numStates):
        chi_ii_1 = mx_plus + sqrt_n_lam * Sxx_plus[:, ii]
        chi_ii_2 = mx_plus - sqrt_n_lam * Sxx_plus[:, ii]
        SetCoeffsInGravityModel(chi_ii_1)
        chi_ii_1_prop = IntegrateRK4(EquationsOfMotion, (0.0, timeStep_s), chi_ii_1, propagationTimeStep_s)
        SetCoeffsInGravityModel(chi_ii_2)
        chi_ii_2_prop =  IntegrateRK4(EquationsOfMotion, (0.0, timeStep_s), chi_ii_2, propagationTimeStep_s)

        chi_list_prop_1.append(chi_ii_1_prop)
        chi_list_prop_2.append(chi_ii_2_prop)
        mx_minus += wi_m * chi_ii_1_prop
        mx_minus += wi_m * chi_ii_2_prop

    mx_minus = HandleShadowParams(mx_minus)
    st.OnScreenLogMessage(f'mx_minus = {mx_minus}', "SPSGPS", st.Severity.Info)

    Pxx_minus = w0_c * np.outer(mx_plus - mx_minus, mx_plus - mx_minus)
    for ii in range(numStates):
        chi_ii_1 = chi_list_prop_1[ii]
        chi_ii_2 = chi_list_prop_2[ii]
        Pxx_minus += wi_c * np.outer(chi_ii_1 - mx_minus, chi_ii_1 - mx_minus)
        Pxx_minus += wi_c * np.outer(chi_ii_2 - mx_minus, chi_ii_2 - mx_minus)
    Pxx_minus += Pww
    
    #############################
    ####    Filter Update    ####
    #############################

    if j > 0:
        st.OnScreenLogMessage(f"Measurement at index {j} should be: {omega_est_list[0][j - 1]}", "SPSGPS", st.Severity.Info)

    if not angVelOutOfBounds:
        # st.OnScreenLogMessage(f'Made it to filter update stage...', "SPSGPS", st.Severity.Info)

        # Sigma points and mz_minus
        Sxx_minus = np.linalg.cholesky(Pxx_minus)
        mz_minus: npt.NDArray = MeasurementModel(mx_minus, T_i_c_SPS)

        chi_list_minus = [copy.deepcopy(mx_minus)]
        # chi_list_minus = [copy.deepcopy(chi_0)]
        z_hx_list = [copy.deepcopy(mz_minus)]
        mz_minus = w0_m * mz_minus
        for ii in range(numStates):
            chi_ii_1 = mx_minus + sqrt_n_lam * Sxx_minus[:, ii]
            chi_ii_2 = mx_minus - sqrt_n_lam * Sxx_minus[:, ii]
            chi_list_minus.append(chi_ii_1)
            chi_list_minus.append(chi_ii_2)

            z_hx_ii_1 = MeasurementModel(chi_ii_1, T_i_c_SPS)
            z_hx_ii_2 = MeasurementModel(chi_ii_2, T_i_c_SPS)

            mz_minus += wi_m * z_hx_ii_1
            mz_minus += wi_m * z_hx_ii_2
            z_hx_list.append(z_hx_ii_1)
            z_hx_list.append(z_hx_ii_2)

        # st.OnScreenLogMessage(f'Made it to Pxz and Pzz calculation...', "SPSGPS", st.Severity.Info)

        # Pxz and Pzz
        Pxz_minus = w0_c * np.outer(chi_list_minus[0] - mx_minus, z_hx_list[0] - mz_minus)
        Pzz_minus = w0_c * np.outer(z_hx_list[0] - mz_minus, z_hx_list[0] - mz_minus)
        for ii in range(1, len(chi_list_minus)):
            Pxz_minus += wi_c * np.outer(chi_list_minus[ii] - mx_minus, z_hx_list[ii] - mz_minus)
            Pzz_minus += wi_c * np.outer(z_hx_list[ii] - mz_minus, z_hx_list[ii] - mz_minus)
        Pzz_minus += alpha_underweight * Pvv

        # Kalman gain
        K = Pxz_minus @ np.linalg.inv(Pzz_minus)
        
        # st.OnScreenLogMessage(f'Made it to measurement acquisition...', "SPSGPS", st.Severity.Info)
        z_new_SPS = np.zeros(numMeasurements)
        for ee in range(numSPS):
            r_SPS = 1e-3 * cameraTruePositions[ee]

            # Synthetic measurements
            r_sc_true = 1e-3 * x_sc_true[0:3]
            v_sc_true = 1e-3 * x_sc_true[3:6]
            r_sc_SPS = r_sc_true - r_SPS

            # TODO: Hardcoding covariances for now! Watch out...
            rho_synth = np.linalg.norm(r_sc_SPS)
            rho_dot_synth = np.dot(v_sc_true, r_sc_SPS / rho_synth) + rng.normal(0.0, err_rangeRate_km_s, 1).item()
            alpha_synth = np.arctan2(r_sc_SPS[1], r_sc_SPS[0]) + rng.normal(0.0, err_angle_rad, 1).item()
            dec_synth = np.arccos(r_sc_SPS[2] / rho_synth) + rng.normal(0.0, err_angle_rad, 1).item()
            rho_synth += rng.normal(0.0, err_range_km, 1).item()

            posIndex: int = 7 * ee
            z_new_SPS[posIndex] = rho_synth
            z_new_SPS[posIndex + 1] = rho_dot_synth
            z_new_SPS[posIndex + 2] = alpha_synth
            z_new_SPS[posIndex + 3] = dec_synth
            z_new_SPS[posIndex + 4:posIndex + 7] = omega_est_list[ee][j - 1]

        z_history.append(copy.deepcopy(z_new_SPS))

        mx_plus_candidate = mx_minus + gamma_underweight * K @ (z_new_SPS - mz_minus)
        Pxx_plus_candidate = Pxx_minus - Pxz_minus @ K.T - K @ Pxz_minus.T + K @ Pzz_minus @ K.T
        if True:
        # if (np.abs(mx_plus_candidate - mx_minus) < 6.0 * np.sqrt(np.diag(Pxx_minus))).all():
            mx_plus = copy.deepcopy(mx_plus_candidate)
            Pxx_plus = copy.deepcopy(Pxx_plus_candidate)
        else:
            st.OnScreenLogMessage(f"Measurement at index {j} not processed; exceed 6-sigma distance to mean.", "SPSGPS", st.Severity.Info)
            mx_plus = copy.deepcopy(mx_minus)
            Pxx_plus = copy.deepcopy(Pxx_minus)
    else:
        st.OnScreenLogMessage(f"Measurement at index {j} not processed; no angular velocity estimate exists at this index.", "SPSGPS", st.Severity.Info)
        mx_plus = copy.deepcopy(mx_minus)
        Pxx_plus = copy.deepcopy(Pxx_minus)

    # Update the gravity model
    SetCoeffsInGravityModel(mx_plus)
    
    #####################################
    ####    Clean-up and Printing    ####
    #####################################

    percentComplete = round(100.0 * float(j) / float(len(times)), 3)
    
    endTime = time.perf_counter()
    elapsedSeconds = endTime - startTime
    elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

    if doPrint:
        st.OnScreenLogMessage(f'Elapsed time: {elapsedTime}', "SPSGPS", st.Severity.Info)
        st.OnScreenLogMessage("------------------------------------------------------------------------------------------------------\n", "SPSGPS", st.Severity.Info)

########################
####    PLOTTING    ####
########################

###########################
####    Color Setup    ####
###########################

axisColors_main = ["#CC4444", 
                   "#33BB33", 
                   "#2266FF"]

axisColors_dark = ["#551111",
                   "#115511",
                   "#113388"]

sequenceColors_main = ["#FF0000",
                       "#FF4C00",
                       "#FF8000",
                       "#FFC400",
                       "#FFFF00",
                       "#80FF00",
                       "#00FF00",
                       "#00FF80",
                       "#00FFFF",
                       "#0080FF",
                       "#0000FF",
                       "#8000FF"]

sequenceColors_dark = ["#590000",
                       "#591b00",
                       "#592d00",
                       "#594500",
                       "#595900",
                       "#2d5900",
                       "#005900",
                       "#00592d",
                       "#005959",
                       "#002d59",
                       "#000059",
                       "#2d0059"]

##########################
####    Name Setup    ####
##########################

axisNames = [r"$x$", r"$y$", r"$z$"]

###########################################
####    Plot Information Extraction    ####
###########################################

# Spacecraft states
sc_true_r = np.array([sc_x[0:3] for sc_x in sc_true_history])
sc_true_v = np.array([sc_x[3:6] for sc_x in sc_true_history])
mx_sc_r = np.array([1e3 * mx[idx_rsc_start:idx_rsc_start + 3] for mx in mx_history])
mx_sc_v = np.array([1e3 * mx[idx_vsc_start:idx_vsc_start + 3] for mx in mx_history])
Pxx_sc_r = np.array([1e6 * np.diag(Pxx[idx_rsc_start:idx_rsc_start + 3, 
                                       idx_rsc_start:idx_rsc_start + 3]) for Pxx in Pxx_history])
Pxx_sc_v = np.array([1e6 * np.diag(Pxx[idx_vsc_start:idx_vsc_start + 3, 
                                       idx_vsc_start:idx_vsc_start + 3]) for Pxx in Pxx_history])

# Gravitational parameters
mx_Clm: list[npt.NDArray] = []
mx_Slm: list[npt.NDArray] = []
for ii in range(numClm):
    mx_Clm.append(np.array([mx[idx_Clm_start + ii] for mx in mx_history]))
for ii in range(numSlm):
    mx_Slm.append(np.array([mx[idx_Slm_start + ii] for mx in mx_history]))

Pxx_Clm: list[npt.NDArray] = []
Pxx_Slm: list[npt.NDArray] = []
for ii in range(numClm):
    Pxx_Clm.append(np.array([Pxx[idx_Clm_start + ii, idx_Clm_start + ii] for Pxx in Pxx_history]))
for ii in range(numSlm):
    Pxx_Slm.append(np.array([Pxx[idx_Slm_start + ii, idx_Slm_start + ii] for Pxx in Pxx_history]))

# GM
mx_mu = np.array([mx[idx_mu] for mx in mx_history])
Pxx_mu = np.array([Pxx[idx_mu, idx_mu] for Pxx in Pxx_history])

# Planetary attitude dynamics
truth_sigma = np.array([st.math.DCM_to_MRP(T_i_b).ToVector() for T_i_b in T_i_b_list[0]])
for ii in range(len(truth_sigma)):
    if np.linalg.norm(truth_sigma[ii]) >= 1.0:
        truth_sigma[ii] = st.math.ModifiedRodriguesParameters(truth_sigma[ii]).shadow().ToVector()

mx_sigma = np.array([mx[idx_sigma_start:idx_sigma_start + 3] for mx in mx_history])
mx_omega = np.array([mx[idx_omega_start:idx_omega_start + 3] for mx in mx_history])
Pxx_sigma = np.array([np.diag(Pxx[idx_sigma_start:idx_sigma_start + 3, 
                                  idx_sigma_start:idx_sigma_start + 3]) for Pxx in Pxx_history])
Pxx_omega = np.array([np.diag(Pxx[idx_omega_start:idx_omega_start + 3, 
                                  idx_omega_start:idx_omega_start + 3]) for Pxx in Pxx_history])

st.OnScreenLogMessage(f'truth_sigma = {truth_sigma[:cutoff]}', "SPSGPS", st.Severity.Info)
st.OnScreenLogMessage(f'mx_sigma = {mx_sigma}', "SPSGPS", st.Severity.Info)

# Spacecraft position and velocity
fig1 = plt.figure(layout='constrained')
ax1_1 = fig1.add_subplot(231)
ax1_2 = fig1.add_subplot(232)
ax1_3 = fig1.add_subplot(233)
ax1_4 = fig1.add_subplot(234)
ax1_5 = fig1.add_subplot(235)
ax1_6 = fig1.add_subplot(236)
sc_pos_axes = [ax1_1, ax1_2, ax1_3]
sc_vel_axes = [ax1_4, ax1_5, ax1_6]

for i in range(len(sc_pos_axes)):
    ax = sc_pos_axes[i]
    axName = axisNames[i]
    ax.plot(times_days, mx_sc_r[:, i] - sc_true_r[:, i], label=axName + "-Axis Position", color=axisColors_main[i])
    ax.plot(times_days, -3.0 * np.sqrt(Pxx_sc_r[:, i]), linestyle='dashed', color=axisColors_dark[i], label=axName + r"-Axis Position 3$\sigma$ Interval")
    ax.plot(times_days, 3.0 * np.sqrt(Pxx_sc_r[:, i]), linestyle='dashed', color=axisColors_dark[i])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Error in " + axName + "-Axis Position over Time")
    ax.grid()
    ax.legend()

for i in range(len(sc_vel_axes)):
    ax = sc_vel_axes[i]
    axName = axisNames[i]
    ax.plot(times_days, mx_sc_v[:, i] - sc_true_v[:, i], label=axName + "-Axis Velocity", color=axisColors_main[i])
    ax.plot(times_days, -3.0 * np.sqrt(Pxx_sc_v[:, i]), linestyle='dashed', color=axisColors_dark[i], label=axName + r"-Axis Velocity 3$\sigma$ Interval")
    ax.plot(times_days, 3.0 * np.sqrt(Pxx_sc_v[:, i]), linestyle='dashed', color=axisColors_dark[i])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Velocity error (m/s)")
    ax.set_title("Error in " + axName + "-Axis Velocity over Time")
    ax.grid()
    ax.legend()

fig2 = plt.figure(layout='constrained')
ax2_1 = fig2.add_subplot(231)
ax2_2 = fig2.add_subplot(232)
ax2_3 = fig2.add_subplot(233)
ax2_4 = fig2.add_subplot(234)
ax2_5 = fig2.add_subplot(235)
ax2_6 = fig2.add_subplot(236)

sigma_axes = [ax2_1, ax2_2, ax2_3]
omega_axes = [ax2_4, ax2_5, ax2_6]

for ii in range(len(sigma_axes)):
    ax = sigma_axes[ii]
    axName = axisNames[ii]
    ax.plot(times_days, mx_sigma[:, ii] - truth_sigma[:cutoff, ii], label=axName + "-Axis MRP Error", color=axisColors_main[ii])
    ax.plot(times_days, -3.0 * np.sqrt(Pxx_sigma[:, ii]), linestyle='dashed', color=axisColors_dark[ii], label=axName + r"-Axis MRP 3$\sigma$ Intervals")
    ax.plot(times_days, 3.0 * np.sqrt(Pxx_sigma[:, ii]), linestyle='dashed', color=axisColors_dark[ii])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("MRP Error (m)")
    ax.set_title("MRP Error in " + axName + "-Axis over Time")
    ax.grid()
    ax.legend()

for ii in range(len(omega_axes)):
    ax = omega_axes[ii]
    axName = axisNames[ii]
    ax.plot(times_days, mx_omega[:, ii] - np.asarray(angvel_i_b_list[0])[:cutoff, ii], label=axName + "-Axis Angular Velocity Error", color=axisColors_main[ii])
    ax.plot(times_days, -3.0 * np.sqrt(Pxx_omega[:, ii]), linestyle='dashed', color=axisColors_dark[ii], label=axName + r"-Axis Angular Velocity 3$\sigma$ Intervals")
    ax.plot(times_days, 3.0 * np.sqrt(Pxx_omega[:, ii]), linestyle='dashed', color=axisColors_dark[ii])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Angular Velocity Error (m)")
    ax.set_title("Angular Velocity Error in " + axName + "-Axis over Time")
    ax.grid()
    ax.legend()

fig3 = plt.figure(layout='constrained')
ax3 = fig3.add_subplot(111, projection='3d')

def disable_vert_rotation(event):
    azim = ax3.azim
    elev = ax3.elev
    ax3.view_init(elev=elev, azim=azim, roll=0.0)

fig3.canvas.mpl_connect('motion_notify_event', disable_vert_rotation)

ax3.plot(sc_true_r[:, 0], sc_true_r[:, 1], sc_true_r[:, 2], color=axisColors_main[1], label='True Trajectory')
ax3.plot(mx_sc_r[:, 0], mx_sc_r[:, 1], mx_sc_r[:, 2], color=axisColors_main[0], label='Estimated Trajectory')
ax3.scatter(sc_true_r[0, 0], sc_true_r[0, 1], sc_true_r[0, 2], color=sequenceColors_main[3], label='Start')
ax3.scatter(sc_true_r[-1, 0], sc_true_r[-1, 1], sc_true_r[-1, 2], color=sequenceColors_main[-1], label='End')
ax3.set_xlabel(axisNames[0] + "-Axis Position (m)")
ax3.set_ylabel(axisNames[1] + "-Axis Position (m)")
ax3.set_zlabel(axisNames[2] + "-Axis Position (m)")
ax3.set_title("3D Trajectory")
ax3.grid()
ax3.legend()


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

set_axes_equal(ax3)

# Gravitational parameter and select spherical harmonic coefficients
fig4 = plt.figure(layout='constrained')
ax4_1 = fig4.add_subplot(131)
ax4_2 = fig4.add_subplot(132)
ax4_3 = fig4.add_subplot(133)

# Plot C[l,m]
for ii in range(numClm):
    idx_l = indices_lowerTriangular_Clm[0][ii + 3]
    idx_m = indices_lowerTriangular_Clm[1][ii + 3]

    mx_Clm_percentError = 100.0 * (mx_Clm[ii] - gravModel_true.Clm[idx_l, idx_m]) / gravModel_true.Clm[idx_l, idx_m]
    Pxx_Clm_percentError = 100.0 * np.sqrt(Pxx_Clm[ii]) / np.abs(gravModel_true.Clm[idx_l, idx_m])

    if ii == 0:
        ax4_1.plot(times_days, mx_Clm[ii] - gravModel_true.Clm[idx_l, idx_m], color=sequenceColors_main[ii % len(sequenceColors_main)], label=r"Error in $C[2,0]$")
        ax4_1.plot(times_days, -3.0 * np.sqrt(Pxx_Clm[ii]), linestyle='dashed', color=sequenceColors_dark[ii % len(sequenceColors_dark)], label=r"$C[2,0]$ 3$\sigma$ Intervals")
        ax4_1.plot(times_days, 3.0 * np.sqrt(Pxx_Clm[ii]), linestyle='dashed', color=sequenceColors_dark[ii % len(sequenceColors_dark)])
    else:
        ax4_1.plot(times_days, mx_Clm[ii] - gravModel_true.Clm[idx_l, idx_m], color=sequenceColors_main[ii % len(sequenceColors_main)])
        ax4_1.plot(times_days, -3.0 * np.sqrt(Pxx_Clm[ii]), linestyle='dashed', color=sequenceColors_dark[ii % len(sequenceColors_dark)])
        ax4_1.plot(times_days, 3.0 * np.sqrt(Pxx_Clm[ii]), linestyle='dashed', color=sequenceColors_dark[ii % len(sequenceColors_dark)])

customHandles_Clm = [
    Line2D([0], [0], color="dimgray", label=r"Error in $C[l,m]$"),
    Line2D([0], [0], color="dimgray", ls="--", label=r"$C[l,m]$ 3$\sigma$ Intervals")
]
currentHandles, currentLabels = ax4_1.get_legend_handles_labels()
allHandles = currentHandles + customHandles_Clm

ax4_1.set_xlabel("Time (days)")
ax4_1.set_ylabel(r"Normalized $C[l,m]$ Coefficients")
ax4_1.set_title(r"Error in Normalized $C[l,m]$ Coefficients over Time")
ax4_1.grid()
ax4_1.legend(handles=allHandles)

# Plot S[l,m]
for ii in range(numSlm):
    idx_l = indices_lowerTriangular_Slm[0][ii + 1] + 1
    idx_m = indices_lowerTriangular_Slm[1][ii + 1] + 1

    mx_Slm_percentError = 100.0 * (mx_Slm[ii] - gravModel_true.Slm[idx_l, idx_m]) / gravModel_true.Slm[idx_l, idx_m]
    Pxx_Slm_percentError = 100.0 * np.sqrt(Pxx_Slm[ii]) / np.abs(gravModel_true.Slm[idx_l, idx_m])
    
    ax4_2.plot(times_days, mx_Slm[ii] - gravModel_true.Slm[idx_l, idx_m], color=sequenceColors_main[ii % len(sequenceColors_main)])
    ax4_2.plot(times_days, -3.0 * np.sqrt(Pxx_Slm[ii]), linestyle='dashed', color=sequenceColors_dark[ii % len(sequenceColors_dark)])
    ax4_2.plot(times_days, 3.0 * np.sqrt(Pxx_Slm[ii]), linestyle='dashed', color=sequenceColors_dark[ii % len(sequenceColors_dark)])

customHandles_Slm = [
    Line2D([0], [0], color="dimgray", label=r"Error in $S[l,m]$"),
    Line2D([0], [0], color="dimgray", ls="--", label=r"$S[l,m]$ 3$\sigma$ Intervals")
]
currentHandles, currentLabels = ax4_2.get_legend_handles_labels()
allHandles = currentHandles + customHandles_Slm

ax4_2.set_xlabel("Time (days)")
ax4_2.set_ylabel(r"Normalized $S[l,m]$ Coefficients")
ax4_2.set_title(r"Error in Normalized $S[l,m]$ Coefficients over Time")
ax4_2.grid()
ax4_2.legend(handles=allHandles)

# Plot GM
ax4_3.plot(times_days, mx_mu - 1e-9 * gravModel_true.mu, color='dimgray', label=r'Gravitational Parameter ($\mu$) Error')
ax4_3.plot(times_days, -3.0 * np.sqrt(Pxx_mu), linestyle='dashed', color='black', label=r"$\mu$ Error 3$\sigma$ Interval")
ax4_3.plot(times_days, 3.0 * np.sqrt(Pxx_mu), linestyle='dashed', color='black')
ax4_3.set_xlabel("Time (days)")
ax4_3.set_ylabel(r"Error in $\mu$ (km^3/s^2)")
ax4_3.set_title(r"Error in Gravitational Parameter $\mu$ over Time")
ax4_3.grid()
ax4_3.legend()


plt.show()

st.leave_sim()
