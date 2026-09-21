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
sys.path.append(thisRepo)

from py_src.star_tracker.star_tracker import main
from py_src.star_tracker.star_tracker.cam_matrix import *
from py_src.star_tracker.star_tracker.array_transformations import *

from SPS.grav_vesta_DAWN import grav_vesta_DAWN, grav_base

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


def quat_mult(q1, q2):
    w = q1[3]
    x = q1[0]
    y = q1[1]
    z = q1[2]
    w2 = q2[3]
    x2 = q2[0]
    y2 = q2[1]
    z2 = q2[2]
    return np.array([w * x2 + x * w2 + y * z2 - z * y2,
                     w * y2 - x * z2 + y * w2 + z * x2,
                     w * z2 + x * y2 - y * x2 + z * w2,
                     w * w2 - x * x2 - y * y2 - z * z2])


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

numImages: int = 100

##############################
####    INITIALIZATION    ####
##############################

site = "Vesta"

hash_object = hashlib.sha256(site.encode('utf-8'))
int_seed = int(hash_object.hexdigest(), 16)

#################################
####    PLANET PARAMETERS    ####
#################################

J2000: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "J2000Frame")
J2000Frame = J2000.GetBodyFixedFrame()
planetEntity: st.Entity = st.GetThisSystem().GetParam(st.VarType.entityRef, "Planet")
planetFixedFrame = planetEntity.GetBodyFixedFrame()
planetName: str = planetEntity.getName()

cameraEntities: list[st.Entity] = st.GetThisSystem().GetParamArray(st.VarType.entityRef, "SPSCameras")
orbiter: st.Entity = st.GetThisSystem().GetParam(st.VarType.entityRef, "Orbiter")

outputDir = os.path.join(thisRepo, "output", planetName)
renderDirsLocal: list[str] = []
renderDirs: list[str] = []
for i in range(len(cameraEntities)):
    renderDirsLocal.append(os.path.join("Local", "Repos", "SPS-Star-Tracker", "output", planetName, "Renders", str(i).zfill(3)))
    renderDirs.append(os.path.join(outputDir, "Renders", str(i).zfill(3)))

truthDataPaths: list[str] = []
attitudeEstDataPaths: list[str] = []
gravTruthDataPaths: list[str] = []
gravEstDataPaths: list[str] = []
q_i_b_DataPaths: list[str] = []

for i in range(len(cameraEntities)):
    truthDataPaths.append(os.path.join(outputDir, "truth_data_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    attitudeEstDataPaths.append(os.path.join(outputDir, "attitudes_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    gravTruthDataPaths.append(os.path.join(outputDir, "true_gravities_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    gravEstDataPaths.append(os.path.join(outputDir, "measurements_" + planetName + "_" + str(i).zfill(3) + ".csv"))
    q_i_b_DataPaths.append(os.path.join(outputDir, "q_i_b_" + planetName + "_" + str(i).zfill(3) + ".csv"))

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
    for i in range(len(cameraEntities)):
        if os.path.exists(renderDirs[i]):
            shutil.rmtree(renderDirs[i])
        Path(renderDirs[i]).mkdir(parents=True)

# Time
t0: st.timestamp = st.timestamp.from_datetime(datetime.datetime(year=2026, month=5, day=22, hour=16))
st.SimGlobals.SimClock.ResetTo(t0)

st.OnScreenLogMessage(f"t0 = {t0.as_utc_string()}", "SPSGPS", st.Severity.Info)

time.sleep(0.1)

timeStep_s: float = 100.0

true_data: list[npt.NDArray] = []
true_accelerations: list[npt.NDArray] = []
measured_accelerations: list[npt.NDArray] = []
q_i_b_data: list[npt.NDArray] = []

times = timeStep_s * np.linspace(0.0, numImages - 1, numImages)

# mu: float = 1e9 * planetEntity.GetParam(st.VarType.double, ["Dynamics", "GravitationalParameter_km3_s2"])
radiusEquatorial: float = planetEntity.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])
Omega = planetEntity.getAngVelocity().WRT(J2000Frame).ExprIn(planetFixedFrame)

st.OnScreenLogMessage(f"Planet angular velocity (Omega) = {np.rad2deg(Omega)} deg/s", "SPSGPS", st.Severity.Info)
# Omega = np.array([0.0, 0.0, 2.66e-6])  # Expressed in the planet-fixed frame

startTime = time.perf_counter()
elapsedSeconds: float = 0.0
printInterval: int = 10

cameraTruePositions: list[npt.NDArray] = []
for i in range(len(cameraEntities)):
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

for ee in range(len(cameraEntities)):
    if not os.path.exists(renderDirs[ee]):
        Path(renderDirs[ee]).mkdir(parents=True)

    if is_dir_empty(renderDirs[ee]):
        # locs: list[npt.NDArray] = []
        # vels: list[npt.NDArray] = []
        # rots: list[npt.NDArray] = []
        # names: list[str] = []
        
        planetStateDummy = st.frames.FramedLocVelAcc(st.frames.rva_struct(np.zeros(3), np.zeros(3), np.zeros(3)), planetFixedFrame)
        tNow: datetime.datetime = t0.as_datetime()
        st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tNow))
        for i in range(numImages):
            doPrint: bool = i % printInterval == 0 and globalDoPrint

            # Sample gravity vector
            stateNow = st.frames.FramedLocVelAcc(st.frames.rva_struct(cameraTruePositions[ee], np.zeros(3), np.zeros(3)), planetFixedFrame)
            g = st.SimGlobals.SampleVectorField("Gravity", stateNow).ExprIn(planetFixedFrame)

            g_thirdBody = st.SimGlobals.SampleVectorField("ThirdBodyGravity", stateNow).ExprIn(planetFixedFrame)
            g_thirdBody -= st.SimGlobals.SampleVectorField("ThirdBodyGravity", planetStateDummy).ExprIn(planetFixedFrame)
            st.OnScreenLogMessage(f"True third body gravity = {g_thirdBody}", "SPSGPS", st.Severity.Info)

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
            _q_i_b = st.math.DCM_to_Quat(planetRot)
            q_i_b_data.append(_q_i_b)

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

            T_PCPF_S_true = ra_dec_to_rot(ra_pcpf, de_pcpf).T
            q_PCPF_S_true_passive = st.math.DCM_to_Quat(T_PCPF_S_true)

            # Render (old version)
            pos_framed = st.frames.FramedLoc(cameraTruePositions[ee], planetFixedFrame)
            vel_framed = st.frames.FramedLocVel(st.frames.rv_struct(cameraTruePositions[ee], np.zeros(3)), planetFixedFrame)

            cameraEntities[ee].setRotation(st.frames.FramedRot(T_I_S_true, J2000Frame))

            EridaniRenderPayload = st.ParamMap()
            EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Loc", pos_framed.WRT_ExprIn(J2000Frame))
            EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Vel", vel_framed.vel_WRT_ExprIn(J2000Frame))
            EridaniRenderPayload.AddParam(st.VarType.doubleV4, "Rot", q_I_S_true_passive)
            EridaniRenderPayload.AddParam(st.VarType.entityRef, "Frame", J2000)
            EridaniRenderPayload.AddParam(st.VarType.string, "NameOverride", "SPSRender" + str(i).zfill(5))
            EridaniRenderPayload.AddParam(st.VarType.string, "OutputPathOverride", str(renderDirsLocal[ee]))
            payload = st.SimGlobals.Request("EridaniSingleRender", EridaniRenderPayload, timeout=datetime.timedelta(seconds=60.0))

            if doPrint:
                st.OnScreenLogMessage(f'Rendering image {i} of {numImages} ({round(100.0 * float(i) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}', "SPSGPS", st.Severity.Info)

            true_data.append(q_I_S_true_passive)

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

    else:
        st.OnScreenLogMessage(f"Render directory {ee} not empty; skipping render step...\n", "SPSGPS", st.Severity.Info)

#########################################
####    OBTAIN ATTITUDE ESTIMATES    ####
#########################################

for ee in range(len(cameraEntities)):
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

T_i_b_list: list[list[npt.NDArray]] = []
T_i_c_list: list[list[npt.NDArray]] = []
g_est_list: list[list[npt.NDArray]] = []

for ee in range(len(cameraEntities)):
    truthData.append(read_csv(truthDataPaths[ee]))
    attitudeEstData.append(read_csv(attitudeEstDataPaths[ee], ignore=[0, 1, 2, 3], hasHeader=True))
    gravTruthData.append(read_csv(gravTruthDataPaths[ee]))
    gravEstData.append(read_csv(gravEstDataPaths[ee]))
    q_i_b_list.append(read_csv(q_i_b_DataPaths[ee]))

    # Very basic error handling if datasets are not the same length
    if not (len(truthData) == len(attitudeEstData) == len(gravEstData)):
        st.OnScreenLogMessage(f'Warning: early exit due to dataset length mismatch; truthData length = {len(truthData)}, attitudeEstData length = {len(attitudeEstData)}, and gravEstData length = {len(gravEstData)}.', "SPSGPS", st.Severity.Info)
        exit(0)

    # Initialize all inertial-to-planet attitude matrices
    T_i_b_list.append([])
    T_i_c_list.append([])
    g_est_list.append([])
    for i in range(len(times)):
        T_i_b_list[ee].append(st.math.Quat_to_DCM(normalize(q_i_b_list[ee][i])))
        q_i_c = np.array([attitudeEstData[ee][i][1], attitudeEstData[ee][i][2], attitudeEstData[ee][i][3], attitudeEstData[ee][i][0]])
        T_i_c_list[ee].append(st.math.Quat_to_DCM(normalize(q_i_c)))
        g_est_list[ee].append(gravEstData[ee][i])

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
sc_vel = orbiter.GetParam(st.VarType.doubleV3, "Velocity")

st.OnScreenLogMessage(f"sc_pos = {sc_pos}", "SPSGPS", st.Severity.Info)
st.OnScreenLogMessage(f"sc_vel = {sc_vel}", "SPSGPS", st.Severity.Info)

# Setting up the two gravity models
estDegree: int = 4  # = estOrder, keep them the same for simplicity
gravModel_true: grav_base = grav_vesta_DAWN()
gravModel_est: grav_base = grav_vesta_DAWN()
trueGravSampler = GravSampler(gravModel_true, 20, 20)
customGravSampler = GravSampler(gravModel_est, estDegree, estDegree)

mu_true = gravModel_true.mu
Clm_true = gravModel_true.Clm
Slm_true = gravModel_true.Slm

#############################################
####    Run the filter in km and km/s    ####
#############################################

# Mean initialization for Kalman filter
mx_0: npt.NDArray = np.concat((1e-3 * sc_pos, 1e-3 * sc_vel))
for ee in range(len(cameraEntities)):
    mx_0 = np.concat((mx_0, 1e-3 * cameraTruePositions[ee]))

mx_0 = np.concat((mx_0, np.array([gravModel_true.mu * 1e-9])))

numSphericalHarmonicStates: int = 0
for n in range(2, estDegree + 1):
    numSphericalHarmonicStates += 2 * (n + 1)
mx_0 = np.concat((mx_0, np.zeros(numSphericalHarmonicStates)))

numStates: int = len(mx_0)
st.OnScreenLogMessage(f"Number of KF states = {numStates}", "SPSGPS_KF_Initialization", st.Severity.Info)

# Covariance initialization for Kalman filter
Pxx_0: npt.NDArray = np.zeros((numStates, numStates))

# Spacecraft position covariance
Pxx_0[0:3, 0:3] = np.identity(3) * 1.0 ** 2  # m

# Spacecraft velocity covariance
Pxx_0[3:6, 3:6] = np.identity(3) * 0.01 ** 2  # m/s

# SPS position covariance
for ee in range(len(cameraEntities)):
    Pxx_0[6 + 3 * ee:9 + 3 * ee, 6 + 3 * ee:9 + 3 * ee] = np.diag(np.array([1.0, 1.0, 1.0])) * 2.0 ** 2

# Gravitational parameter covariance
Pxx_0[6 + 3 * len(cameraEntities), 6 + 3 * len(cameraEntities)] = 0.1 ** 2

# Spherical harmonic coefficient covariance
for sh in range(numSphericalHarmonicStates):
    Pxx_0[6 + 3 * len(cameraEntities) + 1 + sh, 6 + 3 * len(cameraEntities) + 1 + sh] = 0.005 ** 2

# Override C[2,0]
Pxx_0[6 + 3 * len(cameraEntities) + 1, 6 + 3 * len(cameraEntities) + 1] = 0.05 ** 2

numMeasurements: int = 7 * len(cameraEntities)
Pww: npt.NDArray = Pxx_0 * 0.1 ** 2
# Override gravitational parameter process noise covariance
Pww[6 + 3 * len(cameraEntities), 6 + 3 * len(cameraEntities)] = 0.0002 ** 2

Pvv: npt.NDArray = np.zeros((numMeasurements, numMeasurements))
for ee in range(len(cameraEntities)):
    posIndex: int = 7 * ee
    Pvv[posIndex:posIndex + 3, posIndex:posIndex + 3] = np.diag(np.ones(3)) * 0.5 ** 2
    Pvv[posIndex + 3, posIndex + 3] = 0.005 ** 2
    Pvv[posIndex + 4, posIndex + 4] = 0.0005 ** 2
    Pvv[posIndex + 5, posIndex + 5] = 0.001 ** 2
    Pvv[posIndex + 6, posIndex + 6] = 0.001 ** 2

# Add a bit of scatter to initial mean based on Pww
# st.OnScreenLogMessage(f"mx_0 before Pww = {mx_0}", "SPSGPS", st.Severity.Info)
mx_0 += np.linalg.cholesky(Pww) @ rng.normal(0.0, 1.0, numStates)
# st.OnScreenLogMessage(f"mx_0 after Pww  = {mx_0}", "SPSGPS", st.Severity.Info)


def SetCoeffsInGravityModel(x: npt.NDArray):
    mu_new = x[6 + 3 * len(cameraEntities)]
    startIdxClm: int = int(6 + 3 * len(cameraEntities) + 1)
    startIdxSlm: int = startIdxClm + int(numSphericalHarmonicStates / 2)
    Clm_linear = np.concat((np.zeros(3), x[startIdxClm:startIdxSlm]))
    Slm_linear = np.concat((np.zeros(3), x[startIdxSlm:int(startIdxSlm + numSphericalHarmonicStates / 2)]))

    Clm = np.zeros((estDegree + 1, estDegree + 1))
    Slm = np.zeros((estDegree + 1, estDegree + 1))

    tril_indices_Clm = np.tril_indices(estDegree + 1)
    tril_indices_Slm = np.tril_indices(estDegree + 1)

    Clm[tril_indices_Clm] = Clm_linear
    Slm[tril_indices_Slm] = Slm_linear
    # st.OnScreenLogMessage(f"Clm: {Clm}", "SPSGPS", st.Severity.Info)

    gravModel_est.mu = 1e9 * mu_new  # km^3/s^2 to m^3/s^2
    gravModel_est.Clm = Clm
    gravModel_est.Slm = Slm


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


def EquationsOfMotion(t: float, x: npt.NDArray):
    x_dot = np.zeros(len(x))
    x_dot[0:3] = copy.deepcopy(x[3:6])
    x_dot[3:6] = 1e-3 * HarmonicGravity(1e3 * x[0:3])
    # st.OnScreenLogMessage(f'len(x_dot) = {len(x_dot)}', "SPSGPS", st.Severity.Info)
    return x_dot


def EquationsOfMotion_Spacecraft(t: float, x: npt.NDArray):
    x_dot = np.zeros(len(x))
    x_dot[0:3] = x[3:6]
    x_dot[3:6] = HarmonicGravity_True(x[0:3])
    return x_dot


def MeasurementModel(x: npt.NDArray):
    r_sc = x[0:3]
    v_sc = x[3:6]
    z_exp = np.zeros(7 * len(cameraEntities))
    for ee in range(len(cameraEntities)):
        r_SPS = x[6 + 3 * ee:9 + 3 * ee]
        r_sc_SPS = r_sc - r_SPS
        rho = np.linalg.norm(r_sc_SPS)
        rho_dot = np.dot(v_sc, r_sc_SPS / rho)
        alpha = np.arctan2(r_sc_SPS[1], r_sc_SPS[0])
        dec = np.arccos(r_sc_SPS[2] / rho)

        posIndex: int = 7 * ee
        z_exp[posIndex:posIndex + 3] = r_SPS
        z_exp[posIndex + 3] = rho
        z_exp[posIndex + 4] = rho_dot
        z_exp[posIndex + 5] = alpha
        z_exp[posIndex + 6] = dec
    return z_exp


def IntegrateRK4(eom_func, t_span: tuple[float, float], x: npt.NDArray, t_step: float):
    out_x = copy.deepcopy(x)
    t_curr = t_span[0]
    while t_curr < t_span[1]:
        dt = min(t_step, t_span[1] - t_curr)
        k1 = eom_func(t_curr, out_x)
        k2 = eom_func(t_curr + 0.5 * dt, out_x + 0.5 * k1 * dt)
        k3 = eom_func(t_curr + 0.5 * dt, out_x + 0.5 * k2 * dt)
        k4 = eom_func(t_curr + dt, out_x + k3 * dt)
        out_x += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_curr += dt
    return out_x


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
T_g_c = np.identity(3)  # transformation from gravity to camera frame (no alignment)

# UKF params
alpha_ukf: float = 1.0
beta_ukf: float = 2.0
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
propagationTimeStep_s = 50.0

# times = times[:40]
for j in range(len(times)):

    ######################################
    ####    Measurement Processing    ####
    ######################################

    # tRightNow: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    # tRightNow += datetime.timedelta(seconds=timeStep_s)
    # st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tRightNow))
    
    doPrint: bool = j % printInterval == 0

    #################################################
    ####    Accumulate SPS Position Estimates    ####
    #################################################
    
    estimatedPositions: list[npt.NDArray] = []
    for ee in range(len(cameraEntities)):
        T_i_b: npt.NDArray = T_i_b_list[ee][j]
        truth_j = truthData[ee][j]
        attitudeEst_j = attitudeEstData[ee][j]
        gravEst_j = gravEstData[ee][j]
        
        if attitudeEst_j[0] == 999 or attitudeEst_j[1] == 999 or attitudeEst_j[2] == 999 or attitudeEst_j[3] == 999:
            st.OnScreenLogMessage(f'Warning: skipped measurement at index {j} (invalid quaternion).', "SPSGPS", st.Severity.Info)
            continue
        
        q_i_c = np.array([attitudeEst_j[1], attitudeEst_j[2], attitudeEst_j[3], attitudeEst_j[0]])
        T_i_c = st.math.Quat_to_DCM(normalize(q_i_c))

        g_sensorFrame = np.array([gravEst_j[0], gravEst_j[1], gravEst_j[2]])

        # Coarse estimates
        r_coarse_1 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, gravModel_est.mu, 
                                                      Omega, planetData, planetFixedFrame, False, np.zeros(3), 0.0, 
                                                      20, ThirdBodyGravityForMeasurements)
        r_coarse_2 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, gravModel_est.mu, 
                                                      Omega, planetData, planetFixedFrame, True, r_coarse_1, 0.0, 
                                                      20, ThirdBodyGravityForMeasurements)
        
        # if doPrint:
            # st.OnScreenLogMessage(f'Sample point {j}:', "SPSGPS", st.Severity.Info)
            # st.OnScreenLogMessage(f'r_expected = {cameraTruePositions[ee]}', "SPSGPS", st.Severity.Info)
            # st.OnScreenLogMessage(f'r_coarse_1 = {r_coarse_1}', "SPSGPS", st.Severity.Info)
            # st.OnScreenLogMessage(f'r_coarse_2 = {r_coarse_2}', "SPSGPS", st.Severity.Info)
        
        fineOutputs = st.ProcPlanet.SPS.FineEstimate(r_coarse_2, T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, Omega, 
                                                     planetData, planetFixedFrame, gradientWalkFactor, tol, doPrint, j, 0.0, 
                                                     20, HarmonicGravityForMeasurements, ThirdBodyGravityForMeasurements)

        # q_c_b = st.math.DCM_to_Quat(T_i_b @ T_i_c.T)

        r_bestEstimate = fineOutputs.pos
        estimatedPositions.append(r_bestEstimate)
    
    ########################################
    ####    Save States for Plotting    ####
    ########################################

    sc_true_history.append(copy.deepcopy(x_sc_true))
    mx_history.append(copy.deepcopy(mx_plus))
    Pxx_history.append(copy.deepcopy(Pxx_plus))

    ######################################
    ####    Spacecraft Propagation    ####
    ######################################

    x_sc_true = IntegrateRK4(EquationsOfMotion_Spacecraft, (0.0, timeStep_s), x_sc_true, propagationTimeStep_s)
    
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

    # Sigma points and mz_minus
    Sxx_minus = np.linalg.cholesky(Pxx_minus)
    mz_minus: npt.NDArray = MeasurementModel(mx_minus)
    chi_list_minus = [copy.deepcopy(mx_minus)]
    # chi_list_minus = [copy.deepcopy(chi_0)]
    z_hx_list = [copy.deepcopy(mz_minus)]
    mz_minus = w0_m * mz_minus
    for ii in range(numStates):
        chi_ii_1 = mx_minus + sqrt_n_lam * Sxx_minus[:, ii]
        chi_ii_2 = mx_minus - sqrt_n_lam * Sxx_minus[:, ii]
        # chi_ii_1 = copy.deepcopy(chi_list_plus_1[ii])
        # chi_ii_2 = copy.deepcopy(chi_list_plus_2[ii])
        chi_list_minus.append(chi_ii_1)
        chi_list_minus.append(chi_ii_2)

        z_hx_ii_1 = MeasurementModel(chi_ii_1)
        z_hx_ii_2 = MeasurementModel(chi_ii_2)
        mz_minus += wi_m * z_hx_ii_1
        mz_minus += wi_m * z_hx_ii_2
        z_hx_list.append(z_hx_ii_1)
        z_hx_list.append(z_hx_ii_2)

    # Pxz and Pzz
    # st.OnScreenLogMessage(f"mx_diff = {chi_list_minus[0] - mx_minus}", "SPSGPS", st.Severity.Info)
    Pxz_minus = w0_c * np.outer(chi_list_minus[0] - mx_minus, z_hx_list[0] - mz_minus)
    Pzz_minus = w0_c * np.outer(z_hx_list[0] - mz_minus, z_hx_list[0] - mz_minus)
    for ii in range(1, len(chi_list_minus)):
        Pxz_minus += wi_c * np.outer(chi_list_minus[ii] - mx_minus, z_hx_list[ii] - mz_minus)
        Pzz_minus += wi_c * np.outer(z_hx_list[ii] - mz_minus, z_hx_list[ii] - mz_minus)
    Pzz_minus += alpha_underweight * Pvv

    # Kalman gain
    K = Pxz_minus @ np.linalg.inv(Pzz_minus)
    # st.OnScreenLogMessage(f"Pxz_minus = {Pxz_minus}", "SPSGPS", st.Severity.Info)
    # st.OnScreenLogMessage(f"Pzz_minus = {Pzz_minus}", "SPSGPS", st.Severity.Info)
    # st.OnScreenLogMessage(f"K = {K}", "SPSGPS", st.Severity.Info)

    z_new = np.zeros(numMeasurements)
    for ee in range(len(cameraEntities)):
        r_SPS_true = 1e-3 * cameraTruePositions[ee]
        r_SPS_est = 1e-3 * cameraTruePositions[ee]
        # r_SPS_est = 1e-3 * estimatedPositions[ee]

        # Synthetic measurements
        r_sc_true = 1e-3 * x_sc_true[0:3]
        v_sc_true = 1e-3 * x_sc_true[3:6]
        r_sc_SPS = r_sc_true - r_SPS_true

        # TODO: Hardcoding covariances for now! Watch out...
        rho_synth = np.linalg.norm(r_sc_SPS)
        rho_dot_synth = np.dot(v_sc_true, r_sc_SPS / rho_synth) + rng.normal(0.0, 0.0005, 1).item()
        alpha_synth = np.arctan2(r_sc_SPS[1], r_sc_SPS[0]) + rng.normal(0.0, 0.001, 1).item()
        dec_synth = np.arccos(r_sc_SPS[2] / rho_synth) + rng.normal(0.0, 0.001, 1).item()
        rho_synth += rng.normal(0.0, 0.005, 1).item()

        posIndex: int = 7 * ee
        z_new[posIndex:posIndex + 3] = r_SPS_est
        z_new[posIndex + 3] = rho_synth
        z_new[posIndex + 4] = rho_dot_synth
        z_new[posIndex + 5] = alpha_synth
        z_new[posIndex + 6] = dec_synth

    z_history.append(copy.deepcopy(z_new))

    if True:
    # if (np.abs(z_new - mz_minus) < 6.0 * np.sqrt(np.diag(Pzz_minus))).all():
        mx_plus = mx_minus + gamma_underweight * K @ (z_new - mz_minus)
        Pxx_plus = Pxx_minus - Pxz_minus @ K.T - K @ Pxz_minus.T + K @ Pzz_minus @ K.T
    else:
        st.OnScreenLogMessage(f"Measurement at index {j} not processed; exceed 6-sigma distance to mean.", "SPSGPS", st.Severity.Info)
        mx_plus = copy.deepcopy(mx_minus)
        Pxx_plus = copy.deepcopy(Pxx_minus)

    # Force all S[l,0] terms to be zero (necessary for spherical harmonics)
    # TODO: HARDCODED INDICES AT THE MOMENT
    mx_plus[31] = 0.0
    mx_plus[34] = 0.0
    mx_plus[38] = 0.0

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

sc_true_x = np.array([sc_x[0] for sc_x in sc_true_history])
sc_true_y = np.array([sc_x[1] for sc_x in sc_true_history])
sc_true_z = np.array([sc_x[2] for sc_x in sc_true_history])
sc_true_vx = np.array([sc_x[3] for sc_x in sc_true_history])
sc_true_vy = np.array([sc_x[4] for sc_x in sc_true_history])
sc_true_vz = np.array([sc_x[5] for sc_x in sc_true_history])

mx_x = np.array([1e3 * mx[0] for mx in mx_history])
mx_y = np.array([1e3 * mx[1] for mx in mx_history])
mx_z = np.array([1e3 * mx[2] for mx in mx_history])
mx_vx = np.array([1e3 * mx[3] for mx in mx_history])
mx_vy = np.array([1e3 * mx[4] for mx in mx_history])
mx_vz = np.array([1e3 * mx[5] for mx in mx_history])

mx_SPS1_x = np.array([1e3 * mx[6] for mx in mx_history])
mx_SPS1_y = np.array([1e3 * mx[7] for mx in mx_history])
mx_SPS1_z = np.array([1e3 * mx[8] for mx in mx_history])

mx_SPS2_x = np.array([1e3 * mx[9] for mx in mx_history])
mx_SPS2_y = np.array([1e3 * mx[10] for mx in mx_history])
mx_SPS2_z = np.array([1e3 * mx[11] for mx in mx_history])

mx_SPS3_x = np.array([1e3 * mx[12] for mx in mx_history])
mx_SPS3_y = np.array([1e3 * mx[13] for mx in mx_history])
mx_SPS3_z = np.array([1e3 * mx[14] for mx in mx_history])

mx_SPS4_x = np.array([1e3 * mx[15] for mx in mx_history])
mx_SPS4_y = np.array([1e3 * mx[16] for mx in mx_history])
mx_SPS4_z = np.array([1e3 * mx[17] for mx in mx_history])

mx_mu = np.array([mx[18] for mx in mx_history])
mx_C20 = np.array([mx[19] for mx in mx_history])
mx_C30 = np.array([mx[22] for mx in mx_history])
mx_C40 = np.array([mx[26] for mx in mx_history])
mx_S22 = np.array([mx[33] for mx in mx_history])
mx_S33 = np.array([mx[37] for mx in mx_history])

Pxx_x = np.array([1e6 * Pxx[0, 0] for Pxx in Pxx_history])
Pxx_y = np.array([1e6 * Pxx[1, 1] for Pxx in Pxx_history])
Pxx_z = np.array([1e6 * Pxx[2, 2] for Pxx in Pxx_history])
Pxx_vx = np.array([1e6 * Pxx[3, 3] for Pxx in Pxx_history])
Pxx_vy = np.array([1e6 * Pxx[4, 4] for Pxx in Pxx_history])
Pxx_vz = np.array([1e6 * Pxx[5, 5] for Pxx in Pxx_history])

Pxx_SPS1_x = np.array([1e3 * Pxx[6, 6] for Pxx in Pxx_history])
Pxx_SPS1_y = np.array([1e3 * Pxx[7, 7] for Pxx in Pxx_history])
Pxx_SPS1_z = np.array([1e3 * Pxx[8, 8] for Pxx in Pxx_history])

Pxx_SPS2_x = np.array([1e3 * Pxx[9, 9] for Pxx in Pxx_history])
Pxx_SPS2_y = np.array([1e3 * Pxx[10, 10] for Pxx in Pxx_history])
Pxx_SPS2_z = np.array([1e3 * Pxx[11, 11] for Pxx in Pxx_history])

Pxx_SPS3_x = np.array([1e3 * Pxx[12, 12] for Pxx in Pxx_history])
Pxx_SPS3_y = np.array([1e3 * Pxx[13, 13] for Pxx in Pxx_history])
Pxx_SPS3_z = np.array([1e3 * Pxx[14, 14] for Pxx in Pxx_history])

Pxx_SPS4_x = np.array([1e3 * Pxx[15, 15] for Pxx in Pxx_history])
Pxx_SPS4_y = np.array([1e3 * Pxx[16, 16] for Pxx in Pxx_history])
Pxx_SPS4_z = np.array([1e3 * Pxx[17, 17] for Pxx in Pxx_history])

Pxx_mu = np.array([Pxx[18, 18] for Pxx in Pxx_history])
Pxx_C20 = np.array([Pxx[19, 19] for Pxx in Pxx_history])
Pxx_C30 = np.array([Pxx[22, 22] for Pxx in Pxx_history])
Pxx_C40 = np.array([Pxx[26, 26] for Pxx in Pxx_history])
Pxx_S22 = np.array([Pxx[33, 33] for Pxx in Pxx_history])
Pxx_S33 = np.array([Pxx[37, 37] for Pxx in Pxx_history])

# st.OnScreenLogMessage(f"sc_true_x = {sc_true_x}", "SPSGPS", st.Severity.Info)
# st.OnScreenLogMessage(f"mx_x      = {mx_x}", "SPSGPS", st.Severity.Info)

fig1 = plt.figure(layout='constrained')
ax1_1 = fig1.add_subplot(231)
ax1_2 = fig1.add_subplot(232)
ax1_3 = fig1.add_subplot(233)
ax1_4 = fig1.add_subplot(234)
ax1_5 = fig1.add_subplot(235)
ax1_6 = fig1.add_subplot(236)

# Position
ax1_1.plot(times, mx_x - sc_true_x, label=r"$x$-Axis Position", color='blue')
ax1_1.plot(times, -3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r', label=r"$x$-Axis Position 3$\sigma$ Interval")
ax1_1.plot(times, 3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r')
ax1_1.set_xlabel("Time (s)")
ax1_1.set_ylabel("Position error (m)")
ax1_1.set_title(r"Error in $x$-Axis Position over Time")
ax1_1.grid()
ax1_1.legend()

ax1_2.plot(times, mx_y - sc_true_y, label=r"$y$-Axis Position", color='blue')
ax1_2.plot(times, -3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r', label=r"$y$-Axis Position 3$\sigma$ Interval")
ax1_2.plot(times, 3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r')
ax1_2.set_xlabel("Time (s)")
ax1_2.set_ylabel("Position error (m)")
ax1_2.set_title(r"Error in $y$-Axis Position over Time")
ax1_2.grid()
ax1_2.legend()

ax1_3.plot(times, mx_z - sc_true_z, label=r"$z$-Axis Position", color='blue')
ax1_3.plot(times, -3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r', label=r"$z$-Axis Position 3$\sigma$ Interval")
ax1_3.plot(times, 3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r')
ax1_3.set_xlabel("Time (s)")
ax1_3.set_ylabel("Position error (m)")
ax1_3.set_title(r"Error in $z$-Axis Position over Time")
ax1_3.grid()
ax1_3.legend()

# Velocity
ax1_4.plot(times, mx_vx - sc_true_vx, label=r"$x$-Axis Velocity", color='blue')
ax1_4.plot(times, -3.0 * np.sqrt(Pxx_vx), linestyle='dashed', color='r', label=r"$x$-Axis Velocity 3$\sigma$ Interval")
ax1_4.plot(times, 3.0 * np.sqrt(Pxx_vx), linestyle='dashed', color='r')
ax1_4.set_xlabel("Time (s)")
ax1_4.set_ylabel("Velocity error (m/s)")
ax1_4.set_title(r"Error in $x$-Axis Velocity over Time")
ax1_4.grid()
ax1_4.legend()

ax1_5.plot(times, mx_vy - sc_true_vy, label=r"$y$-Axis Velocity", color='blue')
ax1_5.plot(times, -3.0 * np.sqrt(Pxx_vy), linestyle='dashed', color='r', label=r"$y$-Axis Velocity 3$\sigma$ Interval")
ax1_5.plot(times, 3.0 * np.sqrt(Pxx_vy), linestyle='dashed', color='r')
ax1_5.set_xlabel("Time (s)")
ax1_5.set_ylabel("Velocity error (m/s)")
ax1_5.set_title(r"Error in $y$-Axis Velocity over Time")
ax1_5.grid()
ax1_5.legend()

ax1_6.plot(times, mx_vz - sc_true_vz, label=r"$z$-Axis Velocity", color='blue')
ax1_6.plot(times, -3.0 * np.sqrt(Pxx_vz), linestyle='dashed', color='r', label=r"$z$-Axis Velocity 3$\sigma$ Interval")
ax1_6.plot(times, 3.0 * np.sqrt(Pxx_vz), linestyle='dashed', color='r')
ax1_6.set_xlabel("Time (s)")
ax1_6.set_ylabel("Velocity error (m/s)")
ax1_6.set_title(r"Error in $z$-Axis Velocity over Time")
ax1_6.grid()
ax1_6.legend()

# Gravitational parameter and select spherical harmonic coefficients
fig2 = plt.figure(layout='constrained')
ax2_1 = fig2.add_subplot(231)
ax2_2 = fig2.add_subplot(232)
ax2_3 = fig2.add_subplot(233)
ax2_4 = fig2.add_subplot(234)
ax2_5 = fig2.add_subplot(235)
ax2_6 = fig2.add_subplot(236)

# Gravitational parameter
ax2_1.plot(times, mx_mu - 1e-9 * gravModel_true.mu, label=r"$m_{x,\mu} - \mu_{true}$", color='blue')
ax2_1.plot(times, -3.0 * np.sqrt(Pxx_mu), linestyle='dashed', color='r', label=r"$P_{xx,\mu}$")
ax2_1.plot(times, 3.0 * np.sqrt(Pxx_mu), linestyle='dashed', color='r')
ax2_1.set_xlabel("Time (s)")
ax2_1.set_ylabel("Gravitational parameter error (km^3/s^2)")
ax2_1.set_title(r"Error in $m_{x,\mu}$ over Time")
ax2_1.grid()
ax2_1.legend()

# Spherical harmonic coefficients
# C[2,0]
ax2_2.plot(times, mx_C20 - gravModel_true.Clm[2, 0], label=r"$m_{x,C20} - C_{true}[2,0]$", color='blue')
ax2_2.plot(times, -3.0 * np.sqrt(Pxx_C20), linestyle='dashed', color='r', label=r"$P_{xx,C20}$")
ax2_2.plot(times, 3.0 * np.sqrt(Pxx_C20), linestyle='dashed', color='r')
ax2_2.set_xlabel("Time (s)")
ax2_2.set_ylabel("C[2,0] (normalized, non-dimensional)")
ax2_2.set_title(r"Error in $m_{x,C20}$ over Time")
ax2_2.grid()
ax2_2.legend()

# C[3,0]
ax2_3.plot(times, mx_C30 - gravModel_true.Clm[3, 0], label=r"$m_{x,C30} - C_{true}[3,0]$", color='blue')
ax2_3.plot(times, -3.0 * np.sqrt(Pxx_C30), linestyle='dashed', color='r', label=r"$P_{xx,C30}$")
ax2_3.plot(times, 3.0 * np.sqrt(Pxx_C30), linestyle='dashed', color='r')
ax2_3.set_xlabel("Time (s)")
ax2_3.set_ylabel("C[3,0] (normalized, non-dimensional)")
ax2_3.set_title(r"Error in $m_{x,C30}$ over Time")
ax2_3.grid()
ax2_3.legend()

# C[4,0]
ax2_4.plot(times, mx_C40 - gravModel_true.Clm[4, 0], label=r"$m_{x,C40} - C_{true}[4,0]$", color='blue')
ax2_4.plot(times, -3.0 * np.sqrt(Pxx_C40), linestyle='dashed', color='r', label=r"$P_{xx,C40}$")
ax2_4.plot(times, 3.0 * np.sqrt(Pxx_C40), linestyle='dashed', color='r')
ax2_4.set_xlabel("Time (s)")
ax2_4.set_ylabel("C[4,0] (normalized, non-dimensional)")
ax2_4.set_title(r"Error in $m_{x,C40}$ over Time")
ax2_4.grid()
ax2_4.legend()

# S[2,2]
ax2_5.plot(times, mx_S22 - gravModel_true.Slm[2, 2], label=r"$m_{x,S22} - S_{true}[2,2]$", color='blue')
ax2_5.plot(times, -3.0 * np.sqrt(Pxx_S22), linestyle='dashed', color='r', label=r"$P_{xx,S22}$")
ax2_5.plot(times, 3.0 * np.sqrt(Pxx_S22), linestyle='dashed', color='r')
ax2_5.set_xlabel("Time (s)")
ax2_5.set_ylabel("S[2,2] (normalized, non-dimensional)")
ax2_5.set_title(r"Error in $m_{x,S22}$ over Time")
ax2_5.grid()
ax2_5.legend()

# S[3,3]
ax2_6.plot(times, mx_S33 - gravModel_true.Slm[3, 3], label=r"$m_{x,S33} - S_{true}[3,3]$", color='blue')
ax2_6.plot(times, -3.0 * np.sqrt(Pxx_S33), linestyle='dashed', color='r', label=r"$P_{xx,S33}$")
ax2_6.plot(times, 3.0 * np.sqrt(Pxx_S33), linestyle='dashed', color='r')
ax2_6.set_xlabel("Time (s)")
ax2_6.set_ylabel("S[3,3] (normalized, non-dimensional)")
ax2_6.set_title(r"Error in $m_{x,S33}$ over Time")
ax2_6.grid()
ax2_6.legend()

fig3 = plt.figure(layout='constrained')
ax3_1 = fig3.add_subplot(111, projection='3d')

ax3_1.plot(sc_true_x, sc_true_y, sc_true_z, color='green', label='True Trajectory')
ax3_1.plot(mx_x, mx_y, mx_z, color='red', label='Estimated Trajectory')
ax3_1.set_xlabel("X-Axis Position (m)")
ax3_1.set_ylabel("Y-Axis Position (m)")
ax3_1.set_zlabel("Z-Axis Position (m)")
ax3_1.set_title("3D Trajectory")
ax3_1.grid()
ax3_1.legend()


fig4 = plt.figure(layout='constrained')
ax4_1 = fig4.add_subplot(4, 3, 1)
ax4_2 = fig4.add_subplot(4, 3, 2)
ax4_3 = fig4.add_subplot(4, 3, 3)

ax4_4 = fig4.add_subplot(4, 3, 4)
ax4_5 = fig4.add_subplot(4, 3, 5)
ax4_6 = fig4.add_subplot(4, 3, 6)

ax4_7 = fig4.add_subplot(4, 3, 7)
ax4_8 = fig4.add_subplot(4, 3, 8)
ax4_9 = fig4.add_subplot(4, 3, 9)

ax4_10 = fig4.add_subplot(4, 3, 10)
ax4_11 = fig4.add_subplot(4, 3, 11)
ax4_12 = fig4.add_subplot(4, 3, 12)

# SPS 1
ax4_1.plot(times, mx_SPS1_x - cameraTruePositions[0][0], label=r"$x$-Position Error", color='blue')
ax4_1.plot(times, -3.0 * np.sqrt(Pxx_SPS1_x), linestyle='dashed', color='r', label=r"$x$-Position 3$\sigma$ Intervals")
ax4_1.plot(times, 3.0 * np.sqrt(Pxx_SPS1_x), linestyle='dashed', color='r')
ax4_1.set_xlabel("Time (s)")
ax4_1.set_ylabel("Position error (m)")
ax4_1.set_title(r"SPS1 Position Error in $x$-Axis over Time")
ax4_1.grid()
ax4_1.legend()

ax4_2.plot(times, mx_SPS1_y - cameraTruePositions[0][1], label=r"$y$-Position Error", color='blue')
ax4_2.plot(times, -3.0 * np.sqrt(Pxx_SPS1_y), linestyle='dashed', color='r', label=r"$y$-Position 3$\sigma$ Intervals")
ax4_2.plot(times, 3.0 * np.sqrt(Pxx_SPS1_y), linestyle='dashed', color='r')
ax4_2.set_xlabel("Time (s)")
ax4_2.set_ylabel("Position error (m)")
ax4_2.set_title(r"SPS1 Position Error in $y$-Axis over Time")
ax4_2.grid()
ax4_2.legend()

ax4_3.plot(times, mx_SPS1_z - cameraTruePositions[0][2], label=r"$z$-Position Error", color='blue')
ax4_3.plot(times, -3.0 * np.sqrt(Pxx_SPS1_z), linestyle='dashed', color='r', label=r"$z$-Position 3$\sigma$ Intervals")
ax4_3.plot(times, 3.0 * np.sqrt(Pxx_SPS1_z), linestyle='dashed', color='r')
ax4_3.set_xlabel("Time (s)")
ax4_3.set_ylabel("Position error (m)")
ax4_3.set_title(r"SPS1 Position Error in $z$-Axis over Time")
ax4_3.grid()
ax4_3.legend()

# SPS 2
ax4_4.plot(times, mx_SPS2_x - cameraTruePositions[1][0], label=r"$x$-Position Error", color='blue')
ax4_4.plot(times, -3.0 * np.sqrt(Pxx_SPS2_x), linestyle='dashed', color='r', label=r"$x$-Position 3$\sigma$ Intervals")
ax4_4.plot(times, 3.0 * np.sqrt(Pxx_SPS2_x), linestyle='dashed', color='r')
ax4_4.set_xlabel("Time (s)")
ax4_4.set_ylabel("Position error (m)")
ax4_4.set_title(r"SPS2 Position Error in $x$-Axis over Time")
ax4_4.grid()
ax4_4.legend()

ax4_5.plot(times, mx_SPS2_y - cameraTruePositions[1][1], label=r"$y$-Position Error", color='blue')
ax4_5.plot(times, -3.0 * np.sqrt(Pxx_SPS2_y), linestyle='dashed', color='r', label=r"$y$-Position 3$\sigma$ Intervals")
ax4_5.plot(times, 3.0 * np.sqrt(Pxx_SPS2_y), linestyle='dashed', color='r')
ax4_5.set_xlabel("Time (s)")
ax4_5.set_ylabel("Position error (m)")
ax4_5.set_title(r"SPS2 Position Error in $y$-Axis over Time")
ax4_5.grid()
ax4_5.legend()

ax4_6.plot(times, mx_SPS2_z - cameraTruePositions[1][2], label=r"$z$-Position Error", color='blue')
ax4_6.plot(times, -3.0 * np.sqrt(Pxx_SPS2_z), linestyle='dashed', color='r', label=r"$z$-Position 3$\sigma$ Intervals")
ax4_6.plot(times, 3.0 * np.sqrt(Pxx_SPS2_z), linestyle='dashed', color='r')
ax4_6.set_xlabel("Time (s)")
ax4_6.set_ylabel("Position error (m)")
ax4_6.set_title(r"SPS2 Position Error in $z$-Axis over Time")
ax4_6.grid()
ax4_6.legend()

# SPS 3
ax4_7.plot(times, mx_SPS3_x - cameraTruePositions[2][0], label=r"$x$-Position Error", color='blue')
ax4_7.plot(times, -3.0 * np.sqrt(Pxx_SPS3_x), linestyle='dashed', color='r', label=r"$x$-Position 3$\sigma$ Intervals")
ax4_7.plot(times, 3.0 * np.sqrt(Pxx_SPS3_x), linestyle='dashed', color='r')
ax4_7.set_xlabel("Time (s)")
ax4_7.set_ylabel("Position error (m)")
ax4_7.set_title(r"SPS3 Position Error in $x$-Axis over Time")
ax4_7.grid()
ax4_7.legend()

ax4_8.plot(times, mx_SPS3_y - cameraTruePositions[2][1], label=r"$y$-Position Error", color='blue')
ax4_8.plot(times, -3.0 * np.sqrt(Pxx_SPS3_y), linestyle='dashed', color='r', label=r"$y$-Position 3$\sigma$ Intervals")
ax4_8.plot(times, 3.0 * np.sqrt(Pxx_SPS3_y), linestyle='dashed', color='r')
ax4_8.set_xlabel("Time (s)")
ax4_8.set_ylabel("Position error (m)")
ax4_8.set_title(r"SPS3 Position Error in $y$-Axis over Time")
ax4_8.grid()
ax4_8.legend()

ax4_9.plot(times, mx_SPS3_z - cameraTruePositions[2][2], label=r"$z$-Position Error", color='blue')
ax4_9.plot(times, -3.0 * np.sqrt(Pxx_SPS3_z), linestyle='dashed', color='r', label=r"$z$-Position 3$\sigma$ Intervals")
ax4_9.plot(times, 3.0 * np.sqrt(Pxx_SPS3_z), linestyle='dashed', color='r')
ax4_9.set_xlabel("Time (s)")
ax4_9.set_ylabel("Position error (m)")
ax4_9.set_title(r"SPS3 Position Error in $z$-Axis over Time")
ax4_9.grid()
ax4_9.legend()

# SPS 2
ax4_10.plot(times, mx_SPS4_x - cameraTruePositions[3][0], label=r"$x$-Position Error", color='blue')
ax4_10.plot(times, -3.0 * np.sqrt(Pxx_SPS4_x), linestyle='dashed', color='r', label=r"$x$-Position 3$\sigma$ Intervals")
ax4_10.plot(times, 3.0 * np.sqrt(Pxx_SPS4_x), linestyle='dashed', color='r')
ax4_10.set_xlabel("Time (s)")
ax4_10.set_ylabel("Position error (m)")
ax4_10.set_title(r"SPS4 Position Error in $x$-Axis over Time")
ax4_10.grid()
ax4_10.legend()

ax4_11.plot(times, mx_SPS4_y - cameraTruePositions[3][1], label=r"$y$-Position Error", color='blue')
ax4_11.plot(times, -3.0 * np.sqrt(Pxx_SPS4_y), linestyle='dashed', color='r', label=r"$y$-Position 3$\sigma$ Intervals")
ax4_11.plot(times, 3.0 * np.sqrt(Pxx_SPS4_y), linestyle='dashed', color='r')
ax4_11.set_xlabel("Time (s)")
ax4_11.set_ylabel("Position error (m)")
ax4_11.set_title(r"SPS4 Position Error in $y$-Axis over Time")
ax4_11.grid()
ax4_11.legend()

ax4_12.plot(times, mx_SPS4_z - cameraTruePositions[3][2], label=r"$z$-Position Error", color='blue')
ax4_12.plot(times, -3.0 * np.sqrt(Pxx_SPS4_z), linestyle='dashed', color='r', label=r"$z$-Position 3$\sigma$ Intervals")
ax4_12.plot(times, 3.0 * np.sqrt(Pxx_SPS4_z), linestyle='dashed', color='r')
ax4_12.set_xlabel("Time (s)")
ax4_12.set_ylabel("Position error (m)")
ax4_12.set_title(r"SPS4 Position Error in $z$-Axis over Time")
ax4_12.grid()
ax4_12.legend()


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


set_axes_equal(ax3_1)

plt.show()

st.leave_sim()
