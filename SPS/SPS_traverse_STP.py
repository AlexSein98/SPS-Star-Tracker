import os, sys, time, datetime, traceback
import spaceteams as st
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

from SPS.global_config import globalConfig
from py_src.star.python.transformations import latlon_to_T, r_to_latlonalt, r_hat_to_ra_dec, normalize
from py_src.star.python.catalog import ra_dec_to_rot
from SPS.SPS_sigma_points import read_csv, write_csv


os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

##################################
####    PLANET DATA IMPORT    ####
##################################

planetData = st.ProcPlanet.DataStore()

moonGlobalData = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, "Core/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014")
args = st.ProcPlanet.GeoBin_Extra_Args()
args.cubicInterp = True

planetData.AddGeoBinAltimetryLayer(1.0, moonGlobalData, args)


# Wait for Eridani to load
st.GetThisSystem().AddOrSetParam(st.VarType.bool, "Ready", False)

def SetIsReady(paramMap: st.ParamMap, timeNow: st.timestamp):
    st.GetThisSystem().SetParam(st.VarType.bool, "Ready", True)

st.SimGlobals.Subscribe("EridaniLoadingComplete", SetIsReady)

while not st.GetThisSystem().GetParam(st.VarType.bool, "Ready"):
    pass

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

regenerateStarCatalog: bool     = True
delete_old: bool                = True
reprocess_star_tracker: bool    = True
doCalibration: bool             = True

calibrationCutoff: int = 300
endpoint: int = calibrationCutoff + 200

numImages: int = calibrationCutoff + 200

##############################
####    SITE SELECTION    ####
##############################

# site: str = "Ideal"
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

planetName: str = "Moon"

truthDataPath = globalConfig.outputDir + "truth_data_" + planetName + ".csv"
attitudeEstDataPath = globalConfig.outputDir + "attitudes_" + planetName + ".csv"
gravTruthDataPath = globalConfig.outputDir + "true_gravities_" + planetName + ".csv"
gravEstDataPath = globalConfig.outputDir + "measurements_" + planetName + ".csv"
q_i_b_DataPath = globalConfig.outputDir + "q_i_b_" + planetName + ".csv"

#############################
####    ERROR SOURCES    ####
#############################

addMeasurementBias: bool = globalConfig.addMeasurementBias
addMeasurementNoise: bool = globalConfig.addMeasurementNoise

sigma_2 = np.array([[1e-8, 0.0, 0.0],
                    [0.0, 1e-8, 0.0],
                    [0.0, 0.0, 1e-8]])  # Worst case for the BMA220 IMU (once averaged out)
biasSigma_2 = np.array([[0.15 ** 2, 0.0, 0.0],
                        [0.0, 0.15 ** 2, 0.0],
                        [0.0, 0.0, 0.15 ** 2]])
bias = np.linalg.cholesky(biasSigma_2) @ np.random.randn(3)

############################
####    RENDER SETUP    ####
############################

# Delete all old images
if delete_old:
    if os.path.exists(globalConfig.renderDir):
        shutil.rmtree(globalConfig.renderDir)
    os.mkdir(globalConfig.renderDir)

# Time
tNow = globalConfig.tNow
tNow_datetime = st.timestamp.from_datetime(datetime.datetime(year=2026, month=5, day=22, hour=16))
calibrationTimeStep_s: float = 1.0
traverseTimeStep_s: float = 100.0

true_data: list[npt.NDArray] = []
true_accelerations: list[npt.NDArray] = []
measured_accelerations: list[npt.NDArray] = []
q_i_b_data: list[npt.NDArray] = []

times_calibration = calibrationTimeStep_s * np.linspace(0.0, calibrationCutoff - 1, calibrationCutoff)
times_traverse = times_calibration[-1] + traverseTimeStep_s + traverseTimeStep_s * np.linspace(
    0.0, endpoint - calibrationCutoff - 1, endpoint - calibrationCutoff)
times = np.concat((times_calibration, times_traverse))

print(f"Times = {np.round(times[calibrationCutoff - 5:calibrationCutoff + 5], 1)}")

gravModel = globalConfig.planet.gravModel
radiusEquatorial: float = gravModel.radius
radiusPolar: float = gravModel.polarRadius
Omega = np.array([0.0, 0.0, gravModel.omega])  # Expressed in the planet-fixed frame

startTime = time.perf_counter()
elapsedSeconds: float = 0.0
printInterval: int = 10

# TODO: ellipsoid for non-Moon testing
cameraPosPlanetFixed = st.PlanetUtils.LLA_to_PCPF(st.PlanetUtils.LatLonAlt(np.deg2rad(phi_pg_0), np.deg2rad(lon_pg_0), h_ellp_0), radiusEquatorial)
cameraPosPlanetFixed, _ = st.ProcPlanet.SampleGround(planetData, cameraPosPlanetFixed, radiusEquatorial, 0.0, 20)
lat_pc, lon_pc, h_pc = st.PlanetUtils.PCPF_to_LLA(cameraPosPlanetFixed, radiusEquatorial)

###################################
####    TRAVERSE PARAMETERS    ####
###################################

# Random number generator
rng = np.random.default_rng(int_seed)

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
    if i >= calibrationCutoff:
        pos_i += traverseDirection * traverseStep_m + rng.normal(0.0, traverseFollowingError_m_1sigma, size=3)
        pos_i, _ = st.ProcPlanet.SampleGround(planetData, pos_i, radiusEquatorial, 0.0, 20)


#######################################
####    REGENERATE STAR CATALOG    ####
#######################################

J2000: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "J2000Frame")
planetEntity: st.Entity = st.GetThisSystem().GetParam(st.VarType.entityRef, "Planet")

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
    planetLoc = planetEntity.getLocation().WRT_ExprIn(J2000.GetBodyFixedFrame())

    ground.create_star_catalog(starcat_file=starcat_file, brightness_thresh=b_thresh,
                               excess_rows=excess_rows, index_col=index_col, fov=fov,
                               save_vals=save_vals, rB=planetLoc, save_dir=save_dir, t=simTimeAstropy)

#############################
####    RENDER IMAGES    ####
#############################

if not os.path.exists(globalConfig.renderDir) or is_dir_empty(globalConfig.renderDir):
    for i in range(numImages):
        doPrint: bool = i % printInterval == 0

        # Step time forward by the correct dt
        tNow: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()

        if i < calibrationCutoff:
            tNow += datetime.timedelta(seconds=calibrationTimeStep_s)
        else:
            tNow += datetime.timedelta(seconds=traverseTimeStep_s)

        st.SimGlobals.SimClock.ResetTo(st.timestamp.from_datetime(tNow))
        time.sleep(0.1)

        # Sample gravity vector
        positionNow = positions[i]
        planetFixedFrame = planetEntity.GetBodyFixedFrame()
        stateNow = st.frames.FramedLocVelAcc(st.frames.rva_struct(positionNow, np.zeros(3), np.zeros(3)), planetFixedFrame)
        g = st.SimGlobals.SampleVectorField("Gravity", stateNow).ExprIn(planetFixedFrame)    
        g -= np.cross(Omega, np.cross(Omega, positionNow))  # Handle being on the surface of the planet
        g_true = copy.deepcopy(g)
        
        lat_pc, lon_pc, h_pc = r_to_latlonalt(positionNow, radiusEquatorial)
        T_P_G = latlon_to_T(lat_pc, lon_pc).T
        g_IMU_frame = (T_P_G @ np.array([g]).T).T[0]

        if addMeasurementBias:
            g_IMU_frame += bias

        if addMeasurementNoise:
            g_IMU_frame += np.linalg.cholesky(sigma_2) @ np.random.randn(3)
        
        measured_accelerations.append(g_IMU_frame)

        planetRot = planetEntity.getRotation().DCM_WRT(J2000.GetBodyFixedFrame())
        q_i_b_data.append(st.math.DCM_to_Quat(planetRot))

        gInertial_true = (planetRot.T @ np.array([g_true]).T).T[0]
        true_accelerations.append(gInertial_true)

        gInertial = (planetRot.T @ T_P_G.T @ np.array([g_IMU_frame]).T).T[0]
        ra_true, de_true = r_hat_to_ra_dec(-normalize(gInertial_true))
        ra, de = r_hat_to_ra_dec(-normalize(gInertial))

        # Render
        EridaniRenderPayload = st.ParamMap()
        EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Loc", positionNow)
        EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Vel", np.zeros(3))
        EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Rot", ra_dec_to_rot(ra, de))
        EridaniRenderPayload.AddParam(st.VarType.entityRef, "Frame", planetEntity)
        EridaniRenderPayload.AddParam(st.VarType.string, "NameOverride", "SPSRender" + str(i).zfill(5))
        payload = st.SimGlobals.Request("EridaniRenderImage", EridaniRenderPayload, timeout=datetime.timedelta(seconds=60.0))

        if doPrint:
            print(f'Rendering image {i} of {numImages} ({round(100.0 * float(i) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}')

        true_data.append(st.math.DCM_to_Quat(ra_dec_to_rot(ra_true, de_true)))

        endTime = time.perf_counter()
        elapsedSeconds = endTime - startTime
        elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

        if doPrint:
            projectedRemainingSeconds: float = elapsedSeconds * float(numImages - i + 1) / float(i + 1)
            projectedRemainingTime = datetime.timedelta(seconds=round(projectedRemainingSeconds))
            print(f'Elapsed time: {elapsedTime}. Remaining time estimate: {projectedRemainingTime}\n')
        
    write_csv(truthDataPath, true_data)
    write_csv(gravTruthDataPath, true_accelerations)
    write_csv(gravEstDataPath, measured_accelerations)
    write_csv(q_i_b_DataPath, q_i_b_data)

else:
    print("Render directory not empty; skipping render step...\n")


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
    np.set_printoptions(suppress=True)

    imgSourceDir = globalConfig.renderDir

    data_path = './data' # full path to your data
    cam_config_file_path = './data/cam_config/Custom_cam.json' # full path (including filename) of your cam config file
    darkframe_file_path = './Images/darkframes/darkframe.png' # full path (including filename) of your darkframe file
    image_extension = ".png" # the image extension to search for in the data_path directory
    cat_prefix ='' # if the catalog has a prefix, define it here

    #################################
    ####    Support Functions    ####
    #################################

    print(f'imgSourceDir = {imgSourceDir}')

    ###################################
    ####    Process Star Images    ####
    ###################################

    # Load star tracker and catalog data
    if darkframe_file_path == '': darkframe_file_path = None
    if darkframe_file_path is not None:
        if not os.path.exists(darkframe_file_path):
            darkframe_file_path = None
            print("unable to find provided darkframe file, proceeding without one...")
        else:    print("darkframe file: " + darkframe_file_path)
    else:    print("no darkframe file provided, proceeding without one...")

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

    dir_contents = os.listdir(imgSourceDir)
    for i in range(len(dir_contents)):
        dir_contents[i] = imgSourceDir + "/" + dir_contents[i]
    dir_contents.sort()

    image_names = []

    for item in dir_contents:
        if image_extension in item:
            image_names+=[os.path.abspath(item)]

    idx: int = 0
    for image_filename in image_names:
        image_name += [image_filename]
        # print("===================================================")
        # print(image_filename)

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
                print('est q: ' + str(q_est)+'\n')
            q_rotate = np.array([0.5, -0.5, 0.5, 0.5])  # w-last quaternion
            q_est = quat_mult(q_est, q_rotate)  # w-last quaternion
            qs += [q_est[3]]
            qv0 += [q_est[0]]
            qv1 += [q_est[1]]
            qv2 += [q_est[2]]
        except AssertionError:
            if VERBOSE:
                print('NO VALID STARS FOUND\n')
            qs += [999]
            qv0 += [999]
            qv1 += [999]
            qv2 += [999]

        ttime += [time.time()]
        sram  += [psutil.virtual_memory().percent]
        #scpu  += [psutil.cpu_percent(2)]
        scpu  += [psutil.cpu_percent()]

        print(f'Completed image {idx} ({round(float(idx) / float(len(image_names)) * 100.0, 2)} %)')
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

    print("\n\n took " + str(time.time()-total_start) + " seconds to complete \n\n")
    print("data saved to: " + attitudeEstDataPath)
else:
    print("Quaternion measurements already processed; skipping processing step...\n")

# Get data from files
truthData = read_csv(truthDataPath)
attitudeEstData = read_csv(attitudeEstDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
gravTruthData = read_csv(gravTruthDataPath)
gravEstData = read_csv(gravEstDataPath)
q_i_b_list = read_csv(q_i_b_DataPath)

# Very basic error handling if datasets are not the same length
if not (len(truthData) == len(attitudeEstData) == len(gravEstData)):
    print(f'Warning: early exit due to dataset length mismatch; truthData length = {len(truthData)}, attitudeEstData length = {len(attitudeEstData)}, and gravEstData length = {len(gravEstData)}.')
    exit(0)

# Initialize all inertial-to-planet attitude matrices
T_i_b_list: list[npt.NDArray] = []
T_i_c_list: list[npt.NDArray] = []
g_est_list: list[npt.NDArray] = []
for i in range(len(times)):
    T_i_b_list.append(st.math.Quat_to_DCM(q_i_b_list[i]))
    q_i_c = np.array([attitudeEstData[i][1], attitudeEstData[i][2], attitudeEstData[i][3], attitudeEstData[i][0]])
    T_i_c_list.append(st.math.Quat_to_DCM(q_i_c))
    g_est_list.append(gravEstData[i])

###############################
####    SPS CALIBRATION    ####
###############################

eps: float = 1.0e-4
T_calibration = np.identity(3)
if doCalibration:
    phi_pc, lon_pc, _ = r_to_latlonalt(cameraPosPlanetFixed, radiusEquatorial)

    def SampleTrueGravity(pos_SPS_PCPF: npt.NDArray, j: float, _gravTruthData: list[npt.NDArray]) -> npt.NDArray:
        return _gravTruthData[j]

    SampleTrueGravity_Wrapped = lambda pos, j : SampleTrueGravity(pos, j, gravTruthData)
    T_calibration = st.ProcPlanet.SPS.CalculateAlignment(calibrationCutoff, cameraPosPlanetFixed, 
        T_i_c_list, T_i_b_list, g_est_list, times, SampleTrueGravity_Wrapped, eps)

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
T_g_c = T_calibration.T  # transformation from gravity to camera frame

estimatedPositions: list[npt.NDArray] = []

for j in range(len(times[calibrationCutoff:endpoint])):

    ######################################
    ####    Measurement Processing    ####
    ######################################
    
    doPrint: bool = j % printInterval == 0

    T_i_b: npt.NDArray = T_i_b_list[j + calibrationCutoff]
    truth_j = truthData[j + calibrationCutoff]
    attitudeEst_j = attitudeEstData[j + calibrationCutoff]
    gravEst_j = gravEstData[j + calibrationCutoff]
    
    if attitudeEst_j[0] == 999 or attitudeEst_j[1] == 999 or attitudeEst_j[2] == 999 or attitudeEst_j[3] == 999:
        print(f'Warning: skipped measurement at index {j} (invalid quaternion).')
        continue
    
    q_i_c = np.array([attitudeEst_j[1], attitudeEst_j[2], attitudeEst_j[3], attitudeEst_j[0]])
    T_i_c = st.math.Quat_to_DCM(q_i_c)

    Omega = np.array([0.0, 0.0, gravModel.omega])
    g_sensorFrame = np.array([gravEst_j[0], gravEst_j[1], gravEst_j[2]])

    # Coarse estimates
    r_coarse_1 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, 
                                                  globalConfig.planet.gravModel.mu, Omega, planetData, 
                                                  planetFixedFrame, np.zeros(3), 0.0, 20)
    r_coarse_2 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, 
                                                  globalConfig.planet.gravModel.mu, Omega, planetData, 
                                                  planetFixedFrame, r_coarse_1, 0.0, 20)
    
    if doPrint:
        print(f'Sample point {j}:')
        print(f'r_expected = {positions[j]}')
        print(f'r_coarse_1 = {r_coarse_1}')
        print(f'r_coarse_2 = {r_coarse_2}')
    
    fineOutputs = st.ProcPlanet.SPS.FineEstimate(r_coarse_2, T_i_b, T_i_c, T_g_c, g_sensorFrame, radiusEquatorial, 
                                                 Omega, planetData, planetFixedFrame, gradientWalkFactor, tol, 
                                                 doPrint, j + calibrationCutoff, 0.0, 20)

    r_bestEstimate = fineOutputs.pos
    phi_pg = fineOutputs.phi_pg
    lon = fineOutputs.lon
    alt = fineOutputs.alt
    i = fineOutputs.iterations

    estimatedPositions.append(r_bestEstimate)
    
    if doPrint:
        print(f'r_bestEstimate = {r_bestEstimate}')
        print(f'Estimated lat = {round(phi_pg, 6)} deg')
        print(f'Estimated lon = {round(lon, 6)} deg')
        print(f'True lat = {round(latTruth, 6)} deg')
        print(f'True lon = {round(lonTruth, 6)} deg\n')
    
    ##################################
    ####    Filter Propagation    ####
    ##################################

    # TODO: pretty sure we can't just add error here because that breaks the Kalman Filter?
    # mx_minus = mx_plus + traverseDirection * traverseStep_m + rng.normal(0.0, traverseFollowingError_m_1sigma, size=3)
    mx_minus = mx_plus + traverseDirection * traverseStep_m
    mx_minus, _ = st.ProcPlanet.SampleGround(planetData, mx_minus, radiusEquatorial, 0.0, 20)
    Pxx_minus = Pxx_plus + Pww
    
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
    else:
        print(f"Measurement at index {j} not processed; exceed 6-sigma distance to mean.")
        mx_plus = copy.deepcopy(mx_minus)
        Pxx_plus = copy.deepcopy(Pxx_minus)

    mx_plus, _ = st.ProcPlanet.SampleGround(planetData, mx_plus, radiusEquatorial, 0.0, 20)

    z_history.append(r_bestEstimate)
    mx_history.append(mx_plus)
    Pxx_history.append(Pxx_plus)
    
    #####################################
    ####    Clean-up and Printing    ####
    #####################################

    percentComplete = round(100.0 * float(j) / float(len(times[calibrationCutoff:endpoint])), 3)
    
    endTime = time.perf_counter()
    elapsedSeconds = endTime - startTime
    elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

    if doPrint:
        print(f'Elapsed time: {elapsedTime}')
        print("------------------------------------------------------------------------------------------------------\n")

########################
####    PLOTTING    ####
########################

cameraTraversePositions: list[npt.NDArray] = positions[calibrationCutoff:endpoint]

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
ax1.scatter(times[calibrationCutoff:endpoint][::subsample], z_x[::subsample] - camPos_x[::subsample], label=r'$z(0)$', color='purple')
ax1.plot(times[calibrationCutoff:endpoint], mx_x - camPos_x, label=r"$m_{x}(0)$", color='blue')
ax1.plot(times[calibrationCutoff:endpoint], -3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r', label=r"$P_{xx}(0,0)$")
ax1.plot(times[calibrationCutoff:endpoint], 3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Position error (m)")
ax1.set_title(r"Error in $m_{x}(0)$ over Time")
ax1.grid()
ax1.legend()

ax2.scatter(times[calibrationCutoff:endpoint][::subsample], z_y[::subsample] - camPos_y[::subsample], label=r'$z(1)$', color='purple')
ax2.plot(times[calibrationCutoff:endpoint], mx_y - camPos_y, label=r"$m_{x}(1)$", color='blue')
ax2.plot(times[calibrationCutoff:endpoint], -3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r', label=r"$P_{xx}(1,1)$")
ax2.plot(times[calibrationCutoff:endpoint], 3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position error (m)")
ax2.set_title(r"Error in $m_{x}(1)$ over Time")
ax2.grid()
ax2.legend()

ax3.scatter(times[calibrationCutoff:endpoint][::subsample], z_z[::subsample] - camPos_z[::subsample], label=r'$z(2)$', color='purple')
ax3.plot(times[calibrationCutoff:endpoint], mx_z - camPos_z, label=r"$m_{x}(2)$", color='blue')
ax3.plot(times[calibrationCutoff:endpoint], -3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r', label=r"$P_{xx}(2,2)$")
ax3.plot(times[calibrationCutoff:endpoint], 3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r')
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
