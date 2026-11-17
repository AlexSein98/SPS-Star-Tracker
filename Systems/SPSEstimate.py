# THIS COMMENT LINE SHOULD BE THE FIRST LINE OF THE FILE
# DON'T CHANGE ANY OF THE BELOW; NECESSARY FOR JOINING SIMULATION
import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
    exit(1)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)
import numpy as np
# DON'T CHANGE ANY OF THE ABOVE; NECESSARY FOR JOINING SIMULATION
################################################################

##############################
####    LOAD LIBRARIES    ####
##############################

import csv
import psutil
import numpy.typing as npt
from pathlib import Path
import copy
import hashlib
import shutil
from PIL import Image
import astropy.time as astrotime
import matplotlib.pyplot as plt

thisRepo = os.path.join(str(st.path_utils.GetLocalAssetsDir()), "Repos", "SPS-Star-Tracker")
localOutput = str(st.path_utils.GetLocalOutputDir())
sys.path.append(thisRepo)
sys.path.append(localOutput)

from py_src.star_tracker.star_tracker import main
from py_src.star_tracker.star_tracker.cam_matrix import *
from py_src.star_tracker.star_tracker.array_transformations import *

from py_src.star.python.transformations import latlon_to_T, T_to_latlon, r_to_latlonalt, r_hat_to_ra_dec, normalize

np.set_printoptions(suppress=True)

##################################
####    PLANET DATA IMPORT    ####
##################################

this = st.GetThisSystem()

datasets: list[str] = this.GetParamArray(st.VarType.string, "Datasets")
priorities: list[float] = this.GetParamArray(st.VarType.double, "DatasetPriorities")

planetDataForRendering = st.ProcPlanet.DataStore()
for i in range(len(datasets)):
    dataPath = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, datasets[i])
    planetDataForRendering.AddGeoBinAltimetryLayer(priorities[i], dataPath, st.ProcPlanet.GeoBin_Extra_Args())

planetDataForSPS = st.ProcPlanet.DataStore()
planetDataForSPS.AddGeoBinAltimetryLayer(0.0, st.path_utils.AssetPathToReal(
    st.AssetType.PlanetData, "Core/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014"),
    st.ProcPlanet.GeoBin_Extra_Args())


os.environ['OPENCV_LOG_LEVEL'] = 'OFF'


# w-last quaternions here
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


##########################
####    USER INPUT    ####
##########################

nmatch: int = 8 # minimum number of stars to match
starMatchPixelTol: float = 1.0 # pixel match tolerance
min_star_area: float = 3.0 # minimum pixel area for a star
max_star_area: float = 200.0 # maximum pixel area for a star
max_num_stars_to_process: int = 40 # maximum number of centroids to attempt to match per image

low_thresh_pxl_intensity = None
hi_thresh_pxl_intensity = None

VERBOSE = False # set True for prints on results
graphics = False # set True for graphics throughout the solve process

J2000: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "J2000Frame")
J2000Frame = J2000.GetBodyFixedFrame()
planetEntity: st.Entity = this.GetParam(st.VarType.entityRef, "Planet")
planetFixedFrame = planetEntity.GetBodyFixedFrame()
planetName: str = planetEntity.getName()

mu: float = 1e9 * planetEntity.GetParam(st.VarType.double, ["Dynamics", "GravitationalParameter_km3_s2"])
radiusEquatorial: float = planetEntity.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])
Omega: float = planetEntity.getAngVelocity().WRT(J2000Frame).ExprIn(planetFixedFrame)

mapName: str = this.GetParam(st.VarType.string, "MapName")
cameraEntity: st.Entity = this.GetParam(st.VarType.entityRef, "Camera")
topDownCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopDownCamera")
spsDevice: st.Entity = this.GetParam(st.VarType.entityRef, "SPS")

renderDirLocal = os.path.join("Local", "..", "Output", planetName, mapName, "Renders")
outputDir = os.path.join(localOutput, planetName, mapName)
renderDir = os.path.join(outputDir, "Renders")

data_path = os.path.join(thisRepo, 'data') # full path to your data
cam_config_file_path = os.path.join(thisRepo, 'data', 'cam_config', 'Custom_cam.json') # full path (including filename) of your cam config file
darkframe_file_path = os.path.join(thisRepo, 'Images', 'darkframes', 'darkframe.png') # full path (including filename) of your darkframe file
image_extension = ".png" # the image extension to search for in the data_path directory
cat_prefix ='' # if the catalog has a prefix, define it here

#########################
####    MAIN CODE    ####
#########################

# Load star tracker utilities
if darkframe_file_path == '': 
    darkframe_file_path = None
if darkframe_file_path is not None:
    if not os.path.exists(darkframe_file_path):
        darkframe_file_path = None
        st.logger_info("Unable to find provided darkframe file, proceeding without one...")
    else:
        st.logger_info("Darkframe file: " + darkframe_file_path)
else:
    st.logger_info("No darkframe file provided, proceeding without one...")

k = np.load(os.path.join(data_path, cat_prefix + 'k.npy'))
m = np.load(os.path.join(data_path, cat_prefix + 'm.npy'))
q = np.load(os.path.join(data_path, cat_prefix + 'q.npy'))
x_cat = np.load(os.path.join(data_path, cat_prefix + 'u.npy'))
indexed_star_pairs = np.load(os.path.join(data_path, cat_prefix + 'indexed_star_pairs.npy'))

cam_file = cam_config_file_path
camera_matrix, _, _ = read_cam_json(cam_file)
dx: float = camera_matrix[0, 0]
isa_thresh: float = starMatchPixelTol * (1.0 / dx)


def GetImageMetadata(filename: str) -> tuple[st.timestamp, npt.NDArray]:
    # from datetime import datetime
    from PIL import Image

    with Image.open(filename) as img:
        timestamp = img.info.get("CaptureTime", None)
        gravityX = img.info.get("GravityX", None)
        gravityY = img.info.get("GravityY", None)
        gravityZ = img.info.get("GravityZ", None)
        
        simTime = st.timestamp.from_datetime(datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"))
        gravity = np.array([float(gravityX), float(gravityY), float(gravityZ)])
        return simTime, gravity


def SPSEstimate(paramMap: st.ParamMap, timestamp: st.timestamp):
    pos_framed = cameraEntity.getLocation()
    vel_framed = cameraEntity.getVelocity()
    rot_framed = cameraEntity.getRotation()

    EridaniRenderPayload = st.ParamMap()
    EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Loc", pos_framed.WRT_ExprIn(J2000Frame))
    EridaniRenderPayload.AddParam(st.VarType.doubleV3, "Vel", vel_framed.vel_WRT_ExprIn(J2000Frame))
    EridaniRenderPayload.AddParam(st.VarType.doubleV4, "Rot", rot_framed.Quat_WRT(J2000Frame))
    EridaniRenderPayload.AddParam(st.VarType.entityRef, "Frame", J2000)
    EridaniRenderPayload.AddParam(st.VarType.string, "NameOverride", "SPSGuessrRender")
    EridaniRenderPayload.AddParam(st.VarType.string, "OutputPathOverride", str(renderDirLocal))
    payload = st.SimGlobals.Request("EridaniSingleRender", EridaniRenderPayload, timeout=datetime.timedelta(seconds=60.0))

    st.OnScreenLogMessage("Finished Eridani render!", "SPSEstimate", st.Severity.Info)

    dir_contents = os.listdir(renderDir)
    for i in range(len(dir_contents)):
        dir_contents[i] = renderDir + "/" + dir_contents[i]
    dir_contents.sort()

    image_names = []

    for item in dir_contents:
        if image_extension in item:
            image_names+=[os.path.abspath(item)]

    image_filename = image_names[0]
    # Run star tracker
    q_est, idmatch, nmatches, x_obs, rtrnd_img = main.star_tracker(
            image_filename, cam_file, m=m, q=q, x_cat=x_cat, k=k, indexed_star_pairs=indexed_star_pairs, darkframe_file=darkframe_file_path, 
            min_star_area=min_star_area, max_star_area=max_star_area, isa_thresh=isa_thresh, nmatch=nmatch, n_stars=max_num_stars_to_process,
            low_thresh_pxl_intensity=low_thresh_pxl_intensity,hi_thresh_pxl_intensity=hi_thresh_pxl_intensity,graphics=graphics,verbose=VERBOSE, watchdog=5)

    st.OnScreenLogMessage("Finished camera attitude estimate!", "SPSEstimate", st.Severity.Info)

    # Collect data
    quatEstimate: npt.NDArray = 999.0 * np.ones(4)
    try:
        assert not np.any(np.isnan(q_est))
        if VERBOSE:
            st.logger_info(f'Estimated quaternion: {str(q_est)}\n')
        q_rotate = np.array([0.5, -0.5, 0.5, 0.5])  # w-last quaternion
        q_est = quat_mult(q_est, q_rotate)  # w-last quaternion
        quatEstimate[0] = -q_est[0]
        quatEstimate[1] = -q_est[1]
        quatEstimate[2] = -q_est[2]
        quatEstimate[3] = q_est[3]
    except AssertionError:
        if VERBOSE:
            st.logger_info('No valid stars found; returning invalid quaternion.\n')

    simTime, gravity = GetImageMetadata(image_filename)

    cameraEntity.AddOrSetParam(st.VarType.timestamp, ["SPS", "MeasurementTime"], simTime)
    cameraEntity.AddOrSetParam(st.VarType.doubleV3, ["SPS", "MeasuredGravity"], gravity)
    cameraEntity.AddOrSetParam(st.VarType.doubleV4, ["SPS", "MeasuredAttitude"], quatEstimate)

    T_i_b: npt.NDArray = planetEntity.getRotation().DCM_WRT(J2000Frame)
    T_i_c: npt.NDArray = st.math.Quat_to_DCM(quatEstimate)
    T_g_c: npt.NDArray = np.identity(3)

    # Coarse estimates
    r_coarse_1 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, gravity, radiusEquatorial, mu, Omega, 
                                                  planetDataForSPS, planetFixedFrame, False, np.zeros(3), 0.0, 20)
    r_coarse_2 = st.ProcPlanet.SPS.CoarseEstimate(T_i_b, T_i_c, T_g_c, gravity, radiusEquatorial, mu, Omega, 
                                                  planetDataForSPS, planetFixedFrame, True, r_coarse_1, 0.0, 20)

    tol: float = 10.0  # m
    gradientWalkFactor: float = 1.0
    fineOutputs = st.ProcPlanet.SPS.FineEstimate(r_coarse_2, T_i_b, T_i_c, T_g_c, gravity, radiusEquatorial, 
                                                 Omega, planetDataForSPS, planetFixedFrame, gradientWalkFactor, 
                                                 tol, False, 0, 0.0, 20)

    r_bestEstimate = fineOutputs.pos
    r_true = spsDevice.getLocation().WRT_ExprIn(planetFixedFrame)
    st.OnScreenLogMessage(f"r_bestEstimate = {r_bestEstimate}", "SPSEstimate", st.Severity.Info)
    st.OnScreenLogMessage(f"r_true         = {r_true}", "SPSEstimate", st.Severity.Info)
    st.OnScreenLogMessage(f"SPS error      = {np.linalg.norm(r_bestEstimate - r_true)}", "SPSEstimate", st.Severity.Info)
    st.OnScreenLogMessage(f"gravity = {gravity}", "SPSEstimate", st.Severity.Info)

    spsDevice.SetParam(st.VarType.doubleV3, "MapCenter", r_bestEstimate)

    st.OnScreenLogMessage("Finished SPS position estimate!", "SPSEstimate", st.Severity.Info)

    nwu = st.PlanetUtils.NorthWestUpFromLocation(r_bestEstimate, radiusEquatorial)
    north = nwu.north()
    west = nwu.west()
    up = nwu.up()
    spsDevice.SetParam(st.VarType.doubleV3, "MapNorth", north)
    spsDevice.SetParam(st.VarType.doubleV3, "MapEast", -west)

    mapScale_m = spsDevice.GetParam(st.VarType.double, "MapScale")
    cameraHeight_m = spsDevice.GetParam(st.VarType.double, "CameraHeight")
    fov: float = 2.0 * np.arctan(mapScale_m / (2.0 * cameraHeight_m))

    topDownCamera.setLocation(st.frames.FramedLoc(r_bestEstimate + up * cameraHeight_m, planetFixedFrame))
    topDownCameraRot = np.vstack((-up, west, north))
    topDownCamera.setRotation(st.frames.FramedRot(topDownCameraRot, planetFixedFrame))

    time.sleep(0.1)

    captureProperties = st.CaptureImageProperties()
    captureProperties.FOV = np.rad2deg(fov)
    # captureProperties.FOV = fov
    captureProperties.nonphys_EV = spsDevice.GetParam(st.VarType.double, "ExposureValue")

    imageResolution: int = spsDevice.GetParam(st.VarType.int32, "ImageResolution")
    captureProperties.ResolutionX = imageResolution
    captureProperties.ResolutionY = imageResolution

    st.OnScreenLogMessage("Started overhead image capture!", "SPSEstimate", st.Severity.Info)
    capturedImage = st.camera.CaptureImageBlocking(topDownCamera, captureProperties)
    st.OnScreenLogMessage("Completed overhead image capture!", "SPSEstimate", st.Severity.Info)
    
    savePath = os.path.join(st.path_utils.GetLocalOutputDir(), "SPSGuessr")
    os.makedirs(savePath, exist_ok=True)
    if capturedImage is not None:
        imageR = np.asarray(capturedImage.as_RGB8().PixelsR).reshape((captureProperties.ResolutionX, captureProperties.ResolutionY))
        imageG = np.asarray(capturedImage.as_RGB8().PixelsG).reshape((captureProperties.ResolutionX, captureProperties.ResolutionY))
        imageB = np.asarray(capturedImage.as_RGB8().PixelsB).reshape((captureProperties.ResolutionX, captureProperties.ResolutionY))
        imageRGB = Image.fromarray(np.dstack((imageR, imageG, imageB)).astype(np.uint8))

        colorImageName = f"SPSGuessrOverheadImage.png"
        colorImageFullPath = os.path.join(savePath, colorImageName)
        imageRGB.save(colorImageFullPath)
        spsDevice.SetParam(st.VarType.string, "ColorImage", colorImageFullPath)

        st.OnScreenLogMessage("Saved overhead image!", "SPSEstimate", st.Severity.Info)
    else:
        st.OnScreenLogMessage("Failed to save overhead image because image was None", "SPSEstimate", st.Severity.Info)

    altimetryImage = np.zeros((imageResolution, imageResolution))
    slopeImage = np.zeros((imageResolution, imageResolution))
    northSamplePoints = np.linspace(0.5, -0.5, imageResolution)
    westSamplePoints = np.linspace(0.5, -0.5, imageResolution)
    for i in range(len(northSamplePoints)):
        for j in range(len(westSamplePoints)):
            r_ij: npt.NDArray = r_bestEstimate + mapScale_m * (northSamplePoints[i] * north + westSamplePoints[j] * west)
            snappedLoc, normal = st.ProcPlanet.SampleGround(planetDataForRendering, r_ij, radiusEquatorial, 0.0, 22)
            altimetryImage[i, j] = np.linalg.norm(snappedLoc) - radiusEquatorial
            slopeImage[i, j] = np.arccos(np.dot(normalize(snappedLoc), normal))

    st.OnScreenLogMessage("Done generating altimetry and slope maps!", "SPSEstimate", st.Severity.Info)

    figAltimetry = plt.figure()
    axAltimetry = figAltimetry.add_subplot(111)
    axAltimetry.set_axis_off()
    axAltimetry.imshow(altimetryImage, cmap='gist_earth', vmin=np.min(altimetryImage), vmax=np.max(altimetryImage))
    altimetryImageName = f"SPSGuessrAltimetryImage.png"
    altimetryImageFullPath = os.path.join(savePath, altimetryImageName)
    figAltimetry.savefig(altimetryImageFullPath, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    figSlope = plt.figure()
    axSlope = figSlope.add_subplot(111)
    axSlope.set_axis_off()
    axSlope.imshow(slopeImage, cmap='gist_earth', vmin=np.min(slopeImage), vmax=np.max(slopeImage))
    slopeImageName = f"SPSGuessrSlopeImage.png"
    slopeImageFullPath = os.path.join(savePath, slopeImageName)
    figSlope.savefig(slopeImageFullPath, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    st.OnScreenLogMessage("Saved altimetry and slope images!", "SPSEstimate", st.Severity.Info)

    spsDevice.SetParam(st.VarType.string, "AltimetryImage", altimetryImageFullPath)
    spsDevice.SetParam(st.VarType.string, "SlopeImage", slopeImageFullPath)
    spsDevice.SetParam(st.VarType.bool, "UpdateMapFromSPS", True)


def SPSEstimateTryCatch(paramMap: st.ParamMap, timestamp: st.timestamp):
    import traceback
    try:
        st.OnScreenLogMessage("Started SPS estimate!", "SPSEstimate", st.Severity.Info)
        SPSEstimate(paramMap, timestamp)
        st.OnScreenLogMessage("Completed SPS estimate!", "SPSEstimate", st.Severity.Info)
    except Exception as e:
        st.logger_fatal(str(e))


st.SimGlobals.Subscribe("SPSEstimate", SPSEstimateTryCatch)

spsDevice.SetParam(st.VarType.bool, "AllSystemsReady", True)

exitFlag = False
while not exitFlag:
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

st.leave_sim()
