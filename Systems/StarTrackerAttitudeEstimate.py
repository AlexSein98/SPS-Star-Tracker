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

#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

used to evaluate ST alg input params and their effect
on solution accuracy and solve times

"""

##############################
####    LOAD LIBRARIES    ####
##############################

import csv
import psutil
import numpy.typing as npt

from pathlib import Path
import sys

# Folder containing this script
REPO_ROOT = Path(__file__).resolve().parent.parent

# Add an adjacent folder to import search path
sys.path.insert(0, str(REPO_ROOT))

from py_src.star_tracker.star_tracker import main
from py_src.star_tracker.star_tracker.cam_matrix import *
from py_src.star_tracker.star_tracker.array_transformations import *


os.system('cls')
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
np.set_printoptions(suppress=True)

home = os.path.join(st.path_utils.GetLocalAssetsDir(), "Repos", "SPS-Star-Tracker")
imgSourceDir = os.path.join(home, "output", "SPSGuessr")

st.logger_info(f"imgSourceDir = {imgSourceDir}")

data_path = os.path.join(home, 'data')  # full path to your data
cam_config_file_path = os.path.join(home, 'data', 'cam_config', 'Custom_cam.json')  # full path (including filename) of your cam config file
darkframe_file_path = os.path.join(home, 'Images', 'darkframes', 'darkframe.png')  # full path (including filename) of your darkframe file
image_extension = ".png"  # the image extension to search for in the data_path directory
cat_prefix = ''  # if the catalog has a prefix, define it here

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

st.logger_info(f"dx = {dx}, isa_thresh = {isa_thresh}; these should be floats!")

this = st.GetThisSystem()
cameraEntity: st.Entity = this.GetParam(st.VarType.entityRef, "Camera")


def GetImageMetadata(filename: str) -> tuple[st.timestamp, npt.NDArray]:
    # from datetime import datetime
    from PIL import Image

    with Image.open(filename) as img:
        timestamp = img.info.get("CaptureTime", None)
        gravityX = img.info.get("GravityX", None)
        gravityY = img.info.get("GravityY", None)
        gravityZ = img.info.get("GravityZ", None)
        
        simTime = st.timestamp.from_datetime(datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"))
        gravity = np.array([gravityX, gravityY, gravityZ])
        return simTime, gravity


def ProcessSPSImages(paramMap: st.ParamMap, timestamp: st.timestamp):
    # sys.excepthook = custom_exception_handler

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
    gx = []
    gy = []
    gz = []

    # Create list of all images in target dir
    total_start = time.time()

    dir_contents = os.listdir(imgSourceDir)
    for i in range(len(dir_contents)):
        dir_contents[i] = os.path.join(imgSourceDir, dir_contents[i])
        # st.logger_info(f'dir_contents[{i}] = {dir_contents[i]}')
    dir_contents.sort()

    image_names = []

    for item in dir_contents:
        if image_extension in item:
            image_names += [os.path.abspath(item)]

    idx: int = 0
    for image_filename in image_names:
        image_name += [image_filename]

        # Run star tracker
        solve_start_time = time.time()

        # st.logger_info(f"Star tracker image proc: file = {image_filename}")

        q_est, idmatch, nmatches, x_obs, rtrnd_img = main.star_tracker(
                image_filename, cam_file, m=m, q=q, x_cat=x_cat, k=k, indexed_star_pairs=indexed_star_pairs, darkframe_file=darkframe_file_path, 
                min_star_area=min_star_area, max_star_area=max_star_area, isa_thresh=isa_thresh, nmatch=nmatch, n_stars=max_num_stars_to_process,
                low_thresh_pxl_intensity=low_thresh_pxl_intensity,hi_thresh_pxl_intensity=hi_thresh_pxl_intensity,graphics=graphics,verbose=VERBOSE, watchdog=5)

        solve_time += [time.time()-solve_start_time]

        # Collect data
        try:
            assert not np.any(np.isnan(q_est))
            if VERBOSE:
                st.logger_info('Estimated q: ' + str(q_est)+'\n')
            q_rotate = np.array([0.5, -0.5, 0.5, 0.5])  # w-last quaternion
            q_est = quat_mult(q_est, q_rotate)  # w-last quaternion
            qs += [q_est[3]]
            qv0 += [q_est[0]]
            qv1 += [q_est[1]]
            qv2 += [q_est[2]]
        except AssertionError:
            if VERBOSE:
                st.logger_info('NO VALID STARS FOUND\n')
            qs += [999]
            qv0 += [999]
            qv1 += [999]
            qv2 += [999]

        simTime, gravity = GetImageMetadata(image_filename)
        ttime += [simTime]
        sram  += [psutil.virtual_memory().percent]
        #scpu  += [psutil.cpu_percent(2)]  # TODO: what is this?
        scpu  += [psutil.cpu_percent()]

        # gravityFramed = st.SimGlobals.SampleVectorField("Gravity", cameraEntity.getAcceleration())
        # gravity = gravityFramed.ExprIn(cameraEntity.GetBodyFixedFrame())
        gx += [gravity[0]]
        gy += [gravity[1]]
        gz += [gravity[2]]

        st.logger_info(f'Completed image {idx} ({round(float(idx) / float(len(image_names)) * 100.0, 2)} %)')
        idx += 1

    data = {
        'Image Name': image_name,
        'Sim Time': ttime,
        'RAM': sram,
        'CPU': scpu,
        'Solve Time (s)': solve_time, 
        'qx': qv0,
        'qy': qv1,
        'qz': qv2,
        'qw': qs,
        'gx': gx,
        'gy': gy,
        'gz': gz
    }

    now = str(datetime.datetime.now())
    now = now.split('.')
    now = now[0]
    now = now.replace(' ', '_')
    now = now.replace(':', '-')

    # Write data to output file
    # keys=sorted(data.keys())  # Why are we sorting??
    keys=data.keys()

    filename = os.path.join(home, "output/SPSCameraAttitudes.csv")

    with open(filename,'w', newline='') as csv_file:
        writer=csv.writer(csv_file)
        writer.writerow(keys)
        writer.writerows(zip(*[data[key] for key in keys]))

    st.logger_info("\n Took " + str(time.time() - total_start) + " seconds to complete \n")
    st.logger_info("Data saved to: " + filename + "\n")


def ProcessSPSTryCatch(paramMap: st.ParamMap, timestamp: st.timestamp):
    import traceback
    try:
        ProcessSPSImages(paramMap, timestamp)
    except Exception as e:
        st.logger_fatal(traceback.format_exc())
        # st.logger_fatal(str(e))


st.SimGlobals.Subscribe("ProcessSPSImages", ProcessSPSTryCatch)

exitFlag = False
while not exitFlag:
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

st.leave_sim()
