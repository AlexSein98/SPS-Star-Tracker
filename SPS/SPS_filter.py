import numpy as np
import numpy.typing as npt
import copy
import os
import time
import datetime
import psutil
import matplotlib.pyplot as plt
from pathlib import Path

from py_src.star.python.render import *
from SPS.SPS_sigma_points import *

from star_tracker import main
from star_tracker.cam_matrix import *
from star_tracker.array_transformations import *

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'


#####################################
####    Classes and Functions    ####
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


class ModifiedRodriguesParameters:
    def __init__(self, _sigma1: float, _sigma2: float, _sigma3: float):
        self.sigma1: float = _sigma1
        self.sigma2: float = _sigma2
        self.sigma3: float = _sigma3
        self.sigmaSquared: float = _sigma1 ** 2 + _sigma2 ** 2 + _sigma3 ** 2
        self.sigma_x = np.array([[0.0, -_sigma3, _sigma2],
                                 [_sigma3, 0.0, -_sigma1],
                                 [-_sigma2, _sigma1, 0.0]])
        self.sigma_xx = self.sigma_x @ self.sigma_x
    
    @classmethod
    def FromVector(cls, v: npt.NDArray):
        return ModifiedRodriguesParameters(v[0], v[1], v[2])
    
    @classmethod
    def FromQuat(cls, q: Quaternion):
        sigma = q.v() / (1.0 + q.w)
        return ModifiedRodriguesParameters(sigma[0], sigma[1], sigma[2])
    
    @classmethod
    def FromMatrix(cls, T: npt.NDArray):
        q = Quaternion.FromMatrix(T)
        return cls.FromQuat(q)

    @classmethod
    def Zero(cls):
        return ModifiedRodriguesParameters(0.0, 0.0, 0.0)

    def ToMatrix(self) -> npt.NDArray:
        T: npt.NDArray = np.identity(3) + (8.0 * self.sigma_xx - 4.0 * (1.0 - self.sigmaSquared) * self.sigma_x) \
            / (1.0 + self.sigmaSquared)
        return T

    def ToQuat(self) -> Quaternion:
        qw = (1.0 - self.sigmaSquared) / (1.0 + self.sigmaSquared)
        qv = 2.0 * np.array([self.sigma1, self.sigma2, self.sigma3]) / (1.0 + self.sigmaSquared)
        return Quaternion(qw, qv[0], qv[1], qv[2])

    def ToVector(self) -> npt.NDArray:
        return np.array([self.sigma1, self.sigma2, self.sigma3])

    def shadow(self):
        return ModifiedRodriguesParameters.FromVector(-self.ToVector() / self.sigmaSquared)

    def dSigma_dsigma(self, s: float, dsigma_x: npt.NDArray, dsigma_xx: npt.NDArray) -> npt.NDArray:
        deriv: npt.NDArray = (8.0 * dsigma_xx + 8.0 * s * self.sigma_x - 
                              4.0 * (1.0 - self.sigmaSquared) * dsigma_x) / (1.0 + self.sigmaSquared) - \
                                (2.0 * s * (8.0 * self.sigma_xx - 4.0 * (1.0 - self.sigmaSquared) * self.sigma_x)) \
                                    / ((1.0 + self.sigmaSquared) ** 2)
        return deriv
    
    def dT_dsigma(self) -> list[npt.NDArray]:
        s1 = self.sigma1
        s2 = self.sigma2
        s3 = self.sigma3
        dsigma_x_1 = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])
        dsigma_x_2 = np.array([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0]])
        dsigma_x_3 = np.array([[0.0, -1.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])
        dsigma_xx_1 = np.array([[0.0, s2, s3],
                                [s2, -2.0 * s1, 0.0],
                                [s3, 0.0, -2.0 * s1]])
        dsigma_xx_2 = np.array([[-2.0 * s2, s1, 0.0],
                                [s1, 0.0, s3],
                                [0.0, s3, -2.0 * s2]])
        dsigma_xx_3 = np.array([[-2.0 * s3, 0.0, s1],
                                [0.0, -2.0 * s3, s2],
                                [s1, s2, 0.0]])
        
        deriv1: npt.NDArray = self.dSigma_dsigma(s1, dsigma_x_1, dsigma_xx_1)
        deriv2: npt.NDArray = self.dSigma_dsigma(s2, dsigma_x_2, dsigma_xx_2)
        deriv3: npt.NDArray = self.dSigma_dsigma(s3, dsigma_x_3, dsigma_xx_3)

        return [deriv1, deriv2, deriv3]


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    delete_old = False
    regenerate_catalog = True
    reprocess_star_tracker = True

    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel
    dem = ReadDEM(planet.demName)

    truthDataPath = globalConfig.outputDir + "truth_data_" + planet.planetName.title() + ".csv"
    attitudeEstDataPath = globalConfig.outputDir + "attitudes_" + globalConfig.nameTitle + ".csv"
    gravEstDataPath = globalConfig.outputDir + "measurements_" + planet.planetName.title() + ".csv"

    # Error sources
    addMeasurementBias: bool = globalConfig.addMeasurementBias
    addMeasurementNoise: bool = globalConfig.addMeasurementNoise
    
    sigma_2 = np.array([[1e-8, 0.0, 0.0],
                        [0.0, 1e-8, 0.0],
                        [0.0, 0.0, 1e-8]])  # Worst case for the BMA220 IMU (once averaged out)
    biasSigma_2 = np.array([[0.15 ** 2, 0.0, 0.0],
                            [0.0, 0.15 ** 2, 0.0],
                            [0.0, 0.0, 0.15 ** 2]])
    bias = np.linalg.cholesky(biasSigma_2) @ np.random.randn(3)

    # Delete all old images
    if delete_old:
        if os.path.exists(globalConfig.renderDir):
            shutil.rmtree(globalConfig.renderDir)
        os.mkdir(globalConfig.renderDir)

    # Get full star catalog
    catalog = read_csv_catalog("./py_src/star/data/catalog.csv")

    # Star rendering parameters
    relativeMagnitude = 8.0  # "full" exposure is set for stars of this magnitude
    relativeFlux = 1.0  # flux for fully-exposed stars
    starMaxPixelRadius = 4

    # Astrophysical parameters
    spice.furnsh("./py_src/star/data/metakernel.txt")
    tJ2000 = '2000 Jan 1, 00:00:00 UTC'
    tNow = globalConfig.tNow
    etJ2000 = spice.str2et(tJ2000)
    etNow = spice.str2et(tNow)
    etOriginal = copy.deepcopy(etNow)
    times = etOriginal + dt + dt * np.linspace(0.0, numImages - 1, numImages)
    
    idx = 0
    numImages = int(0.25 * 8640)
    true_data = []
    measured_accelerations = []

    # Gravity model
    maxDegree: int = globalConfig.grav_maxDegree
    maxOrder: int = globalConfig.grav_maxOrder
    sampler = GravSampler(gravModel, maxDegree, maxOrder)

    # Camera parameters
    dimU: int = 1024
    dimV: int = 1024
    fovU: float = 20.0

    radiusEquatorial: float = gravModel.radius
    radiusPolar: float = gravModel.polarRadius
    Omega = np.array([0.0, 0.0, gravModel.omega])  # Expressed in the planet-fixed frame

    startTime = time.perf_counter()
    elapsedSeconds: float = 0.0
    printInterval: int = 10

    # Something like Connecting Ridge
    phi_pg_0: float = -89.45
    lon_pg_0: float = -137.2
    h_ellp_0: float = 1960.0

    # Something like Apollo 17
    # phi_pg_0: float = 20.19
    # lon_pg_0: float = 30.77
    # h_ellp_0: float = -2500.0

    # Something with low gravity variation
    # phi_pg_0: float = 4.0
    # lon_pg_0: float = -29.4
    # h_ellp_0: float = 0.0
    
    cameraPosPlanetFixed = planetographic_to_cartesian(phi_pg_0, lon_pg_0, h_ellp_0, radiusEquatorial, radiusPolar)
    cameraPosPlanetFixed = SnapToSurface(cameraPosPlanetFixed, planet, dem)
    lat_pc, lon_pc, h_pc = r_to_latlonalt(cameraPosPlanetFixed, planet.radius)
    dt = 1.0

    #############################
    ####    Render Images    ####
    #############################

    if not os.path.exists(globalConfig.renderDir) or is_dir_empty(globalConfig.renderDir):
        for i in range(numImages):
            doPrint: bool = i % printInterval == 0

            # Step time forward by 1 second
            etNow += dt
            planetRot = spice.pxform("J2000", planet.planetFrame, etNow)
            cameraPosPlanetCentered = (planetRot.T @ np.array([cameraPosPlanetFixed]).T).T[0]
            
            phi_pc, _, _ = r_to_latlonalt(cameraPosPlanetFixed, radiusEquatorial)
            g = sampler.SampleAcceleration_Custom(phi_pc, lon_pc, np.linalg.norm(cameraPosPlanetFixed), maxDegree, 
                                                  overrideSphericalHarmonics=False, noRadialTerm=False, 
                                                  includeThirdBody=True, et=etNow)
            g -= np.cross(Omega, np.cross(Omega, cameraPosPlanetFixed))  # Handle being on the surface of the planet
            g_true = copy.deepcopy(g)
            
            lat_pc, lon_pc, h_pc = r_to_latlonalt(cameraPosPlanetFixed, planet.radius)
            T_P_G = latlon_to_T(lat_pc, lon_pc).T
            g_IMU_frame = (T_P_G @ np.array([g]).T).T[0]

            if addMeasurementBias:
                g_IMU_frame += bias

            if addMeasurementNoise:
                g_IMU_frame += np.linalg.cholesky(sigma_2) @ np.random.randn(3)
            
            measured_accelerations.append(g_IMU_frame)
            # measured_accelerations.append(np.array([-np.linalg.norm(g_IMU_frame), 0.0, 0.0]))

            gInertial_true = (planetRot.T @ np.array([g_true]).T).T[0]
            gInertial = (planetRot.T @ T_P_G.T @ np.array([g_IMU_frame]).T).T[0]

            ra_true, de_true = r_hat_to_ra_dec(-normalize(gInertial_true))
            ra, de = r_hat_to_ra_dec(-normalize(gInertial))
            
            planetPos, _ = spice.spkpos(planet.planetName, etNow, "J2000", "NONE", "SSB")
            cameraPos = planetPos + 0.001 * cameraPosPlanetCentered  # needs to be in km
            
            # Camera pointing vector
            ra, de = r_hat_to_ra_dec(normalize(cameraPosPlanetCentered))

            # Render
            idx += 1

            if doPrint:
                print(f'Rendering image {idx} of {numImages} ({round(100.0 * float(idx) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}')
            params = RenderParams(idx, etJ2000, etNow, cameraPos, dimU, dimV, fovU, starMaxPixelRadius, 
                                catalog, relativeMagnitude, relativeFlux, planet)
            
            # Actually do the render
            render(ra, de, params, globalConfig.renderDir)
            true_data.append(get_true_attitude(ra_true, de_true))

            endTime = time.perf_counter()
            elapsedSeconds = endTime - startTime
            elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

            if doPrint:
                projectedRemainingSeconds: float = elapsedSeconds * float(numImages - idx) / float(idx)
                projectedRemainingTime = datetime.timedelta(seconds=round(projectedRemainingSeconds))
                print(f'Elapsed time: {elapsedTime}. Remaining time estimate: {projectedRemainingTime}\n')
            
        write_csv(truthDataPath, true_data)
        write_csv(gravEstDataPath, measured_accelerations)

    else:
        print("Render directory not empty; skipping render step...\n")
    
    #########################################
    ####    Obtain Attitude Estimates    ####
    #########################################

    if regenerate_catalog:
        etOriginal

    if reprocess_star_tracker:
        Path(attitudeEstDataPath).unlink(missing_ok=True)

    if not Path(attitudeEstDataPath).is_file():
        ################################
        #USER INPUT
        ################################
        
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

        ################################
        #SUPPORT FUNCTIONS
        ################################

        print(f'imgSourceDir = {imgSourceDir}')

        ################################
        #MAIN CODE
        ################################

        #load star tracker stuff
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

        #define structures for data capture
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

        # create list of all images in target dir
        total_start = time.time()

        dir_contents = os.listdir(imgSourceDir)
        for i in range(len(dir_contents)):
            dir_contents[i] = imgSourceDir + "/" + dir_contents[i]
        dir_contents.sort()

        image_names = []

        for item in dir_contents:
            if image_extension in item:
                image_names+=[os.path.abspath(item)]
                # image_names += [item]

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
    gravEstData = read_csv(gravEstDataPath)

    # Very basic error handling if datasets are not the same length
    if not (len(truthData) == len(attitudeEstData) == len(gravEstData)):
        print(f'Warning: early exit due to dataset length mismatch; truthData length = {len(truthData)}, attitudeEstData length = {len(attitudeEstData)}, and gravEstData length = {len(gravEstData)}.')
        exit(0)

    # Initialize all inertial-to-planet attitude matrices
    T_i_b_list: list[npt.NDArray] = []
    T_i_c_list: list[npt.NDArray] = []
    g_est_list: list[npt.NDArray] = []
    for i in range(len(times)):
        _T_i_b = spice.pxform("J2000", planet.planetFrame, times[i])
        T_i_b_list.append(_T_i_b)

        attitudeEst_j = attitudeEstData[i]
        # if attitudeEst_j[0] == 999 or attitudeEst_j[1] == 999 or attitudeEst_j[2] == 999 or attitudeEst_j[3] == 999:
        #     print(f'Warning: skipped measurement at index {i} (invalid quaternion).')
        #     continue
        q_est = Quaternion(attitudeEst_j[0], attitudeEst_j[1], attitudeEst_j[2], attitudeEst_j[3]).normalize()
        T_i_c_list.append(q_est.to_matrix())

        gravEst_j = gravEstData[i]
        g_est_list.append(gravEst_j)

    ###############################
    ####    SPS Calibration    ####
    ###############################

    midpoint: int = 100
    endpoint: int = 2000

    s_0: npt.NDArray = np.zeros(3)
    Pss_0: npt.NDArray = np.zeros((3, 3))

    print(f"Initial guess: {ModifiedRodriguesParameters.FromVector(s_0).ToMatrix()}\n")

    ds: npt.NDArray = np.zeros(3)
    eps: float = 7e-5
    iteration: int = 0
    hasPriorEstimate: bool = False

    # Force at least one iteration without setting a dummy value for ds_0
    while np.linalg.norm(ds) > eps or iteration < 1:
        print(f"==============================================")
        print(f"Iteration {iteration} of calibration algorithm")
        print(f"==============================================\n")
        
        lam: npt.NDArray = np.zeros(3)
        Lam: npt.NDArray = np.zeros((3, 3))
        if hasPriorEstimate:
            Lam = np.linalg.inv(Pss_0)
            lam = Lam @ ds
        
        s_0_MRP = ModifiedRodriguesParameters.FromVector(s_0)
        derivs = s_0_MRP.dT_dsigma()
        
        Pvv = np.diag(np.array([1.0, 1.0, 1.0])) * 1e-8
        Pvv_inv = np.linalg.inv(Pvv)
        
        # print(f"g_truth = {g_truth}")

        for j in range(len(times[:midpoint])):
            phi_pc, lon_pc, _ = r_to_latlonalt(cameraPosPlanetFixed, radiusEquatorial)
            g_truth = sampler.SampleAcceleration_Custom(phi_pc, lon_pc, np.linalg.norm(cameraPosPlanetFixed), maxDegree, 
                                                        overrideSphericalHarmonics=False, noRadialTerm=False, 
                                                        includeThirdBody=True, et=times[j])
            g_truth -= np.cross(Omega, np.cross(Omega, cameraPosPlanetFixed))  # Handle being on the surface of the planet
            
            g_product = T_i_c_list[j] @ T_i_b_list[j].T @ g_truth
            H_1 = np.array([derivs[0] @ g_product]).T
            H_2 = np.array([derivs[1] @ g_product]).T
            H_3 = np.array([derivs[2] @ g_product]).T
            H_sigma = np.hstack((H_1, H_2, H_3))
            H_T_Pvv_inv = H_sigma.T @ Pvv_inv

            h_sigma = s_0_MRP.ToMatrix() @ g_product
            lam += H_T_Pvv_inv @ (g_est_list[j] - h_sigma)
            Lam += H_T_Pvv_inv @ H_sigma

            # print(f"g_product = {g_product}")
            # print(f"h_sigma   = {h_sigma}")
            # print(f"lam       = {lam}")
            # print(f"Lam       = {Lam}")

        Pss_0 = np.linalg.inv(Lam)
        ds = Pss_0 @ lam
        # s_0 += 1e3 * np.linalg.norm(ds) * ds
        s_0 += 0.1 * ds

        # s_0_MRP = ModifiedRodriguesParameters.FromVector(s_0)
        # if s_0_MRP.sigmaSquared > 1.0:
        #     s_0_MRP = s_0_MRP.shadow()
        #     s_0 = s_0_MRP.ToVector()

        print(f"|ds| = {np.linalg.norm(ds)}")
        # print(f"s_0   = {s_0}")
        # print(f"Pss_0 = {Pss_0}")

        iteration += 1
        hasPriorEstimate = True
        # print(f"ds = {ds}")
    
    # s_0 /= 2.0
    s_0_MRP_final = ModifiedRodriguesParameters.FromVector(s_0)
    T_calibration = s_0_MRP_final.ToMatrix()
    print(f"\nFinal estimate for calibration MRP:        {s_0}")
    print(f"Final estimate for calibration matrix:     {T_calibration}")
    print(f"Final estimate for calibration covariance: {Pss_0}")

    # exit(0)

    #################################
    ####    SPS Kalman Filter    ####
    #################################
    
    # Position error logging
    # position_errors: list[npt.NDArray] = []
    # distanceErrors_m: list[float] = []
    # distanceErrors_km: list[float] = []

    # Tolerances and scale factors
    limit_km: float = 20.0
    tol: float = 10.0  # m
    scaleFactor = 1000.0 if planet.demUnits == "km" else 1.0
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
    Pww: npt.NDArray = np.diag(np.array([1.0, 1.0, 1.0])) * 1e1 ** 2
    Pvv: npt.NDArray = np.diag(np.array([1.0, 1.0, 1.0])) * 2e2 ** 2

    Hx: npt.NDArray = np.identity(3)

    mx_plus = copy.deepcopy(mx_0)
    Pxx_plus = copy.deepcopy(Pxx_0)

    mx_history: list[npt.NDArray] = []
    Pxx_history: list[npt.NDArray] = []

    alpha_underweight = 3.0
    gamma_underweight = 0.3
    # T_g_c = np.identity(3)  # transformation from gravity to camera frame
    T_g_c = T_calibration.T  # transformation from gravity to camera frame
    for j in range(len(times[midpoint:endpoint])):
        ######################################
        ####    Measurement Processing    ####
        ######################################
        
        doPrint: bool = j % printInterval == 0

        T_i_b: npt.NDArray = T_i_b_list[j + midpoint]
        truth_j = truthData[j + midpoint]
        attitudeEst_j = attitudeEstData[j + midpoint]
        gravEst_j = gravEstData[j + midpoint]
        
        if attitudeEst_j[0] == 999 or attitudeEst_j[1] == 999 or attitudeEst_j[2] == 999 or attitudeEst_j[3] == 999:
            print(f'Warning: skipped measurement at index {j} (invalid quaternion).')
            continue
        
        q_est = Quaternion(attitudeEst_j[0], attitudeEst_j[1], attitudeEst_j[2], attitudeEst_j[3]).normalize()
        T_i_c = q_est.to_matrix()

        Omega = np.array([0.0, 0.0, gravModel.omega])
        g_sensorFrame = np.array([gravEst_j[0], gravEst_j[1], gravEst_j[2]])

        # Coarse estimates
        r_coarse_1 = CoarseEstimate_SurfaceFixed(T_i_b, T_i_c, T_g_c, g_sensorFrame, times[j + midpoint], planet, 
                                                 gravModel, sampler, dem, scaleFactor, Omega)
        r_coarse_2 = CoarseEstimate_SurfaceFixed(T_i_b, T_i_c, T_g_c, g_sensorFrame, times[j + midpoint], planet, 
                                                 gravModel, sampler, dem, scaleFactor, Omega, r_coarse_1)
        r_coarse_3 = CoarseEstimate_SurfaceFixed(T_i_b, T_i_c, T_g_c, g_sensorFrame, times[j + midpoint], planet, 
                                                 gravModel, sampler, dem, scaleFactor, Omega, r_coarse_2)
        r_coarse_4 = CoarseEstimate_SurfaceFixed(T_i_b, T_i_c, T_g_c, g_sensorFrame, times[j + midpoint], planet, 
                                                 gravModel, sampler, dem, scaleFactor, Omega, r_coarse_3)
        r_coarse_5 = CoarseEstimate_SurfaceFixed(T_i_b, T_i_c, T_g_c, g_sensorFrame, times[j + midpoint], planet, 
                                                 gravModel, sampler, dem, scaleFactor, Omega, r_coarse_4)
        
        if doPrint:
            print(f'Sample point {j}:')
            r_expected = planetographic_to_cartesian(latTruth, lonTruth, altTruth, 
                                                     planet.radius, gravModel.polarRadius)
            print(f'r_expected = {r_expected}')
            print(f'r_coarse_1 = {r_coarse_1}')
            print(f'r_coarse_2 = {r_coarse_2}')
            print(f'r_coarse_3 = {r_coarse_3}')
            print(f'r_coarse_4 = {r_coarse_4}')
            print(f'r_coarse_5 = {r_coarse_5}')
        
        fineOutputs = FineEstimate(r_coarse_5, T_i_b, T_i_c, T_g_c, g_sensorFrame, times[j + midpoint], planet, gravModel, sampler, 
                                   dem, scaleFactor, Omega, maxDegree, gradientWalkFactor, tol, doPrint, j + midpoint)

        r_bestEstimate = fineOutputs.pos
        phi_pg = fineOutputs.phi_pg
        lon = fineOutputs.lon
        alt = fineOutputs.alt
        i = fineOutputs.iterations
        
        if doPrint:
            print(f'Estimated lat = {round(phi_pg, 6)} deg')
            print(f'Estimated lon = {round(lon, 6)} deg')
            print(f'True lat = {round(latTruth, 6)} deg')
            print(f'True lon = {round(lonTruth, 6)} deg\n')
        
        ##################################
        ####    Filter Propagation    ####
        ##################################

        mx_minus = copy.deepcopy(mx_plus)
        Pxx_minus = Pxx_plus + Pww
        
        #############################
        ####    Filter Update    ####
        #############################

        # Measurement editing: if more than 600-sigma, discard
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
        
        mx_history.append(mx_plus)
        Pxx_history.append(Pxx_plus)
        
        #####################################
        ####    Clean-up and Printing    ####
        #####################################

        percentComplete = round(100.0 * float(j) / float(len(times[midpoint:endpoint])), 3)
        
        endTime = time.perf_counter()
        elapsedSeconds = endTime - startTime
        elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

        if doPrint:
            print(f'Elapsed time: {elapsedTime}')
            print("------------------------------------------------------------------------------------------------------\n")
    
    ########################
    ####    Plotting    ####
    ########################
    
    fig1 = plt.figure(layout='constrained')
    ax1 = fig1.add_subplot(131)
    ax2 = fig1.add_subplot(132)
    ax3 = fig1.add_subplot(133)

    mx_x = [mx[0] for mx in mx_history]
    mx_y = [mx[1] for mx in mx_history]
    mx_z = [mx[2] for mx in mx_history]

    Pxx_x = [Pxx[0, 0] for Pxx in Pxx_history]
    Pxx_y = [Pxx[1, 1] for Pxx in Pxx_history]
    Pxx_z = [Pxx[2, 2] for Pxx in Pxx_history]

    ax1.plot(times[midpoint:endpoint] - etOriginal, mx_x - cameraPosPlanetFixed[0], label=r"$m_{x}(0)$", color='blue')
    ax1.plot(times[midpoint:endpoint] - etOriginal, -3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r', label=r"$P_{xx}(0,0)$")
    ax1.plot(times[midpoint:endpoint] - etOriginal, 3.0 * np.sqrt(Pxx_x), linestyle='dashed', color='r')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position error (m)")
    ax1.set_title(r"Error in $m_{x}(0)$ over Time")
    ax1.grid()
    ax1.legend()

    ax2.plot(times[midpoint:endpoint] - etOriginal, mx_y - cameraPosPlanetFixed[1], label=r"$m_{x}(1)$", color='blue')
    ax2.plot(times[midpoint:endpoint] - etOriginal, -3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r', label=r"$P_{xx}(1,1)$")
    ax2.plot(times[midpoint:endpoint] - etOriginal, 3.0 * np.sqrt(Pxx_y), linestyle='dashed', color='r')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title(r"Error in $m_{x}(1)$ over Time")
    ax2.grid()
    ax2.legend()
    
    ax3.plot(times[midpoint:endpoint] - etOriginal, mx_z - cameraPosPlanetFixed[2], label=r"$m_{x}(2)$", color='blue')
    ax3.plot(times[midpoint:endpoint] - etOriginal, -3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r', label=r"$P_{xx}(2,2)$")
    ax3.plot(times[midpoint:endpoint] - etOriginal, 3.0 * np.sqrt(Pxx_z), linestyle='dashed', color='r')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position error (m)")
    ax3.set_title(r"Error in $m_{x}(2)$ over Time")
    ax3.grid()
    ax3.legend()

    plt.show()
