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
from SPS.global_config import *

from star_tracker import main
from star_tracker.cam_matrix import *
from star_tracker.array_transformations import *

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'


# ----------------------------
# Rotation helpers
# ----------------------------

def rad_to_arcsec(rad: float) -> float:
    return 3600.0 * np.rad2deg(rad)


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


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        # Ensure normalized quaternion on init
        norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
        norm_1 = 1.0 / norm

        self.w = w * norm_1
        self.x = x * norm_1
        self.y = y * norm_1
        self.z = z * norm_1
    
    @classmethod
    def FromMatrix(quat, m: npt.NDArray):
        r__ = np.array([m[0, 0], m[0, 1], m[0, 2], 
                        m[1, 0], m[1, 1], m[1, 2], 
                        m[2, 0], m[2, 1], m[2, 2]])
        s = np.zeros(3)

        trace = r__[0] + r__[4] + r__[8]
        mtrace = 1. - trace
        cc4 = trace + 1.
        s114 = mtrace + r__[0] * 2.
        s224 = mtrace + r__[4] * 2.
        s334 = mtrace + r__[8] * 2.

        if (1. <= cc4):
            c__ = np.sqrt(cc4 * .25)
            factor = 1. / (c__ * 4.)
            s[0] = (r__[5] - r__[7]) * factor
            s[1] = (r__[6] - r__[2]) * factor
            s[2] = (r__[1] - r__[3]) * factor
        elif (1. <= s114):
            s[0] = np.sqrt(s114 * .25)
            factor = 1. / (s[0] * 4.)
            c__ = (r__[5] - r__[7]) * factor
            s[1] = (r__[3] + r__[1]) * factor
            s[2] = (r__[6] + r__[2]) * factor
        elif (1. <= s224):
            s[1] = np.sqrt(s224 * .25)
            factor = 1. / (s[1] * 4.)
            c__ = (r__[6] - r__[2]) * factor
            s[0] = (r__[3] + r__[1]) * factor
            s[2] = (r__[7] + r__[5]) * factor
        else:
            s[2] = np.sqrt(s334 * .25)
            factor = 1. / (s[2] * 4.)
            c__ = (r__[1] - r__[3]) * factor
            s[0] = (r__[6] + r__[2]) * factor
            s[1] = (r__[7] + r__[5]) * factor

        q = np.zeros(4)
        l2 = c__ * c__ + s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        if (l2 != 1.):
            polish = 1. / np.sqrt(l2)
            c__ *= polish
            s[0] *= polish
            s[1] *= polish
            s[2] *= polish
        if (c__ > 0.):
            q[0] = c__
            q[1] = s[0]
            q[2] = s[1]
            q[3] = s[2]
        else:
            q[0] = -c__
            q[1] = -s[0]
            q[2] = -s[1]
            q[3] = -s[2]
        
        return quat(q[0], q[1], q[2], q[3])

    def positivize(self):
        if self.w < 0.0:
            return Quaternion(-self.w, -self.x, -self.y, -self.z)
        else:
            return self

    def mult(self, other):
        other = other.normalize()
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
                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2).normalize()
    
    def normalize(self):
        mag_1 = 1.0 / np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        return Quaternion(self.w * mag_1, self.x * mag_1, self.y * mag_1, self.z * mag_1)
    
    def conjugate(self):
        return Quaternion(-self.w, self.x, self.y, self.z)
    
    def as_w_first_array(self) -> npt.NDArray:
        return np.array([self.w, self.x, self.y, self.z])
    
    def as_w_last_array(self) -> npt.NDArray:
        return np.array([self.x, self.y, self.z, self.w])
    
    def to_matrix(self) -> npt.NDArray:
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


def read_quats(filename: str) -> list[Quaternion]:
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
            quats.append(Quaternion(w, x, y, z).positivize())
    return quats


def xi_transpose(q: npt.NDArray, active: bool) -> npt.NDArray:
    qw = copy.deepcopy(q[0])
    qx = copy.deepcopy(q[1])
    qy = copy.deepcopy(q[2])
    qz = copy.deepcopy(q[3])

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


def estimate_omega(quats: list[Quaternion], deltaT: float, T_unrotate: list[npt.NDArray]) -> npt.NDArray:
    """
    IMPORTANT:
    q_dot is kept as a raw 4-vector and is NOT normalized.
    """
    # quats = align_quaternion_signs(quats)

    N = len(quats) - 1
    omega_sum = np.zeros(3)
    omega_history: list[npt.NDArray] = []

    for i in range(1, N):
        q_prev = Quaternion.FromMatrix(T_unrotate[i - 1] @ quats[i - 1].to_matrix())
        q_curr = Quaternion.FromMatrix(T_unrotate[i - 1] @ quats[i].to_matrix())
        q_next = Quaternion.FromMatrix(T_unrotate[i - 1] @ quats[i + 1].to_matrix())

        q_dot = np.array([[(q_next.w - q_prev.w) / (2.0 * deltaT),
                           (q_next.x - q_prev.x) / (2.0 * deltaT),
                           (q_next.y - q_prev.y) / (2.0 * deltaT),
                           (q_next.z - q_prev.z) / (2.0 * deltaT)]]).T
        
        T_extra = np.array([[0, 0, 1], 
                            [1, 0, 0], 
                            [0, 1, 0]])
        omega_i = (2.0 * T_extra @ xi_transpose(q_curr.as_w_first_array(), True) @ q_dot).T[0]
        omega_history.append(omega_i)
        omega_sum += omega_i

    return omega_sum / float(len(omega_history)), omega_history


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


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel

    # Delete all old images
    delete_old = False
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
    # planetRot = spice.pxform("J2000", planet.planetFrame, etNow)
    
    idx = 0
    numImages = int(0.1 * 8640)

    # Camera parameters
    dimU: int = 1024
    dimV: int = 1024
    fovU: float = 20.0

    radiusEquatorial: float = gravModel.radius
    radiusPolar: float = gravModel.polarRadius

    startTime = time.perf_counter()
    elapsedSeconds: float = 0.0
    printInterval: int = 10

    # Something like College Station's coordinates
    phi_pg: float = 30.0
    lon: float = -96.0
    h_ellp: float = 20.0
    cameraPosPlanetFixed = planetographic_to_cartesian(phi_pg, lon, h_ellp, radiusEquatorial, radiusPolar)
    lat_pc, lon_pc, h_pc = r_to_latlonalt(cameraPosPlanetFixed, planet.radius)
    dt = 100.0

    if not os.path.exists(globalConfig.renderDir) or is_dir_empty(globalConfig.renderDir):
        for i in range(numImages):
            doPrint: bool = i % printInterval == 0

            # Step time forward by 1 second
            etNow += dt
            planetRot = spice.pxform("J2000", planet.planetFrame, etNow)
            cameraPosPlanetCentered = (planetRot.T @ np.array([cameraPosPlanetFixed]).T).T[0]
            
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

            endTime = time.perf_counter()
            elapsedSeconds = endTime - startTime
            elapsedTime = datetime.timedelta(seconds=round(elapsedSeconds))

            if doPrint:
                projectedRemainingSeconds: float = elapsedSeconds * float(numImages - idx) / float(idx)
                projectedRemainingTime = datetime.timedelta(seconds=round(projectedRemainingSeconds))
                print(f'Elapsed time: {elapsedTime}. Remaining time estimate: {projectedRemainingTime}\n')
    else:
        print("Render directory not empty; skipping render step...\n")
    
    filename = globalConfig.outputDir + "angvel_" + globalConfig.nameTitle + ".csv"
    if delete_old:
        Path(filename).unlink(missing_ok=True)

    if not Path(filename).is_file():
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

        with open(filename,'w', newline='') as csv_file:
            writer=csv.writer(csv_file)
            writer.writerow(keys)
            writer.writerows(zip(*[data[key] for  key in keys]))

        print("\n\n took " + str(time.time()-total_start) + " seconds to complete \n\n")
        print("data saved to: " + filename)
    else:
        print("Quaternion measurements already processed; skipping processing step...\n")

    # ----------------------------
    # Angular velocity estimation
    # ----------------------------

    times = etOriginal + dt * np.linspace(0.0, numImages - 1, numImages)
    T_unrotate: list[npt.NDArray] = []
    T_planet: list[npt.NDArray] = []
    q_planet: list[Quaternion] = []
    omega_exp: list[npt.NDArray] = []
    xForms = spice.sxform('J2000', planet.planetFrame, times[1:-1])
    
    for i in range(len(xForms)):
        _T_planet, _omega_exp = spice.xf2rav(xForms[i])
        _T_unrotate = Quaternion(0.5, 0.5, -0.5, 0.5).to_matrix() @ _T_planet
        # _T_unrotate = _T_planet.T
        # _T_unrotate = np.identity(3)
        T_unrotate.append(_T_unrotate)
        T_planet.append(_T_planet)
        q_planet.append(Quaternion.FromMatrix(_T_planet).positivize())
        omega_exp.append(_omega_exp)

    angvelData = read_quats(filename)
    omega, omega_hist = estimate_omega(angvelData, dt, T_unrotate)

    # _, omega_final = spice.xf2rav(spice.sxform('J2000', planet.planetFrame, etOriginal + dt * numImages))
    # angacc: npt.NDArray = (omega_final - omega_exp) / (dt * numImages)

    # omega = (T_unrotate @ np.array([omega]).T).T[0]
    # angacc = (T_unrotate @ np.array([angacc]).T).T[0]

    # print(f"Angular acceleration = {angacc}")

    est = np.rad2deg(np.linalg.norm(omega))
    exp = np.rad2deg(np.linalg.norm(omega_exp[0])) 

    print(f"Estimated omega (deg/s): {np.round(np.rad2deg(omega), 6)}")
    print(f"Expected omega (deg/s): {np.round(np.rad2deg(omega_exp[0]), 6)}")
    print(f"Estimated magnitude (deg/s): {est:.6f}")
    print(f"Expected magnitude (deg/s): {exp:.6f}")
    
    # Plotting
        
    labels = [f'{globalConfig.nameTitle} Simulation']
    expected = [exp]
    estimated = [est]

    x = np.arange(len(expected))
    width = 0.35

    fig = plt.figure(layout='constrained')
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.bar(x - width/2, expected, width, label='Expected')
    ax1.bar(x + width/2, estimated, width, label='Estimated')

    ax1.set_ylabel('Angular Velocity (deg/s)')
    ax1.set_title('Angular Velocity Estimation Validation')
    ax1.set_xticks(x, labels)
    ax1.legend()

    quats_w = [quat.w for quat in q_planet]
    quats_x = [quat.x for quat in q_planet]
    quats_y = [quat.y for quat in q_planet]
    quats_z = [quat.z for quat in q_planet]
    angvel_x = [angvel[0] for angvel in omega_hist]
    angvel_y = [angvel[1] for angvel in omega_hist]
    angvel_z = [angvel[2] for angvel in omega_hist]
    # angvel_x = [angvel[0] for angvel in omega_exp]
    # angvel_y = [angvel[1] for angvel in omega_exp]
    # angvel_z = [angvel[2] for angvel in omega_exp]

    ax2.plot(np.linspace(0.0, len(q_planet) - 1, len(q_planet)), quats_w, label=r"$q_{w}$")
    ax2.plot(np.linspace(0.0, len(q_planet) - 1, len(q_planet)), quats_x, label=r"$q_{x}$")
    ax2.plot(np.linspace(0.0, len(q_planet) - 1, len(q_planet)), quats_y, label=r"$q_{y}$")
    ax2.plot(np.linspace(0.0, len(q_planet) - 1, len(q_planet)), quats_z, label=r"$q_{z}$")

    # ax2.plot(np.linspace(0.0, len(omega_hist) - 1, len(omega_hist)), angvel_x, label=r"$\omega_{x}$")
    # ax2.plot(np.linspace(0.0, len(omega_hist) - 1, len(omega_hist)), angvel_y, label=r"$\omega_{y}$")
    # ax2.plot(np.linspace(0.0, len(omega_hist) - 1, len(omega_hist)), angvel_z, label=r"$\omega_{z}$")

    ax2.grid()
    ax2.legend()

    # plt.show()

    # ------------------------------------------------
    # Kalman filter for angular velocity estimation
    # ------------------------------------------------

    # angvel_history = [(T_gravity.T @ T_unrotate @ np.array([angvel]).T).T[0] for angvel in omega_hist]
    angvel_history = copy.deepcopy(omega_hist)
    
    mx_0: npt.NDArray = copy.deepcopy(omega_exp[0])
    Pxx_0: npt.NDArray = np.diag(np.array([0.2, 0.2, 1.0])) * 1e-5 ** 2
    Pww: npt.NDArray = np.diag(np.array([0.2, 0.2, 1.0])) * 1e-6 ** 2
    Pvv: npt.NDArray = np.diag(np.array([0.2, 0.2, 1.0])) * 1e-5 ** 2

    Hx: npt.NDArray = np.identity(3)

    mx_plus = copy.deepcopy(mx_0)
    Pxx_plus = copy.deepcopy(Pxx_0)

    mx_history: list[npt.NDArray] = []
    Pxx_history: list[npt.NDArray] = []

    for k in range(len(angvel_history)):
        # mx_minus = mx_plus + dt * angacc
        mx_minus = copy.deepcopy(mx_plus)
        Pxx_minus = Pxx_plus + Pww

        if (np.abs(angvel_history[k] - mx_minus) < 6.0 * np.sqrt(np.diag(Pxx_minus))).all():
            mz_minus = Hx @ mx_minus
            Pxz_minus = Pxx_minus @ Hx.T
            Pzz_minus = Hx @ Pxx_minus @ Hx.T + Pvv
            K = Pxz_minus @ np.linalg.inv(Pzz_minus)
            
            mx_plus = mx_minus + K @ (angvel_history[k] - mz_minus)
            Pxx_plus = Pxx_minus - Pxz_minus @ K.T - K @ Pxz_minus.T + K @ Pzz_minus @ K.T
        else:
            mx_plus = copy.deepcopy(mx_minus)
            Pxx_plus = copy.deepcopy(Pxx_minus)
        
        mx_history.append(mx_plus)
        Pxx_history.append(Pxx_plus)
    
    fig2 = plt.figure(layout='constrained')
    ax3 = fig2.add_subplot(131)
    ax4 = fig2.add_subplot(132)
    ax5 = fig2.add_subplot(133)

    mx_x = [mx[0] for mx in mx_history]
    mx_y = [mx[1] for mx in mx_history]
    mx_z = [mx[2] for mx in mx_history]

    Pxx_x = [Pxx[0, 0] for Pxx in Pxx_history]
    Pxx_y = [Pxx[1, 1] for Pxx in Pxx_history]
    Pxx_z = [Pxx[2, 2] for Pxx in Pxx_history]

    ax3.plot(times[1:-1] - etOriginal, rad_to_arcsec(mx_x - np.asarray(omega_exp)[:, 0]), label=r"$m_{x,\omega_{x}}$", color='blue')
    ax3.plot(times[1:-1] - etOriginal, -rad_to_arcsec(3.0 * np.sqrt(Pxx_x)), linestyle='dashed', color='r', label=r"$P_{xx,\omega_{x}}$")
    ax3.plot(times[1:-1] - etOriginal, rad_to_arcsec(3.0 * np.sqrt(Pxx_x)), linestyle='dashed', color='r')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angular Velocity (arcsec/s)")
    ax3.set_title(r"Error in $m_{x,\omega_{x}}$ over Time")
    ax3.grid()
    ax3.legend()

    ax4.plot(times[1:-1] - etOriginal, rad_to_arcsec(mx_y - np.asarray(omega_exp)[:, 1]), label=r"$m_{x,\omega_{y}}$", color='blue')
    ax4.plot(times[1:-1] - etOriginal, -rad_to_arcsec(3.0 * np.sqrt(Pxx_y)), linestyle='dashed', color='r', label=r"$P_{xx,\omega_{y}}$")
    ax4.plot(times[1:-1] - etOriginal, rad_to_arcsec(3.0 * np.sqrt(Pxx_y)), linestyle='dashed', color='r')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angular Velocity (arcsec/s)")
    ax4.set_title(r"Error in $m_{x,\omega_{y}}$ over Time")
    ax4.grid()
    ax4.legend()
    
    ax5.plot(times[1:-1] - etOriginal, rad_to_arcsec(mx_z - np.asarray(omega_exp)[:, 2]), label=r"$m_{x,\omega_{z}}$", color='blue')
    ax5.plot(times[1:-1] - etOriginal, -rad_to_arcsec(3.0 * np.sqrt(Pxx_z)), linestyle='dashed', color='r', label=r"$P_{xx,\omega_{z}}$")
    ax5.plot(times[1:-1] - etOriginal, rad_to_arcsec(3.0 * np.sqrt(Pxx_z)), linestyle='dashed', color='r')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Angular Velocity (arcsec/s)")
    ax5.set_title(r"Error in $m_{x,\omega_{z}}$ over Time")
    ax5.grid()
    ax5.legend()

    plt.show()
