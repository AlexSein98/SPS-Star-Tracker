import numpy as np

import sys
import csv

from matplotlib import pyplot as plt
import spiceypy as spice


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


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def mult(self, other):
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
                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    
    def normalize(self):
        mag_1 = 1.0 / np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        return Quaternion(self.w * mag_1, self.x * mag_1, self.y * mag_1, self.z * mag_1)
    
    def conjugate(self):
        return Quaternion(-self.w, self.x, self.y, self.z)
    
    def as_w_first_array(self):
        return np.array([self.w, self.x, self.y, self.z])
    
    def as_w_last_array(self):
        return np.array([self.x, self.y, self.z, self.w])


def matrix_to_angleaxis(m: np.ndarray[float]):
    # Angle in radians
    phi_r = np.arccos(0.5 * (np.trace(m) - 1.0))

    # Axis components
    sin_phi_2 = 0.5 / np.sin(phi_r)
    e0 = (m[1][2] - m[2][1]) * sin_phi_2
    e1 = (m[2][0] - m[0][2]) * sin_phi_2
    e2 = (m[0][1] - m[1][0]) * sin_phi_2

    # Return angle (in degrees) + axis
    phi_d = np.rad2deg(phi_r)
    e = np.array([e0, e1, e2])
    return phi_d, e


def deg_to_arcsec(angle):
    return angle * 3600.0


def arcsec_to_rad(angle):
    return np.deg2rad(angle / 3600.0)


def plot_errors(truthDataPath: str, estDataPath: str, reject: float=0, planetRadius: float=6378136.3):
    truthData = read_csv(truthDataPath)
    estData = read_csv(estDataPath, ignore=[0, 1, 2, 3], hasHeader=True)
    
    # Very basic error handling if datasets are not the same length
    if len(truthData) != len(estData):
        return

    # Compare data
    errorsArcsec = []
    for i in range(len(truthData)):
        truth_i = truthData[i]
        est_i = estData[i]
        
        if est_i[0] == 999 or est_i[1] == 999 or est_i[2] == 999 or est_i[3] == 999:
            continue
        
        q_real = Quaternion(truth_i[0], truth_i[1], truth_i[2], truth_i[3]).normalize()
        q_est = Quaternion(est_i[0], est_i[1], est_i[2], est_i[3]).normalize()
        q_err = q_est.conjugate().mult(q_real).normalize()
        m = spice.q2m(q_err.as_w_first_array())
        phi_d, _ = matrix_to_angleaxis(m)
        phi_arcsec = deg_to_arcsec(phi_d)
        
        if reject != 0 and phi_arcsec > reject:
            continue
        errorsArcsec.append(phi_arcsec)
    
    # Calculate statistics
    mean = np.mean(errorsArcsec)
    median = np.median(errorsArcsec)
    std = np.std(errorsArcsec)
    projectedSPSMeanError = arcsec_to_rad(mean) * planetRadius

    # Print statistics
    print(f'Mean                        = {round(mean, 3)} arcseconds')
    print(f'Median                      = {round(median, 3)} arcseconds')
    print(f'Standard Deviation          = {round(std, 3)} arcseconds')
    print(f'Projected SPS Mean Error    = {round(projectedSPSMeanError, 1)} meters')

    # Plot histogram and box plot of errors
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.hist(errorsArcsec, bins=200, color='skyblue', edgecolor='black')
    ax2.boxplot(errorsArcsec, orientation='horizontal', showfliers=False)

    # Nice plot stuff
    extraPrint = f' (errors > {reject} arcseconds rejected)'
    if reject == 0:
        extraPrint = f' (no errors rejected)'
    ax1.set_title(f'Attitude Error Counts for {len(errorsArcsec)}/{len(truthData)} Samples' + extraPrint)
    ax1.set_xlabel('Error (arcseconds)')
    ax1.set_ylabel('Count')
    ax1.grid()
    ax2.set_title(f'Attitude Error Statistics for {len(errorsArcsec)}/{len(truthData)} Samples' + extraPrint)
    ax2.set_xlabel('Error (arcseconds)')
    ax2.grid()

    # Show plot
    plt.show()


if __name__ == "__main__":
    rEarth = 6378136.3
    # rMoon = 1737400.0
    # rMars = 3389500.0
    # rPhobos = 11000.0

    # Star tracker "measurements" file
    n = len(sys.argv)
    measurements = ".\\output.csv"
    if n > 1:
        measurements = sys.argv[1]
    
    # Truth source directory
    truthSourceDir = ".\\py_src\\star\\"
    if n > 2:
        truthSourceDir = sys.argv[2]
    
    # Cutoff for "good measurements" in arcseconds
    reject = 0.0
    if n > 3:
        reject = float(sys.argv[3])
    
    # Planet radius for evaluating location error
    radius = 0.0
    if n > 4:
        radius = float(sys.argv[4])
    else:
        radius = rEarth

    print(f'{truthSourceDir + "truth_data.csv"}')
    plot_errors(truthSourceDir + "truth_data.csv", measurements, reject=reject, planetRadius=radius)
