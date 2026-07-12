from PIL import Image
import tifffile
from tifffile import imwrite

import copy
import numpy as np
from matplotlib import pyplot as plt


thisDir = "./py_src/star/"

# imgs = [thisDir + "data/Apertures/Aperture_Hexagon_Small.png",
#         thisDir + "data/Apertures/Aperture_Hexagon_Medium.png",
#         thisDir + "data/Apertures/Aperture_Hexagon_Large.png"]

imgs = [thisDir + "data/RealLensAperture.png",
        thisDir + "data/RealLensAperture.png",
        thisDir + "data/RealLensAperture.png"]

imgOutputs = [thisDir + 'python/output/fft/Aperture_Circle_Small.tif',
              thisDir + 'python/output/fft/Aperture_Circle_Medium.tif',
              thisDir + 'python/output/fft/Aperture_Circle_Large.tif']

for index in range(len(imgs)):
    img = Image.open(imgs[index])
    img_array = np.array(img.convert('L')) / 255.0

    print(f'Max = {np.max(img_array)}')
    print(f'Min = {np.min(img_array)}')

    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)
    fshift = fshift ** 10

    centerU = float(fshift.shape[0]) / 2.0
    centerV = float(fshift.shape[1]) / 2.0
    extent = 64
    uMin = int(centerU - extent)
    uMax = int(centerU + extent)
    vMin = int(centerV - extent)
    vMax = int(centerV + extent)
    magnitude_spectrum = np.abs(fshift)[uMin:uMax, vMin:vMax] / (fshift.shape[0] * fshift.shape[1])

    gradient_radius: float = 0.5 * float(magnitude_spectrum.shape[0])
    gradient = np.zeros(magnitude_spectrum.shape)
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            gradient[i, j] = 1.0 - np.clip(np.sqrt((float(i) - gradient_radius) ** 2 + 
                                                (float(j) - gradient_radius) ** 2) / gradient_radius, 0.0, 1.0)

    magnitude_spectrum *= gradient
    magnitude_spectrum /= np.linalg.norm(magnitude_spectrum)

    print(f'Max = {np.max(magnitude_spectrum)}')
    print(f'Min = {np.min(magnitude_spectrum)}')

    # tifffile.imwrite(thisDir + 'python/output/fft/Aperture_Hexagon_Small.tif', magnitude_spectrum.astype(np.float32), compression=0)
    # tifffile.imwrite(thisDir + 'python/output/fft/Aperture_Hexagon_Medium.tif', magnitude_spectrum.astype(np.float32), compression=0)
    tifffile.imwrite(imgOutputs[index], magnitude_spectrum.astype(np.float32), compression=0)
