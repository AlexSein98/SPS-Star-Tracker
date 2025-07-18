import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


thisDir = ".\\py_src\\star\\"

img = Image.open(thisDir + "data\\Aperture_Rectangle_Imperfect.png")
img_array = np.array(img.convert('L'))
f = np.fft.fft2(img_array)
fshift = np.fft.fftshift(f)

centerU = float(fshift.shape[0]) / 2.0
centerV = float(fshift.shape[1]) / 2.0
extent = 96
uMin = int(centerU - extent)
uMax = int(centerU + extent)
vMin = int(centerV - extent)
vMax = int(centerV + extent)
magnitude_spectrum = np.abs(fshift)[uMin:uMax, vMin:vMax]  # [448:576, 448:576]
plt.imsave(thisDir + 'python\\output\\aperture_fft.jpg', magnitude_spectrum, cmap='gray')
