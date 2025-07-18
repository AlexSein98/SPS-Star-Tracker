import numpy as np
from matplotlib import pyplot as plt
import poppy
import scipy.interpolate as interp


def deg_to_arcsec(angle):
    return angle * 3600.0


def clamp(val, low, high):
    return low if val < low else high if val > high else val


def temp_to_rgb(temp: float) -> tuple[int]:
    temp_100 = 0.01 * temp
    r: float = 0
    g: float = 0
    b: float = 0

    # Calculate red
    if temp_100 <= 66:
        r = 255
    else:
        r = clamp(329.698727446 * ((temp_100 - 60) ** -0.1332047592), 0, 255)
    
    # Calculate green
    if temp_100 <= 66:
        g = clamp(99.4708025861 * np.log(temp_100) - 161.1195681661, 0, 255)
    else:
        g = clamp(288.1221695283 * ((temp_100 - 60) ** -0.0755148492), 0, 255)
    
    # Calculate blue
    if temp_100 >= 66:
        b = 255
    else:
        if temp_100 <= 19:
            b = 0
        else:
            b = clamp(138.5177312231 * np.log(temp_100 - 10) - 305.0447927307, 0, 255)
    
    return clamp(int(r), 0, 255), clamp(int(g), 0, 255), clamp(int(b), 0, 255)


def measure_radial_profile(rr, radialprofile):
    return interp.interp1d(rr, radialprofile, kind='cubic', bounds_error=False)


wavelengths = [610e-9, 560e-9, 480e-9]
rgb = temp_to_rgb(15000)
weights = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]

source = {
    'wavelengths': wavelengths,
    'weights': weights
}

diameter = 0.01  # 10 mm aperture
radius = 0.5 * diameter

U: int = 1024
V: int = 1024
fov_deg = 0.05
pixelscale = deg_to_arcsec(fov_deg) / float(U)

wave_offset = 2
osys = poppy.OpticalSystem("test", oversample=2)
osys.add_pupil( poppy.CircularAperture(radius=radius))    # pupil radius in meters
osys.add_pupil( poppy.ThinLens(nwaves=wave_offset, reference_wavelength=source['wavelengths'][1], radius=radius))
osys.add_detector(pixelscale=pixelscale, fov_arcsec=deg_to_arcsec(fov_deg))

psf = osys.calc_psf(source=source)

plt.subplot(111)
# poppy.display_psf(psf, title='Defocused by {0} waves'.format(wave_offset),
#     colorbar_orientation='horizontal')
poppy.display_profiles(psf)
rr, radialProfile = poppy.radial_profile(psf, binsize=1)  # 1 arcsecond bins?
radialProfileFunc = measure_radial_profile(rr, radialProfile)

print(f'measured_val = {radialProfileFunc(deg_to_arcsec(50.0 / 3600.0))}')
plt.show()
