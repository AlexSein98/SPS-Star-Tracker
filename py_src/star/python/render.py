from catalog import *
from planet import *

import cv2
import shutil
import os
import sys

# import poppy
import scipy.interpolate as interp
import scipy.signal as signal
import multiprocessing


class StarParams:
    def __init__(self, rightAscension, declination, pmRA, pmDE, parallax, pHat, qHat):
        self.rightAscension = rightAscension
        self.declination = declination
        self.pmRA = pmRA
        self.pmDE = pmDE
        self.parallax = parallax
        self.pHat = pHat
        self.qHat = qHat


class RenderParams:
    def __init__(self, etJ2000: float, etNow: float, cameraPos: np.ndarray[float], U: int, V: int, fovU: float, pixelRadius: float, catalog, relativeMagnitude: float, relativeFlux: float):
        self.etJ2000 = etJ2000
        self.etNow = etNow
        self.cameraPos = cameraPos
        self.U = U
        self.V = V
        self.fovU = fovU
        self.pixelRadius = pixelRadius
        self.catalog = catalog
        self.relativeMagnitude = relativeMagnitude
        self.relativeFlux = relativeFlux


def diffraction_limit(wavelength: float, apertureDiameter: float):
    # Wavelength and aperture diameter must be in the same units
    return 1.22 * wavelength / apertureDiameter


def point_spread_function_laplacian(baselineIntensity: float, nominalIntensity: float, r: float, w: float):
    intensity = baselineIntensity + nominalIntensity * np.exp(-r / w)
    return intensity


def falloff_laplacian(baselineIntensity: float, nominalIntensity: float, intensity_cutoff: float, w: float):
    return w * np.log(nominalIntensity / (intensity_cutoff - baselineIntensity))


def point_spread_function_gaussian(baselineIntensity: float, nominalIntensity: float, r2: float, sigma: float):
    intensity = baselineIntensity + nominalIntensity * np.exp(-0.5 * r2 / (sigma ** 2))
    return intensity


def falloff_gaussian(baselineIntensity: float, nominalIntensity: float, intensity_cutoff: float, sigma: float):
    logArg = nominalIntensity / (intensity_cutoff - baselineIntensity)
    if logArg < 1.0:
        return 1.0
    else:
        return sigma * np.sqrt(2.0 * np.log(logArg))


def measure_radial_profile(rr, radialprofile):
    return interp.interp1d(rr, radialprofile, kind='cubic', bounds_error=False)


def star_pointing(starParams: StarParams, r: np.ndarray[float], etNow, etJ2000):
    starPointingNominal = r_hat(starParams.rightAscension, starParams.declination)
    starPointingPerturbed = starPointingNominal + (sec_to_year(etNow - etJ2000)) * (arcsec_to_rad(starParams.pmRA) * starParams.pHat + arcsec_to_rad(starParams.pmDE) * starParams.qHat) - arcsec_to_rad(starParams.parallax) * r / AU
    return starPointingPerturbed


def get_stars_in_frame(catalog: np.ndarray[float], cameraAttitudeICRF: np.ndarray[float], fovX: float, fovY: float, toleranceDegrees: float = 1.0) -> list[int]:
    pointing = cameraAttitudeICRF[:, 0]
    fovMaxCos = np.cos(np.deg2rad(np.sqrt(fovX ** 2 + fovY ** 2) + toleranceDegrees))

    # Catalog: [ 0   1    2     3      4         5, 6, 7    8, 9, 10    11    12    13   14   15 ]
    #          [ RA  Dec  pmRA  pmDec  parallax  pHat       qHat        vMag  temp  r    g    b  ]

    validIndices: list[int] = []
    for i in range(len(catalog)):
        if fovMaxCos <= np.dot(pointing, r_hat(catalog[i][0], catalog[i][1])):
            validIndices.append(i)
    return validIndices


def get_planets_in_frame(cameraPos: np.ndarray[float], cameraAttitudeICRF: np.ndarray[float], et: float, fovX: float, fovY: float, correctionMode: int, toleranceDegrees: float = 1.0):
    pointing = cameraAttitudeICRF[:, 0]
    fovMaxCos = np.cos(np.deg2rad(np.sqrt(fovX ** 2 + fovY ** 2) + toleranceDegrees))

    planetColors = []
    planetsList: list[str] = []
    for planet in allPlanets:
        posPlanet, _ = get_planet_pos(planet, cameraPos, et, correctionMode)
        if fovMaxCos <= np.dot(pointing, normalize(posPlanet)):
            planetsList.append(planet)
            planetColors.append(planet_color(planet))
    return planetsList, planetColors


def sample_color_for_planets(i, u, v, centerU, centerV, mode, planetName, planetCoordsUV, fluxPlanet, diffractionLimitPx, planetColors, 
                             pos_PlanetSSB, normal_SunPlanet, rot_PlanetSSB, cameraPos, T_ICRFCamera, fieldOfViewU, fieldOfViewV, dimU, dimV):
    uc = u - centerU
    vc = v - centerV

    if mode == 0:
        # Render planet from Gaussian approximation of point spread function
        pixelDistanceSquared = (planetCoordsUV[0] - uc) ** 2 + (planetCoordsUV[1] - vc) ** 2
        intensity = fluxPlanet * point_spread_function_gaussian(0.0, 1.0, pixelDistanceSquared, diffractionLimitPx)
        return intensity * planetColors[i]
    elif mode == 1:
        # Render planet directly
        sampleColor, sampleNormal = sample_planet(planetName, pos_PlanetSSB, rot_PlanetSSB, cameraPos, T_ICRFCamera, fieldOfViewU, fieldOfViewV, dimU, dimV, uc, vc, numSamplePoints=4)
        lightingMagnitude = get_sphere_lighting(sampleNormal, normal_SunPlanet)
        return np.flip(sampleColor) * lightingMagnitude * fluxPlanet * np.pi
    else:
        return np.zeros(3)


def bloom(img: np.ndarray[float]):
    imgOut = copy.deepcopy(img)
    imgClip = np.clip(img, 0, 255)
    levels = np.arange(3, 315, 24)
    for level in levels:
        blur = cv2.GaussianBlur(img, (level, level), 0)
        imgOut += blur / level ** 2
    return imgOut


def planet_solar_flux_to_mag(planetName: str, et: float, correctionMode: int=1, ev_1AU: float=-15.3):
    pos_PlanetSSB, _ = get_planet_pos(planetName, np.zeros(3), et, correctionMode)
    dist = np.linalg.norm(pos_PlanetSSB) / AU
    flux_frac = 1.0 / (dist ** 2)
    return flux_to_magnitude(flux_frac, ev_1AU, 1.0)


def render(ra: float, de: float, renderParams: RenderParams, home_dir: str):
    # Image dimensions
    dimU: int = renderParams.U
    dimV: int = renderParams.V
    centerU: float = 0.5 * dimU
    centerV: float = 0.5 * dimV
    img_array = np.zeros((dimV, dimU, 3))

    # Camera internal K-matrix (perfect, no distortion)
    aspectRatio = float(dimV) / float(dimU)
    fieldOfViewU = renderParams.fovU
    fieldOfViewV = fieldOfViewU * aspectRatio

    # Stars visible in-frame
    T_ICRFCamera = ra_dec_to_rot(ra, de)
    catalog = renderParams.catalog
    starsInFrame = get_stars_in_frame(catalog, T_ICRFCamera, fieldOfViewU, fieldOfViewV, toleranceDegrees=1.0)
    catalog_subset = subset_catalog(catalog, starsInFrame)

    # Star UV coordinates
    star_coords = []
    etJ2000 = renderParams.etJ2000
    etNow = renderParams.etNow
    cameraPos = renderParams.cameraPos

    # diffractionLimitRad = diffraction_limit(560e-9, 0.005)  # diffraction for perfectly-focused green light hitting a Canon EOS Rebel T7 sensor (22.3 mm width)
    diffractionLimitRad = diffraction_limit(560e-9, 0.005)  # diffraction for perfectly-focused green light hitting a Canon EOS Rebel T7 sensor (22.3 mm width)
    diffractionLimitPx = rad_to_pixel(diffractionLimitRad, fieldOfViewU, dimU)

    for i in range(len(catalog_subset)):
        rightAscension = catalog_subset[i][0]
        declination = catalog_subset[i][1]
        pmRA = catalog_subset[i][2]
        pmDE = catalog_subset[i][3]
        parallax = catalog_subset[i][4]
        pHat = np.array([catalog_subset[i][5], catalog_subset[i][6], catalog_subset[i][7]])
        qHat = np.array([catalog_subset[i][8], catalog_subset[i][9], catalog_subset[i][10]])
        starPointingPerturbed = star_pointing(StarParams(rightAscension, declination, pmRA, pmDE, parallax, pHat, qHat), cameraPos, etNow, etJ2000)
        starPointingUV = camera_to_uv_centered(fieldOfViewU, fieldOfViewV, dimU, dimV, world_to_camera(T_ICRFCamera.T, starPointingPerturbed))
        star_coords.append(starPointingUV)

    # Render stars
    relativeMagnitude = renderParams.relativeMagnitude
    relativeFlux = renderParams.relativeFlux
    for i in range(len(catalog_subset)):
        starPointingUV = star_coords[i]
        flux = magnitude_to_flux(catalog_subset[i][11], relativeMagnitude, relativeFlux)

        falloff = falloff_gaussian(0.0, flux, 1.0 / 255.0, diffractionLimitPx)
        pixelRadius = max(np.ceil(10.0 * falloff), renderParams.pixelRadius)
        pixelLimitsU = [clamp(int(starPointingUV[0] + centerU - pixelRadius), 0, dimU - 1), clamp(int(starPointingUV[0] + centerU + pixelRadius), 0, dimU - 1)]
        pixelLimitsV = [clamp(int(starPointingUV[1] + centerV - pixelRadius), 0, dimV - 1), clamp(int(starPointingUV[1] + centerV + pixelRadius), 0, dimV - 1)]
        
        for u in range(pixelLimitsU[0], pixelLimitsU[1]):
            for v in range(pixelLimitsV[0], pixelLimitsV[1]):
                uc = u - centerU
                vc = v - centerV

                pixelDistanceSquared = (starPointingUV[0] - uc) ** 2 + (starPointingUV[1] - vc) ** 2
                intensity = flux * point_spread_function_gaussian(0.0, 1.0, pixelDistanceSquared, diffractionLimitPx)
                img_array[v][u] += intensity * np.array([catalog_subset[i][13], catalog_subset[i][14], catalog_subset[i][15]])

    # Render planets
    planetsInView, planetColors = get_planets_in_frame(cameraPos, T_ICRFCamera, etNow, fieldOfViewU, fieldOfViewV, correctionMode=0, toleranceDegrees=1.0)
    if len(planetsInView) > 0.0:
        img_planet = np.zeros((dimV, dimU, 3))
        for i in range(len(planetsInView)):
            planetName = planetsInView[i]
            planetPointing, _ = get_planet_pos(planetName, cameraPos, etNow, correctionMode=0)
            planetPointing = normalize(planetPointing)

            # Handle how to render the planet
            mode = 0  # 0 is Gaussian, 1 is raytraced
            planetRadiusPx = planet_radius_pixels(planetName, cameraPos, etNow, fieldOfViewU, dimU, correctionMode=0)
            if planetRadiusPx > 2.0 * diffractionLimitPx:  # something something Nyquist frequency? Idk?
                mode = 1
            
            print(f'planetRadiusPx = {round(planetRadiusPx, 2)}, while 2*diffractionLimitPx = {round(2.0 * diffractionLimitPx, 2)}; choosing mode {mode} ({"Gaussian diffraction" if mode == 0 else "raytracing"})')
            
            planetCoordsUV = camera_to_uv_centered(fieldOfViewU, fieldOfViewV, dimU, dimV, world_to_camera(T_ICRFCamera.T, planetPointing))
            magPlanet = planet_magnitude(planetName, cameraPos, etNow, correctionMode=0)
            fluxPlanet = magnitude_to_flux(magPlanet, relativeMagnitude, relativeFlux)
            
            if mode == 0:
                falloff = falloff_gaussian(0.0, fluxPlanet, 1.0 / 255.0, diffractionLimitPx)
                pixelRadius = max(np.ceil(10.0 * falloff), renderParams.pixelRadius)
            elif mode == 1:
                pixelRadius = np.ceil(1.2 * planetRadiusPx)
            
            pixelLimitsU = [clamp(int(planetCoordsUV[0] + centerU - pixelRadius), 0, dimU - 1), clamp(int(planetCoordsUV[0] + centerU + pixelRadius), 0, dimU - 1)]
            pixelLimitsV = [clamp(int(planetCoordsUV[1] + centerV - pixelRadius), 0, dimV - 1), clamp(int(planetCoordsUV[1] + centerV + pixelRadius), 0, dimV - 1)]
            
            # Let's not sample from SPICE within the loop so we can eventually multithread it
            pos_PlanetSSB, _ = get_planet_pos(planetName, np.zeros(3), etNow, correctionMode=0)  # planet position wrt SSB
            pos_SunPlanet, _ = get_planet_pos("SUN", pos_PlanetSSB, etNow, correctionMode=0)  # Sun position wrt planet
            rot_PlanetSSB = spice.pxform("J2000", planetBodyFrames[planetName], etNow)
            normal_SunPlanet = normalize(pos_SunPlanet)

            for u in range(pixelLimitsU[0], pixelLimitsU[1]):
                for v in range(pixelLimitsV[0], pixelLimitsV[1]):
                    fluxPlanet = 0.5 * magnitude_to_flux(planet_solar_flux_to_mag(planetName, etNow, correctionMode=0), relativeMagnitude, relativeFlux)
                    img_planet[v][u] += sample_color_for_planets(i, u, v, centerU, centerV, mode, planetName, planetCoordsUV, fluxPlanet, diffractionLimitPx, planetColors, 
                                                                pos_PlanetSSB, normal_SunPlanet, rot_PlanetSSB, cameraPos, T_ICRFCamera, fieldOfViewU, fieldOfViewV, dimU, dimV)
        img_array += bloom(img_planet)

    # Save image
    img = cv2.cvtColor(np.clip(img_array, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    destination = home_dir + 'python\\output\\StarRender_ra_' + str(round(ra, 0)) + '_de_' + str(round(de, 0)) + '.png'
    cv2.imwrite(destination, img)


def get_true_attitude(ra: float, de: float):
    T_ICRFCamera = ra_dec_to_rot(ra, de)
    return spice.m2q(T_ICRFCamera)


def write_csv(filepath: str, data: list[np.ndarray[float]]):
    with open(filepath, "w") as dataCSV:
        writer = csv.writer(dataCSV, delimiter=',', quotechar='"', lineterminator='\n')
        for line in data:
            writer.writerow(line)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # Process arguments
    n = len(sys.argv)
    home = ".\\py_src\\star\\"
    if n > 1:
        home = sys.argv[1]

    # Delete all old images
    dir_path = home + "python\\output"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(os.path.join(home + "python", "output"))

    # Lists of right ascensions and declinations to render
    # step = 15.0
    # minDec = 45.0
    # rightAscensions = np.arange(0.0, 360.0, step)
    # declinations = np.arange(-minDec, minDec + step, step)
    rightAscensions = []
    declinations = []
    if n > 3:
        rightAscensions.append(float(sys.argv[2]))
        declinations.append(float(sys.argv[3]))
    else:
        rightAscensions.append(60.0)
        declinations.append(15.0)

    # Get full star catalog
    catalog = read_csv_catalog(home + "data\\catalog.csv")

    # Star rendering parameters
    relativeMagnitude = 6.0  # "full" exposure is set for stars of this magnitude
    relativeFlux = 1.0  # flux for fully-exposed stars
    starMaxPixelRadius = 16

    # Astrophysical parameters
    spice.furnsh(home + "data\\metakernel.txt")
    tJ2000 = '2000 Jan 1, 00:00:00 UTC'
    tNow = '2025 July 4, 00:00:00 UTC'
    etJ2000 = spice.str2et(tJ2000)
    etNow = spice.str2et(tNow)
    # cameraPos, _ = spice.spkezp(399, etNow, "J2000", "NONE", 0)  # Camera position wrt the solar system barycenter (SSB)
    cameraPos1, _ = spice.spkezp(399, etNow, "J2000", "NONE", 0)  # Camera position wrt the solar system barycenter (SSB)
    cameraPos2, _ = spice.spkezp(301, etNow, "J2000", "NONE", 0)  # Camera position wrt the solar system barycenter (SSB)
    cameraPos = 0.5 * (cameraPos1 + cameraPos2)  # Position camera halfway between Earth and the Moon

    # rightAscensions = []
    # declinations = []
    # associatedPlanets = []
    # for i in range(len(allPlanets)):
    #     # if allPlanets[i] != "EARTH":
    #     if True:
    #         ra, de = get_planet_angular_pos(allPlanets[i], cameraPos, etNow, correctionMode=0)
    #         rightAscensions.append(ra)
    #         declinations.append(de)
    #         associatedPlanets.append(allPlanets[i])

    # Camera parameters
    dimU: int = 1024
    dimV: int = 1024
    fovU: float = 20.0

    # Render
    idx = 0
    numImages = len(rightAscensions) * len(declinations)
    true_data = []
    for ra in rightAscensions:
        for de in declinations:
            idx += 1
            print(f'Rendering image {idx} of {numImages} ({round(100.0 * float(idx) / numImages, 2)}%): RA = {round(ra, 1)}, Dec = {round(de, 1)}')
            params = RenderParams(etJ2000, etNow, cameraPos, dimU, dimV, fovU, starMaxPixelRadius, catalog, relativeMagnitude, relativeFlux)
            
            # Actually do the render
            render(ra, de, params, home)
            true_data.append(get_true_attitude(ra, de))
    
    # for i in range(len(rightAscensions)):
    #     ra = rightAscensions[i]
    #     de = declinations[i]
    #     idx += 1
    #     planetPos, _ = get_planet_pos(associatedPlanets[i], cameraPos, etNow, correctionMode=0)
    #     print(f'Planet position = {planetPos}')
    #     fovU = 4.0 * np.rad2deg(planetRadii[associatedPlanets[i]] / np.linalg.norm(planetPos))
    #     relativeMagnitude = planet_solar_flux_to_mag(associatedPlanets[i], etNow, correctionMode=0)
    #     print(f'Rendering image {idx} of {len(rightAscensions)} ({round(100.0 * float(idx) / len(rightAscensions), 2)}%):\n    FOV = {round(fovU, 4)} deg\n    RA = {round(ra, 1)}\n    Dec = {round(de, 1)}\n    Focus planet = {associatedPlanets[i]}')
    #     render(ra, de, RenderParams(etJ2000, etNow, cameraPos, dimU, dimV, fovU, starMaxPixelRadius, catalog, relativeMagnitude, relativeFlux))
    #     true_data.append(get_true_attitude(ra, de))
    
    # Write output
    write_csv(home + "truth_data.csv", true_data)
