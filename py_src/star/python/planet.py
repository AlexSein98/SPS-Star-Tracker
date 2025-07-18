from transformations import *
import spiceypy as spice
import copy
import random
import sys

import cv2


c = 299792.458  # km/s
c_1 = 1.0 / c  # store 1/c for faster division
AU = 1.495978707e8  # km

seed: int = 100
random.seed(seed)


# Source: https://en.wikipedia.org/wiki/Absolute_magnitude#Solar_System_bodies_(H), via the Astronomical Almanac
planetAbsoluteMagnitudes = {
    "MERCURY": -0.613,
    "VENUS": -4.384,
    "EARTH": -3.99,
    "MOON": 0.28,
    "MARS": -1.601,
    "JUPITER": -9.395,
    "SATURN": -8.914,
    "URANUS": -7.110,
    "NEPTUNE": -7.00
}

planetRadii = {
    "MERCURY": 2439.7,
    "VENUS": 6051.8,
    "EARTH": 6378.137,
    "MOON": 1737.4,
    "MARS": 3389.5,
    "JUPITER": 69911.0,
    "SATURN": 58232.0,
    "URANUS": 25362.0,
    "NEPTUNE": 24622.0
}

allPlanets = ["MERCURY", "VENUS", "EARTH", "MOON", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]

planetBodyFrames = {
    "MERCURY": "IAU_MERCURY",
    "VENUS": "IAU_VENUS",
    "EARTH": "ITRF93",
    "MOON": "MOON_ME",
    "MARS": "IAU_MARS",
    "JUPITER": "IAU_JUPITER",
    "SATURN": "IAU_SATURN",
    "URANUS": "IAU_URANUS",
    "NEPTUNE": "IAU_NEPTUNE"
}

planetTextures: dict = {}


def get_planet_pos(planetName: str, pos_ObsSSB: np.ndarray[float], et: float, correctionMode: int=1):
    if correctionMode == 0:
        pos_SSB, _ = spice.spkezp(spice.bodn2c(planetName), et, "J2000", "NONE", 0)
        return (pos_SSB - pos_ObsSSB, 0)
    elif correctionMode == 1:
        lt_0 = 0
        lt_i = 1
        pos = np.zeros(3)
        while abs(lt_0 - lt_i) > 1e-9:
            lt_0 = copy.deepcopy(lt_i)
            pos_SSB, _ = spice.spkezp(spice.bodn2c(planetName), et - lt_0, "J2000", "NONE", 0)
            pos = pos_SSB - pos_ObsSSB
            lt_i = np.linalg.norm(pos) * c_1
        return (pos, lt_i)
    else:
        print(f'Unsupported correction mode {correctionMode}; defaulting to 0 (no aberration correction).')
        pos_SSB, _ = spice.spkezp(spice.bodn2c(planetName), et, "J2000", "NONE", 0)
        return (pos_SSB - pos_ObsSSB, 0)


def get_planet_angular_pos(planetName: str, pos_ObsSSB: np.ndarray[float], et: float, correctionMode: int=1):
    pos, _ = get_planet_pos(planetName, pos_ObsSSB, et, correctionMode)
    ra, dec = r_hat_to_ra_dec(normalize(pos))
    return ra, dec


def default_phase_integral(phaseAngle: float):
    # Sources: 
    #   https://adsabs.harvard.edu/full/seri/Obs../0030//0000097.000.html
    #   https://en.wikipedia.org/wiki/Absolute_magnitude#Solar_System_bodies_(H)
    return -2.5 * np.log10(2.0 / 3.0 * ((1.0 - phaseAngle / 180.0) * np.cos(np.deg2rad(phaseAngle)) + 1.0 / np.pi * np.sin(np.deg2rad(phaseAngle))))


def planet_magnitude(planetName: str, pos_ObsSSB: np.ndarray[float], et: float, correctionMode: int=1):
    # Source: https://en.wikipedia.org/wiki/Absolute_magnitude#Solar_System_bodies_(H)
    
    pos_PlanetSSB, _ = get_planet_pos(planetName, np.zeros(3), et, correctionMode)  # planet position wrt SSB
    pos_PlanetObs, _ = get_planet_pos(planetName, pos_ObsSSB, et, correctionMode)  # planet position wrt observer
    pos_SunPlanet, _ = get_planet_pos("SUN", pos_PlanetSSB, et, correctionMode)  # Sun position wrt planet
    d_PlanetObs = np.linalg.norm(pos_PlanetObs)
    d_SunPlanet = np.linalg.norm(pos_SunPlanet)

    phaseAngle = np.rad2deg(np.arccos(np.dot(normalize(-pos_PlanetObs), normalize(pos_SunPlanet))))

    phaseIntegral: float = default_phase_integral(phaseAngle)  # phase integral
    H: float = planetAbsoluteMagnitudes[planetName]
    if planetName == "MERCURY":
        phaseIntegral = 6.328e-2 * phaseAngle - 1.6336e-3 * phaseAngle ** 2 + 3.3644e-5 * phaseAngle ** 3 - 3.4265e-7 * phaseAngle ** 4 + 1.6893e-9 * phaseAngle ** 5 - 3.0334e-12 * phaseAngle ** 6
    elif planetName == "VENUS":
        if 0.0 < phaseAngle <= 163.7:
            phaseIntegral = -1.044e-3 * phaseAngle + 3.687e-4 * phaseAngle ** 2 - 2.814e-6 * phaseAngle ** 3 + 8.938e-9 * phaseAngle ** 4
        elif 163.7 < phaseAngle < 179.0:
            phaseIntegral = 240.44228 - 2.81914 * phaseAngle + 8.39034e-3 * phaseAngle ** 2
    elif planetName == "EARTH":
        phaseIntegral = -1.060e-3 * phaseAngle + 2.054e-4 * phaseAngle ** 2
    elif planetName == "MOON":
        if phaseAngle <= 150.0:
            # Assuming observer is on or near the Earth and is seeing the Moon's near side; otherwise, should probably disregard the Moon altogether due to potential inaccuracies?
            firstQuarter = np.dot(normalize(np.cross(pos_PlanetObs, -pos_ObsSSB)), np.array([0, 0, 1])) > 0.0
            if firstQuarter:
                phaseIntegral = 2.9994e-2 * phaseAngle - 1.6057e-4 * phaseAngle ** 2 + 3.1543e-6 * phaseAngle ** 3 - 2.0667e-8 * phaseAngle ** 4 + 6.2553e-11 * phaseAngle ** 5
            else:
                phaseIntegral = 3.3234e-2 * phaseAngle - 3.0725e-4 * phaseAngle ** 2 + 6.1575e-6 * phaseAngle ** 3 - 4.7723e-8 * phaseAngle ** 4 + 1.4681e-10 * phaseAngle ** 5
    elif planetName == "MARS":
        if phaseAngle <= 50.0:
            phaseIntegral = 2.267e-2 * phaseAngle - 1.302e-4 * phaseAngle ** 2
        elif 50.0 < phaseAngle <= 120.0:
            phaseIntegral = 1.234 - 2.573e-2 * phaseAngle + 3.445e-4 * phaseAngle ** 2
    elif planetName == "JUPITER":
        if phaseAngle <= 12.0:
            phaseIntegral = -3.7e-4 * phaseAngle + 6.16e-4 * phaseAngle ** 2
        else:
            phaseIntegral = -0.033 - 2.5 * np.log10(1.0 - 1.507 * (phaseAngle / 180.0) - 0.363 * (phaseAngle / 180.0) ** 2 - 0.062 * (phaseAngle / 180.0) ** 3 + 2.809 * (phaseAngle / 180.0) ** 4 - 1.876 * (phaseAngle / 180.0) ** 5)
    elif planetName == "SATURN":
        saturnUpAxis = spice.pxform("J2000", "IAU_SATURN", et)[:, 2]
        ringAngle: float = abs(90.0 - np.rad2deg(np.arccos(np.dot(-normalize(pos_PlanetObs), saturnUpAxis))))
        if phaseAngle < 6.5 and ringAngle < 27.0:
            phaseIntegral = -1.825 * np.sin(np.deg2rad(ringAngle)) + 2.6e-2 * phaseAngle - 0.378 * np.sin(np.deg2rad(ringAngle)) * np.exp(-2.25 * phaseAngle)
        elif phaseAngle <= 6.0:
            phaseIntegral = -0.036 - 3.7e-4 * phaseAngle + 6.16e-4 * phaseAngle ** 2
        elif 6.0 < phaseAngle < 150.0:
            phaseIntegral = 0.026 + 2.446e-4 * phaseAngle + 2.672e-4 * phaseAngle ** 2 - 1.505e-6 * phaseAngle ** 3 + 4.767e-9 * phaseAngle ** 4
    elif planetName == "URANUS":
        if phaseAngle < 3.1:
            uranusUpAxis = spice.pxform("J2000", "IAU_SATURN", et)[:, 2]
            fUranus = 0.0022927  # flattening coefficient
            phiView = 90.0 - np.rad2deg(np.arccos(np.dot(-normalize(pos_PlanetObs), uranusUpAxis)))
            phiS = 90.0 - np.rad2deg(np.arccos(np.dot(normalize(pos_SunPlanet), uranusUpAxis)))
            phiPrime = 0.5 * (abs(np.arctan(np.tan(phiView) / (1.0 - fUranus ** 2))) + abs(np.arctan(np.tan(phiS) / (1.0 - fUranus ** 2))))
            phaseIntegral = -8.4e-4 * phiPrime + 6.587e-3 * phaseAngle + 1.045e-4 * phaseAngle ** 2
    elif planetName == "NEPTUNE":
        if phaseAngle < 133.0 and et > -43135.816087188054:  # ephemeris time for J2000
            phaseIntegral = 7.944e-3 * phaseAngle + 9.617e-5 * phaseAngle ** 2

    m = H + 5.0 * np.log10(d_PlanetObs * d_SunPlanet / (AU ** 2)) + phaseIntegral
    return m


def hex_to_rgb(h: str):
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])


def planet_color(planetName: str):
    if planetName == "MERCURY":
        return hex_to_rgb("1a1a1a")
    elif planetName == "VENUS":
        return hex_to_rgb("e6e6e6")
    elif planetName == "EARTH":
        return hex_to_rgb("2f6a69")
    elif planetName == "MOON":
        return hex_to_rgb("1a1a1a")
    elif planetName == "MARS":
        return hex_to_rgb("993d00")
    elif planetName == "JUPITER":
        return hex_to_rgb("b07f35")
    elif planetName == "SATURN":
        return hex_to_rgb("b08f36")
    elif planetName == "URANUS":
        return hex_to_rgb("5580aa")
    elif planetName == "NEPTUNE":
        return hex_to_rgb("366896")
    else:
        return hex_to_rgb("00ff00")


def planet_radius_pixels(planetName: str, pos_ObsSSB: np.ndarray[float], et: float, fieldOfViewU: float, U: int, correctionMode: int=1):
    pos, _ = get_planet_pos(planetName, pos_ObsSSB, et, correctionMode)
    radiusRad = planetRadii[planetName] / np.linalg.norm(pos)
    radiusPixel = rad_to_pixel(radiusRad, fieldOfViewU, U)
    return radiusPixel


def solveQuadratic(a: float, b: float, c: float):
    discr: float = b * b - 4 * a * c
    print(f'discriminant = {discr}')
    x0 = 0.0
    x1 = 0.0
    if (discr < 0):
        return x0, x1, False
    elif (discr == 0):
        x0 = -0.5 * b / a
        x1 = -0.5 * b / a
    else:
        q: float = -0.5 * (b + np.sqrt(discr)) if (b > 0) else -0.5 * (b - np.sqrt(discr))
        x0 = q / a
        x1 = c / q
    
    if (x0 > x1):
        return x1, x0, True
    else:
        return x0, x1, True


def ray_sphere_intersection(origin: np.ndarray[float], dir: np.ndarray[float], center: np.ndarray[float], R: float):
    L = origin - center
    a: float = np.dot(dir, dir)
    b: float = 2.0 * np.dot(dir, L)
    c: float = np.dot(L, L) - R ** 2
    t0, t1, success = solveQuadratic(a, b, c)

    # print(f'origin = {origin}')
    # print(f'center = {center}')
    print(f'dir = {dir}')
    print(f'L = {normalize(L)}')
    print(f'R = {R}')
    print(f't0 = {t0}')
    if not success:
        return np.zeros(3), False
    else:
        return origin + t0 * dir, True


def ray_sphere_intersection_geometric(origin: np.ndarray[float], dir: np.ndarray[float], center: np.ndarray[float], R: float):
    L = center - origin
    tca = np.dot(L, dir)
    if (tca < 0):
        return np.zeros(3), False
    d2 = np.dot(L, L) - tca * tca
    if (d2 > R ** 2):
        return np.zeros(3), False
    thc = np.sqrt(R ** 2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    return origin + t0 * dir, True


def get_sphere_tex_coords(vec: np.ndarray[float]):
    ra, de = r_hat_to_ra_dec(normalize(vec))
    v = 0.5 + ra / 360.0
    u = 0.5 - de / 180.0
    return u, v


def sample_texture(tex: np.ndarray[float], u: float, v: float, interpMode: int=0):
    if interpMode == 0:
        return tex[int(u * tex.shape[0]), int(v * tex.shape[1])]
    else:
        print(f'Warning: texture interpolation mode {interpMode} is either not valid or not yet implemented; defaulting to mode 0 (nearest neighbor).')
        return tex[int(u * tex.shape[0]), int(v * tex.shape[1])]


def sample_planet(planetName: str, pos_PlanetSSB: np.ndarray[float], rot_PlanetSSB: np.ndarray[float], pos_ObsSSB: np.ndarray[float], 
                  T_worldToCamera: np.ndarray[float], fieldOfViewU: float, fieldOfViewV: float, U: int, V: int, uc: int, vc: int, numSamplePoints: int):
    radius = planetRadii[planetName]
    avgColor = np.zeros(3)
    avgNormal = np.zeros(3)
    for s in range(numSamplePoints):
        randomU = random.uniform(-0.5, 0.5)
        randomV = random.uniform(-0.5, 0.5)
        uSample = float(uc) + randomU
        vSample = float(vc) + randomV
        pointingSample = camera_to_world(T_worldToCamera.T, uv_centered_to_camera(fieldOfViewU, fieldOfViewV, float(U), float(V), uSample, vSample))
        intersection, success = ray_sphere_intersection_geometric(pos_ObsSSB, normalize(pointingSample), pos_PlanetSSB, radius)
        if success:
            normalSSB = normalize(intersection - pos_PlanetSSB)
            texCoordU, texCoordV = get_sphere_tex_coords(rot_PlanetSSB @ normalSSB)
            color = sample_texture(planetTextures[planetName], texCoordU, texCoordV, interpMode=0)
            avgColor += color
            avgNormal += normalSSB
    
    avgColor /= float(numSamplePoints)
    avgNormal /= float(numSamplePoints)
    return avgColor, normalize(avgNormal)


def get_sphere_lighting(normal: np.ndarray[float], normal_SunPlanet: np.ndarray[float]):
    return clamp(np.dot(normal_SunPlanet, normal), 0.0, 1.0)


# Initialize planet textures
for planetName in allPlanets:
    home = ".\\"
    if len(sys.argv) > 1:
        home = sys.argv[1]
    
    print(f'Reading texture for planet {planetName}...')
    tex = cv2.imread(home + "py_src\\star\\data\\textures\\" + planetName + ".jpg")
    planetTextures[planetName] = tex

print('Done initializing planet textures!')
