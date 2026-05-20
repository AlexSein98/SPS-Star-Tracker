from py_src.star.python.render import *
from SPS.global_config import *

import os
import sys
import time
import datetime


if __name__ == "__main__":
    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel

    np.set_printoptions(suppress=True)
    
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
    delete_old = True
    if delete_old:
        if os.path.exists(globalConfig.renderDir):
            shutil.rmtree(globalConfig.renderDir)
        os.mkdir(globalConfig.renderDir)

    # Get full star catalog
    catalog = read_csv_catalog("./py_src/star/data/catalog.csv")

    # Get lat/lon/alt locations
    latLonAlts = []
    with open(globalConfig.outputDir + "sampleLatLongs_" + planet.planetName.title() + ".csv", "r") as sampleLatLongs:
        reader = csv.reader(sampleLatLongs, delimiter=",", quotechar="|", lineterminator="\n")
        for line in reader:
            latLonAlts.append([float(line[0]), float(line[1]), float(line[2])])

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
    planetRot = spice.pxform("J2000", planet.planetFrame, etNow)
    
    idx = 0
    numImages = len(latLonAlts)
    true_data = []
    measured_accelerations = []

    maxDegree: int = globalConfig.grav_maxDegree
    maxOrder: int = globalConfig.grav_maxOrder
    sampler = GravSampler(gravModel, maxDegree, maxOrder)

    # Camera parameters
    dimU: int = 1024
    dimV: int = 1024
    fovU: float = 20.0

    radiusEquatorial: float = gravModel.radius
    radiusPolar: float = gravModel.polarRadius
    planetPos, _ = spice.spkpos(planet.planetName, etNow, "J2000", "NONE", "SSB")
    Omega = np.array([0.0, 0.0, gravModel.omega])

    startTime = time.perf_counter()
    elapsedSeconds: float = 0.0
    printInterval: int = 10

    for i in range(numImages):
        doPrint: bool = i % printInterval == 0

        phi_pg: float = latLonAlts[i][0]
        lon: float = latLonAlts[i][1]
        h_ellp: float = latLonAlts[i][2]

        cameraPosPlanetFixed = planetographic_to_cartesian(phi_pg, lon, h_ellp, radiusEquatorial, radiusPolar)
        cameraPosPlanetCentered = (planetRot.T @ np.array([cameraPosPlanetFixed]).T).T[0]

        phi_pc, _, _ = r_to_latlonalt(cameraPosPlanetFixed, radiusEquatorial)
        g = sampler.SampleAcceleration_Custom(phi_pc, lon, np.linalg.norm(cameraPosPlanetFixed), maxDegree, 
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
        
        measured_accelerations.append(np.array([-np.linalg.norm(g_IMU_frame), 0.0, 0.0]))

        gInertial_true = (planetRot.T @ np.array([g_true]).T).T[0]
        gInertial = (planetRot.T @ T_P_G.T @ np.array([g_IMU_frame]).T).T[0]

        # print(f"\ngInertial_true = {gInertial_true}")
        # print(f"gInertial      = {gInertial}\n")

        ra_true, de_true = r_hat_to_ra_dec(-normalize(gInertial_true))
        ra, de = r_hat_to_ra_dec(-normalize(gInertial))

        cameraPos = planetPos + 0.001 * cameraPosPlanetCentered  # needs to be in km

        # if doPrint:
        #     print(f'Lat = {phi_pg}, lon = {lon}:')
        #     print(f'cameraPosPlanetFixed = {cameraPosPlanetFixed} m')
        #     print(f'cameraPos (SSB) = {cameraPos} km\n')

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
            projectedRemainingSeconds: float = elapsedSeconds * float(len(latLonAlts) - idx) / float(idx)
            projectedRemainingTime = datetime.timedelta(seconds=round(projectedRemainingSeconds))
            print(f'Elapsed time: {elapsedTime}. Remaining time estimate: {projectedRemainingTime}\n')
    
    # Write output
    write_csv(globalConfig.outputDir + "truth_data_" + planet.planetName.title() + ".csv", true_data)
    write_csv(globalConfig.outputDir + "measurements_" + planet.planetName.title() + ".csv", measured_accelerations)

    # Timing stuff
    secondsPerImage = elapsedSeconds / float(len(latLonAlts))
    numImages_higherRes: int = 16020  # for [180 x 89] planet resolution
    extrapolatedSeconds = secondsPerImage * float(numImages_higherRes)
    extrapolatedTime = datetime.timedelta(seconds=round(extrapolatedSeconds))

    secondsPerImageText = f'({round(secondsPerImage, 3)} seconds per image)'
    finalTimingInfo = f'{elapsedTime} for {len(latLonAlts)} images ' + secondsPerImageText
    timeForMoreImages = f'{extrapolatedTime}'

    print(f'Render complete; total time: ' + finalTimingInfo)
    print(f'Projected time for {numImages_higherRes} images: {extrapolatedTime}')
