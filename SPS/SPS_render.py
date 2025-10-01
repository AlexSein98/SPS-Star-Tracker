from py_src.star.python.render import *
from SPS.global_config import *

import os
import sys


if __name__ == "__main__":
    planet = globalConfig.planet
    gravModel: grav_base = planet.gravModel

    np.set_printoptions(suppress=True)

    # Process arguments
    n = len(sys.argv)
    home = "./py_src/star/"
    if n > 1:
        home = sys.argv[1]

    # Delete all old images
    delete_old = True
    if delete_old:
        dir_path = home + "python/output/" + planet.planetName.title()
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(os.path.join(home + "python", "output", planet.planetName.title()))

    # Get full star catalog
    catalog = read_csv_catalog(home + "data/catalog.csv")

    # Get lat/lon/alt locations
    latLonAlts = []
    with open("./sampleLatLongs_" + planet.planetName.title() + ".csv", "r") as sampleLatLongs:
        reader = csv.reader(sampleLatLongs, delimiter=",", quotechar="|", lineterminator="\n")
        for line in reader:
            latLonAlts.append([float(line[0]), float(line[1]), float(line[2])])

    # Star rendering parameters
    relativeMagnitude = 6.0  # "full" exposure is set for stars of this magnitude
    relativeFlux = 1.0  # flux for fully-exposed stars
    starMaxPixelRadius = 16

    # Astrophysical parameters
    spice.furnsh(home + "data/metakernel.txt")
    tJ2000 = '2000 Jan 1, 00:00:00 UTC'
    tNow = '2025 July 4, 00:00:00 UTC'
    etJ2000 = spice.str2et(tJ2000)
    etNow = spice.str2et(tNow)
    planetRot = spice.pxform("J2000", planet.planetFrame, etNow)
    
    idx = 0
    numImages = len(latLonAlts)
    true_data = []

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

    for i in range(len(latLonAlts)):
        # lat = latLonAlts[i][0]
        # lon = latLonAlts[i][1]
        # alt = latLonAlts[i][2]

        phi_pg: float = latLonAlts[i][0]
        lon: float = latLonAlts[i][1]
        h_ellp: float = latLonAlts[i][2]

        cameraPosPlanetFixed = planetographic_to_cartesian(phi_pg, lon, h_ellp, radiusEquatorial, radiusPolar)
        cameraPosPlanetCentered = (planetRot.T @ np.array([cameraPosPlanetFixed]).T).T[0]

        phi_pc, _, _ = r_to_latlonalt(cameraPosPlanetFixed, radiusEquatorial)
        g = sampler.SampleAcceleration_Custom(phi_pc, lon, np.linalg.norm(cameraPosPlanetFixed), maxDegree, overrideSphericalHarmonics=False, includeThirdBody=False, et=etNow)
        Omega = np.array([0.0, 0.0, gravModel.omega])
        g -= np.cross(Omega, np.cross(Omega, cameraPosPlanetFixed))  # Handle being on the surface of the Earth

        gInertial = (planetRot.T @ np.array([g]).T).T[0]
        ra, de = r_hat_to_ra_dec(-normalize(gInertial))

        cameraPos = planetPos + 0.001 * cameraPosPlanetCentered  # needs to be in km

        print(f'Lat = {phi_pg}, lon = {lon}:')
        print(f'cameraPosPlanetFixed = {cameraPosPlanetFixed} m')
        print(f'cameraPos (SSB) = {cameraPos} km\n')

        # Render
        idx += 1
        print(f'Rendering image {idx} of {numImages} ({round(100.0 * float(idx) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}')
        params = RenderParams(idx, etJ2000, etNow, cameraPos, dimU, dimV, fovU, starMaxPixelRadius, 
                              catalog, relativeMagnitude, relativeFlux, planet)
        
        # Actually do the render
        render(ra, de, params, home)
        true_data.append(get_true_attitude(ra, de))
    
    # Write output
    write_csv(home + "truth_data_" + planet.planetName.title() + ".csv", true_data)
