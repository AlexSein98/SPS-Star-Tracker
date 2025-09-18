from py_src.star.python.render import *
from SPS.gravity import *

import os
import sys


if __name__ == "__main__":
    rEarth = 6378136.3
    rMoon = 1737400.0
    rMars = 3396190.0
    rPhobos = 11000.0

    np.set_printoptions(suppress=True)

    # Process arguments
    n = len(sys.argv)
    home = ".\\py_src\\star\\"
    if n > 1:
        home = sys.argv[1]

    # Delete all old images
    delete_old = True
    if delete_old:
        dir_path = home + "python\\output"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(os.path.join(home + "python", "output"))

    # Get full star catalog
    catalog = read_csv_catalog(home + "data\\catalog.csv")

    # Get lat/lon/alt locations
    latLonAlts = []
    with open(".\\sampleLatLongs.csv", "r") as sampleLatLongs:
        reader = csv.reader(sampleLatLongs, delimiter=",", quotechar="|", lineterminator="\n")
        for line in reader:
            latLonAlts.append([float(line[0]), float(line[1]), float(line[2])])

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
    moonRot = spice.pxform("J2000", "MOON_PA", etNow)
    
    idx = 0
    numImages = len(latLonAlts)
    true_data = []

    maxDegree = 128
    maxOrder = 128
    sampler = GravSampler(maxDegree, maxOrder)

    for i in range(len(latLonAlts)):
        lat = latLonAlts[i][0]
        lon = latLonAlts[i][1]
        alt = latLonAlts[i][2]
        cameraPosMoonFixed = r_hat(lon, lat)  # * (alt + rMoon)
        cameraPosMoonCentered = (moonRot.T @ np.array([cameraPosMoonFixed]).T).T[0]

        g = sampler.SampleGravity(lat, lon, moon.radius + 0.001 * alt, maxDegree)
        gInertial = (moonRot.T @ np.array([g]).T).T[0]
        ra, de = r_hat_to_ra_dec(-normalize(gInertial))
    
        moonPos, _ = spice.spkezp(301, etNow, "J2000", "NONE", 0)
        cameraPos = moonPos + cameraPosMoonCentered

        # Camera parameters
        dimU: int = 1024
        dimV: int = 1024
        fovU: float = 20.0

        # Render
        idx += 1
        print(f'Rendering image {idx} of {numImages} ({round(100.0 * float(idx) / numImages, 2)}%): RA = {round(ra, 3)}, Dec = {round(de, 3)}')
        params = RenderParams(etJ2000, etNow, cameraPos, dimU, dimV, fovU, starMaxPixelRadius, catalog, relativeMagnitude, relativeFlux)
        
        # Actually do the render
        render(ra, de, params, home)
        true_data.append(get_true_attitude(ra, de))
    
    # Write output
    write_csv(home + "truth_data.csv", true_data)
