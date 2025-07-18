import csv
import json
import sys

from transformations import *


def ra_dec_to_rot(rightAscension: float, declination: float):
    return R3(np.deg2rad(rightAscension)) @ R2(np.deg2rad(-declination))


def r_hat(rightAscension: float, declination: float):
    return normalize(ra_dec_to_rot(rightAscension, declination)[:, 0])


def p_hat(rightAscension: float, declination: float):
    return normalize(ra_dec_to_rot(rightAscension, declination)[:, 1])


def q_hat(rightAscension: float, declination: float):
    return normalize(ra_dec_to_rot(rightAscension, declination)[:, 2])


def time_to_degrees(angleInTime: str):
    # No need to account for negatives because all "hours" (and hence minutes and seconds) are positive
    return 15.0 * (float(angleInTime[0:2]) + float(angleInTime[4:6]) / 60.0 + float(angleInTime[8:12]) / 3600.0)


def arc_to_degrees(angleInArcTime: str):
    # Need to account for a negative in front of the "degrees" negating the "minutes" and "seconds" as well
    isNegative = angleInArcTime[0] == "-"
    if isNegative:
        return float(angleInArcTime[0:3]) - float(angleInArcTime[5:7]) / 60.0 - float(angleInArcTime[9:11]) / 3600.0
    else:
        return float(angleInArcTime[0:3]) + float(angleInArcTime[5:7]) / 60.0 + float(angleInArcTime[9:11]) / 3600.0


# Source: https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
# Temperature 'temp' must be in Kelvin
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


def magnitude_to_flux(mag: float, refMag: float, refFlux: float):
    return refFlux * 10 ** (0.4 * (refMag - mag))


def flux_to_magnitude(flux: float, refMag: float, refFlux: float):
    return refMag - 2.5 * np.log10(flux / refFlux)


def read_json_catalog(catalogPath: str):
    with open(catalogPath, 'r', encoding='utf-8') as catalogFile:
        catalogJSON = json.load(catalogFile)
        catalog = []

        for i in range(len(catalogJSON)):
            # Right ascension, declination, and associated proper motion
            str_rightAscension = catalogJSON[i]["RA"]
            str_declination = catalogJSON[i]["Dec"]
            str_properMotionRA = catalogJSON[i]["pmRA"]
            str_properMotionDE = catalogJSON[i]["pmDE"]
            rightAscension = time_to_degrees(str_rightAscension)
            declination = arc_to_degrees(str_declination)
            properMotionRA = float(str_properMotionRA)
            properMotionDE = float(str_properMotionDE)

            # Parallax
            parallax = 0.0
            if "Parallax" in catalogJSON[i]:
                str_parallax = catalogJSON[i]["Parallax"]
                parallax = float(str_parallax)
            
            # Right ascension and declination unit vectors
            pHat_i = p_hat(rightAscension, declination)
            qHat_i = q_hat(rightAscension, declination)
            
            # Magnitude
            vMag = 10.0
            if "Vmag" in catalogJSON[i]:
                str_vMag = catalogJSON[i]["Vmag"]
                vMag = float(str_vMag)
            
            # Temperature (and color)
            temp = 6000.0
            r = 255
            g = 255
            b = 255
            if "K" in catalogJSON[i]:
                str_temp = catalogJSON[i]["K"]
                temp = float(str_temp)
                r, g, b = temp_to_rgb(temp)

            # Output for this star
            catalog.append([rightAscension, declination, properMotionRA, properMotionDE, parallax, pHat_i[0], pHat_i[1], pHat_i[2], qHat_i[0], qHat_i[1], qHat_i[2], vMag, temp, r, g, b])
        return catalog


def export_json_catalog_to_csv(catalogPath: str, home: str=".\\py_src\\star\\"):
    catalog = read_json_catalog(catalogPath)
    with open(home + "data\\catalog.csv", "w") as catalogCSV:
        writer = csv.writer(catalogCSV, delimiter=',', quotechar='"', lineterminator='\n')
        for line in catalog:
            writer.writerow(line)


def read_csv_catalog(catalogPath: str):
    with open(catalogPath, 'r') as catalogCSV:
        reader = csv.reader(catalogCSV, delimiter=',', quotechar='"', lineterminator='\n')
        catalog = []
        for row in reader:
            catalog.append([float(element) for element in row])
        return catalog


def subset_catalog(catalog, indices):
    catalog_subset = []
    for i in range(len(catalog)):
        if i in indices:
            catalog_subset.append(catalog[i])
    return catalog_subset


def subset_catalog_by_magnitude(catalog, magnitudeCutoff: float):
    catalog_subset = []
    for i in range(len(catalog)):
        if catalog[i][11] < magnitudeCutoff:
            catalog_subset.append(catalog[i])
    return catalog_subset


if __name__ == "__main__":
    home: str = ".\\py_src\\star\\"
    if len(sys.argv) > 1:
        home = sys.argv[1]
    
    # TEST 1: READ ORIGINAL (JSON) BRIGHT STAR CATALOG (BSC)
    # read_json_catalog(home + "data\\bsc5-all.json")

    # TEST 2: CONVERT ORIGINAL BSC TO CSV
    export_json_catalog_to_csv(home + "data\\bsc5-all.json", home)

    # TEST 3: READ CSV VERSION OF BSC CATALOG AND PRINT
    # catalog = read_csv_catalog(home + "data\\catalog.csv")
    # print(catalog)
    pass
