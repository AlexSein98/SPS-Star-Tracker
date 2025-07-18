from catalog import *
import spiceypy as spice
import itertools


#########################################################################
####    WARNING: DO NOT RUN OR USE THIS FILE, IT IS NOT SUPPORTED    ####
#########################################################################


def generate_database_unique(catalogPath, magnitudeCutoff):
    catalog = read_csv_catalog(catalogPath)
    subset = subset_catalog_by_magnitude(catalog, magnitudeCutoff)
    numStars = len(subset)
    uniqueCombos = list(itertools.combinations(range(numStars), r=3))
    numCombos = len(uniqueCombos)

    database = []
    k = 0
    for combo in uniqueCombos:
        i1, i2, i3 = combo

        r_1 = r_hat(subset[i1][0], subset[i1][1])
        r_2 = r_hat(subset[i2][0], subset[i2][1])
        r_3 = r_hat(subset[i3][0], subset[i3][1])

        r_12 = normalize(r_2 - r_1)
        r_13 = normalize(r_3 - r_1)
        r_23 = normalize(r_3 - r_2)

        cosAlpha1 = np.dot(r_12, r_13)
        cosAlpha3 = np.dot(r_13, r_23)
        database.append([i1, i2, i3, cosAlpha1, cosAlpha3])

        if k % (int(float(numCombos) / 1000.0)) == 0:
            print(f'Done processing {k} combos of {numCombos} ({round(100.0 * float(k) / float(numCombos), 2)}%)')
        k += 1
    
    # Sort and save database
    database_sorted = sorted(database, key=lambda x: x[4])
    with open(".\\py_src\\star\\data\\database.csv", "w") as databaseCSV:
        writer = csv.writer(databaseCSV, delimiter=',', quotechar='"', lineterminator='\n')
        for line in database_sorted:
            writer.writerow(line)


def read_database(databasePath):
    with open(databasePath, 'r') as databaseCSV:
        reader = csv.reader(databaseCSV, delimiter=',', quotechar='"', lineterminator='\n')
        database = []
        for row in reader:
            database.append([float(element) for element in row])
        return np.asarray(database)


# def k_vector_search(database, cosAlpha1, cosAlpha2, cosAlpha3, tolerance):
#     kVector = database[:, 4]
#     isCloseAlpha1 = np.isclose(kVector, np.full(cosAlpha1), tolerance)
#     isCloseAlpha1_sum = np.sum(isCloseAlpha1)
#     if (isCloseAlpha1_sum > 0):
#         if (isCloseAlpha1_sum > 1):

#         else:

#     isCloseAlpha2 = np.isclose(kVector, np.full(cosAlpha1))
#     isCloseAlpha3 = np.isclose(kVector, np.full(cosAlpha1))

#########################################################################
####    WARNING: DO NOT RUN OR USE THIS FILE, IT IS NOT SUPPORTED    ####
#########################################################################

if __name__ == "__main__":
    generate_database_unique(".\\py_src\\star\\data\\catalog.csv", 3.0)
