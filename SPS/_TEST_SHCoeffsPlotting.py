import spaceteams as st

import os
import sys
import csv
import copy
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from SPS.grav_vesta_DAWN import grav_vesta_DAWN, grav_base


def read_csv(path: str, ignore: list=[], hasHeader=False):
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"', lineterminator='\n')
        data = []

        headerBool = False
        if hasHeader:
            headerBool = True
        for row in reader:
            if headerBool:
                headerBool = False
                continue
            data.append([float(row[i]) for i in range(len(row)) if i not in ignore])
        return data


def write_csv(filepath: str, data: list[np.ndarray[float]]):
    with open(filepath, "w") as dataCSV:
        writer = csv.writer(dataCSV, delimiter=',', quotechar='"', lineterminator='\n')
        for line in data:
            writer.writerow(line)


def WeightedLeastSquares(data: npt.NDArray, variances: npt.NDArray) -> npt.NDArray:
    numRows = data.shape[0]
    numCols = data.shape[1]

    H = np.identity(numCols)
    W = np.zeros((numRows * numCols, numRows * numCols))
    W[0:numCols, 0:numCols] = np.diag(1.0 / variances[0])
    z = copy.deepcopy(data[0])

    for i in range(1, numRows):
        H = np.vstack((H, np.identity(numCols)))
        W[i * numCols:(i + 1) * numCols, i * numCols:(i + 1) * numCols] = np.diag(1.0 / variances[i])
        z = np.concat((z, data[i]))
    H_T_W = H.T @ W
    H_T_W_H = H_T_W @ H
    print(f"H_T_W_H dims = {H_T_W_H.shape}")
    return np.linalg.inv(H_T_W @ H) @ H_T_W @ z


doLeastSquares: bool = True
startIdx: int = 0
endIdx: int = 400
tooSmallCutoff: float = 1e-8


thisRepo = os.path.join(str(st.path_utils.GetLocalAssetsDir()), "Repos", "SPS-Star-Tracker")
localOutput = str(st.path_utils.GetLocalOutputDir())
sys.path.append(thisRepo)
sys.path.append(localOutput)

planetName = "VestaLongTerm"
outputDir = os.path.join(localOutput, planetName)

path_mx_Clm = os.path.join(outputDir, "mx_Clm_" + planetName + ".csv")
path_mx_Slm = os.path.join(outputDir, "mx_Slm_" + planetName + ".csv")
path_Pxx_Clm = os.path.join(outputDir, "Pxx_Clm_" + planetName + ".csv")
path_Pxx_Slm = os.path.join(outputDir, "Pxx_Slm_" + planetName + ".csv")

# Take the last estimated values for now (could do WLS later)
Clm_data = np.asarray(read_csv(path_mx_Clm))
Slm_data = np.asarray(read_csv(path_mx_Slm))
Clm_variances = np.asarray(read_csv(path_Pxx_Clm))
Slm_variances = np.asarray(read_csv(path_Pxx_Slm))

Clm_linear = Clm_data[-1]
Slm_linear = Slm_data[-1]

if doLeastSquares:
    Clm_linear = WeightedLeastSquares(Clm_data[startIdx:endIdx], Clm_variances[startIdx:endIdx])
    Slm_linear = WeightedLeastSquares(Slm_data[startIdx:endIdx], Slm_variances[startIdx:endIdx])

# Setting up the true gravity model
estDegree: int = 8  # = estOrder, keep them the same for simplicity
gravModel_true: grav_base = grav_vesta_DAWN()
Clm_true = gravModel_true.Clm[:estDegree + 1, :estDegree + 1]
Slm_true = gravModel_true.Slm[:estDegree + 1, :estDegree + 1]

for i in range(estDegree + 1):
    for j in range(estDegree + 1):
        if abs(Clm_true[i, j]) < tooSmallCutoff:
            Clm_true[i, j] = 0.0
        if abs(Slm_true[i, j]) < tooSmallCutoff:
            Slm_true[i, j] = 0.0

# Populate lower triangular coefficient matrices
indices_lowerTriangular_Clm = np.tril_indices(estDegree + 1)
indices_lowerTriangular_Slm = np.tril_indices(estDegree)
Clm_linear = np.concat((np.zeros(3), Clm_linear))
Slm_linear_noFirstColumn = np.concat((np.zeros(1), Slm_linear))

Clm = np.zeros((estDegree + 1, estDegree + 1))
Slm = np.zeros((estDegree + 1, estDegree + 1))
Slm_subset = np.zeros((estDegree, estDegree))

Clm[indices_lowerTriangular_Clm] = Clm_linear
Slm_subset[indices_lowerTriangular_Slm] = Slm_linear_noFirstColumn
Slm[1:estDegree + 1, 1:estDegree + 1] = Slm_subset

# Delta and percent diff in Clm and Slm
delta_Clm = Clm - Clm_true
delta_Slm = Slm - Slm_true
# percentDiff_Clm = np.nan * np.ones((estDegree + 1, estDegree + 1))
# percentDiff_Slm = np.nan * np.ones((estDegree + 1, estDegree + 1))
percentDiff_Clm = np.zeros((estDegree + 1, estDegree + 1))
percentDiff_Slm = np.zeros((estDegree + 1, estDegree + 1))
for i in range(estDegree + 1):
    for j in range(estDegree + 1):
        if Clm_true[i, j] != 0:
            percentDiff_Clm[i, j] = 100.0 * abs(delta_Clm[i, j] / Clm_true[i, j])
        if Slm_true[i, j] != 0:
            percentDiff_Slm[i, j] = 100.0 * abs(delta_Slm[i, j] / Slm_true[i, j])

# print(f"percentDiff_Clm = {percentDiff_Clm}")
# print(f"percentDiff_Slm = {percentDiff_Slm}")

# Generate mask for the upper triangle
# keep_mask preserves data where it is True (lower triangle + diagonal)
keep_mask_Clm = np.tril(np.ones_like(percentDiff_Clm, dtype=bool))
keep_mask_Slm = np.tril(np.ones_like(percentDiff_Slm, dtype=bool))

# Apply the mask: replace upper triangle with NaN so it stays blank
shaded_matrix_Clm = np.where(keep_mask_Clm, percentDiff_Clm, np.nan)
shaded_matrix_Slm = np.where(keep_mask_Slm, percentDiff_Slm, np.nan)

# Plot shaded table
fig = plt.figure(layout='constrained')
ax_Clm = fig.add_subplot(121)
ax_Slm = fig.add_subplot(122)

ax_Clm_disp = ax_Clm.imshow(shaded_matrix_Clm, cmap='Blues', vmin=0.0, vmax=100.0)
ax_Slm_disp = ax_Slm.imshow(shaded_matrix_Slm, cmap='Blues', vmin=0.0, vmax=100.0)

# shadingCutoff_Clm = 0.5 * np.nanmax(shaded_matrix_Clm)
# shadingCutoff_Slm = 0.5 * np.nanmax(shaded_matrix_Slm)
shadingCutoff_Clm = 50.0
shadingCutoff_Slm = 50.0

# Add numeric annotations inside the cells
for i in range(percentDiff_Clm.shape[0]):
    for j in range(percentDiff_Clm.shape[1]):
        if not np.isnan(shaded_matrix_Clm[i, j]):
            ax_Clm.text(j, i, f"{shaded_matrix_Clm[i, j]:.2f}", 
                    ha="center", va="center", 
                    color="white" if shaded_matrix_Clm[i, j] > shadingCutoff_Clm else "black")
        if not np.isnan(shaded_matrix_Slm[i, j]):
            ax_Slm.text(j, i, f"{shaded_matrix_Slm[i, j]:.2f}", 
                    ha="center", va="center", 
                    color="white" if shaded_matrix_Slm[i, j] > shadingCutoff_Slm else "black")

# 4. Styling the table grid and labels
labelsX = [str(int(i)) for i in np.linspace(0, percentDiff_Clm.shape[0] - 1, percentDiff_Clm.shape[0])]
labelsY = [str(int(i)) for i in np.linspace(0, percentDiff_Clm.shape[1] - 1, percentDiff_Clm.shape[1])]
ax_Clm.set_xticks(np.arange(len(labelsX)))
ax_Clm.set_yticks(np.arange(len(labelsY)))
ax_Clm.set_xticklabels(labelsX)
ax_Clm.set_yticklabels(labelsY)
ax_Slm.set_xticks(np.arange(len(labelsX)))
ax_Slm.set_yticks(np.arange(len(labelsY)))
ax_Slm.set_xticklabels(labelsX)
ax_Slm.set_yticklabels(labelsY)

# Remove the outer frame and cleanup layout
ax_Clm.spines[:].set_visible(False)
ax_Slm.spines[:].set_visible(False)
row_bounds_Clm = np.arange(-0.5, Clm.shape[0], 1)
col_bounds_Clm = np.arange(-0.5, Clm.shape[1], 1)
ax_Clm.set_xticks(col_bounds_Clm, minor=True)
ax_Clm.set_yticks(row_bounds_Clm, minor=True)
ax_Clm.grid(which='minor', color='black', linestyle='-', linewidth=2)
row_bounds_Slm = np.arange(-0.5, Slm.shape[0], 1)
col_bounds_Slm = np.arange(-0.5, Slm.shape[1], 1)
ax_Slm.set_xticks(col_bounds_Slm, minor=True)
ax_Slm.set_yticks(row_bounds_Slm, minor=True)
ax_Slm.grid(which='minor', color='black', linestyle='-', linewidth=2)
# ax_Clm.grid()
# ax_Slm.grid()
fig.colorbar(ax_Clm_disp, ax=ax_Clm, shrink=0.8)
fig.colorbar(ax_Slm_disp, ax=ax_Slm, shrink=0.8)

ax_Clm.set_title(r"Percent Error in $C[l,m]$ Estimates")
ax_Slm.set_title(r"Percent Error in $S[l,m]$ Estimates")

# plt.tight_layout()
plt.show()