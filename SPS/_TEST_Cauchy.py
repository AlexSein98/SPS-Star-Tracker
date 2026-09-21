import numpy as np
import numpy.typing as npt

import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def pdf_gauss(x: float | npt.NDArray, mean: float, std: float):
    return (1.0 / np.sqrt(2.0 * np.pi * std ** 2)) * np.exp(-((x - mean) ** 2) / (std ** 2))


def pdf_cauchy(x: float | npt.NDArray, median: float, gamma: float):
    return (1.0 / np.pi) * (gamma / ((x - median) ** 2  + gamma ** 2))


def pdf_mix(x: float | npt.NDArray, x0: float, gamma: float, factor: float):
    return factor * pdf_cauchy(x, x0, gamma) + (1.0 - factor) * pdf_gauss(x, x0, gamma)


np.set_printoptions(suppress=True)


mean_true = 0.0
std_dev_true = 50.0

min_sample = -500.0
max_sample = 500.0
num_samples = 1000

rng = np.random.default_rng(seed=100)
x_data = rng.normal(loc=mean_true, scale=std_dev_true, size=num_samples)

cauchyFactor = 0.5
y_data = pdf_mix(x_data, mean_true, std_dev_true, cauchyFactor)

counts, bin_edges = np.histogram(y_data, bins=101, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

p0, _ = curve_fit(pdf_gauss, bin_centers, counts, p0=[0.0, 50.0])
print(f"Initial parameters = {p0}")

# parameters, _ = curve_fit(pdf_mix, bin_centers, counts, p0=[p0[0], p0[1], 0.5], 
                        #   bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, 1.0]))
# print(f"Parameters = {parameters}")

parameters, covariance = curve_fit(pdf_mix, x_data, y_data)
print(f"Parameters = {parameters}")

x_smooth = np.linspace(min_sample, max_sample, num_samples)
# y_smooth = pdf_gauss(x_smooth, parameters[0], parameters[1])
y_smooth = pdf_mix(x_smooth, parameters[0], parameters[1], parameters[2])


fig = plt.figure(layout='constrained')
ax = fig.add_subplot(111)

ax.scatter(x_data, y_data, color='red', label='Sample Points')
# ax.plot(x_smooth, y_smooth, color='blue', label=f'Best Fit:\nx0 = {round(parameters[0], 2)}\ngamma = {round(parameters[1], 2)}')
ax.plot(x_smooth, y_smooth, color='blue', label=f'Best Fit:\nx0 = {round(parameters[0], 2)}\ngamma = {round(parameters[1], 2)}\nmix = {round(parameters[2], 2)}')

ax.grid()
ax.legend()

plt.show()
