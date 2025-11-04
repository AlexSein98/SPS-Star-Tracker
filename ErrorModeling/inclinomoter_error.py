import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import matplotlib.ticker as mtick

def temperatureError(T, null_coeff, scale_coeff, raw_output, errors, source_flags):
    delta_T = T - 20
    Temp_error_sqrd = 0
    if source_flags["null coefficient"]:
        null_term = delta_T * (delta_T * scale_coeff - 1) * errors["null coefficient"]
        Temp_error_sqrd += null_term ** 2
    
    if source_flags["scale coefficient"]:
        scale_term = delta_T * (delta_T * null_coeff + raw_output) * errors["scale coefficient"]
        Temp_error_sqrd += scale_term ** 2
    
    if source_flags["temperature"]:
        Temp_term = (2 * delta_T * null_coeff * scale_coeff - null_coeff + scale_coeff * raw_output) * errors["temperature"]
        Temp_error_sqrd += Temp_term ** 2
        
    if source_flags["output"]:
        output_term = (delta_T * scale_coeff - 1) * errors["output"]
        Temp_error_sqrd += output_term ** 2

    return np.sqrt(Temp_error_sqrd)

def output_to_deg(output, is_error):
    # linear range
    raw_ouput_at_0 = 32768
    counts_per_degree = 17582
    
    if is_error:
        return output / counts_per_degree
    else:
        return (output - raw_ouput_at_0) / counts_per_degree


# initialize varaibles
theta_i = random.gauss(mu=0, sigma=0.5)
iterations = 2000
timespan = 24 * 3600
timestep = 100

Temperature = 50
null_temp_coeff = 0.0005
scale_temp_coeff = 0.00375

errors = {
    "temperature" : 0.03,
    "null coefficient" : 0.0001,
    "scale coefficient" : 0.0001,
    "output": 4,
}

error_sources = {
    "temperature" : True,
    "null coefficient" : True,
    "scale coefficient" : True,
    "output": False,
}


all_sims = []
i = 0
while i < iterations:
    theta = theta_i
    theta_vals = []
    t = 0
    while t < timespan:
        sigma_per_sec = output_to_deg(temperatureError(Temperature, null_temp_coeff, scale_temp_coeff, 2, errors, error_sources), is_error=True)
        sigma = sigma_per_sec * (timestep ** 0.5)

        theta += random.gauss(mu=0, sigma=sigma)
        theta_vals.append(theta)

        t += timestep

    all_sims.append(theta_vals)
    i += 1


# Convert sims to array and build time axis
arr = np.array(all_sims)       # shape: (iterations, times)
t = np.arange(0, timespan, timestep)


# --- Seaborn / matplotlib visualization ---
sns.set(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Plot a random subset of thin, faded traces so the ensemble shape is visible
rng = np.random.default_rng(0)
n_sample_traces = min(100, arr.shape[0])
sample_idx = rng.choice(arr.shape[0], size=n_sample_traces, replace=False)
for trace in arr[sample_idx]:
    ax.plot(t, trace, color="purple", alpha=0.06, linewidth=0.8)

# Plot mean trace and a shaded ±3σ band
mean_trace = arr.mean(axis=0)
std_trace = arr.std(axis=0)

upper = mean_trace + 3 * std_trace
lower = mean_trace - 3 * std_trace

sns.lineplot(x=t, y=mean_trace, ax=ax, color="darkviolet", linewidth=2.0, label="Mean")
ax.fill_between(t, lower, upper, color="mediumpurple", alpha=0.25, label="±3σ envelope")

# Labels, title, legend, tidy axes
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\theta$ (deg)")
ax.set_title("Monte Carlo Inclinometer Simulations")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.4)
ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
ax.yaxis.set_major_locator(mtick.MaxNLocator(6))

plt.tight_layout()
plt.savefig("theta_simulation_3sigma.png", dpi=300, bbox_inches="tight")
plt.show()
