import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import matplotlib.ticker as mtick


# ------------------------------------------------------------------
# Helper: DCM from roll (phi) and pitch (theta)
#   phi   = rotation about x (roll)
#   theta = rotation about y (pitch)
#   Angles are in DEGREES.
#   T = R_y(theta) * R_x(phi)
# ------------------------------------------------------------------
def dcm_from_angles(theta_deg, phi_deg):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)

    cth, sth = np.cos(th), np.sin(th)
    cph, sph = np.cos(ph), np.sin(ph)

    # R_y(theta) * R_x(phi)
    T = np.array([
        [cth,        sth * sph,   -sth * cph],
        [0.0,        cph,          sph      ],
        [sth,       -cth * sph,    cth * cph]
    ])
    return T

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


# ------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------
# Initial angles (deg)
theta_i = random.gauss(mu=0.0, sigma=0.5)   # pitch
phi_i   = random.gauss(mu=0.0, sigma=0.5)   # roll

iterations = 2000
timespan = 24 * 3600        # seconds (24 hours)
timestep = 100               # seconds

Temperature = 50
null_temp_coeff = 0.0005
scale_temp_coeff = 0.00375

# Error settings (same structure you had before)
errors = {
    "temperature":       0.03,
    "null coefficient":  0.0001,
    "scale coefficient": 0.0001,
    "output":            4,
}

error_sources = {
    "temperature":       True,
    "null coefficient":  True,
    "scale coefficient": True,
    "output":            False,
}

# "True" reference vector (e.g. gravity in sensor frame when level)
v_true = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------
# Monte Carlo loop
# ------------------------------------------------------------------
all_meas_tilt = []   # measured tilt angle alpha(t), depends on BOTH phi & theta

for _ in range(iterations):
    theta = theta_i
    phi   = phi_i

    alpha_vals = []

    t = 0.0
    while t < timespan:
        # 1-sigma angle-rate (deg / sqrt(s)) from temp-driven error
        sigma_per_sec = output_to_deg(
            temperatureError(
                Temperature,
                null_temp_coeff,
                scale_temp_coeff,
                2,                 # raw_output (same placeholder as before)
                errors,
                error_sources
            ),
            is_error=True
        )

        # Random-walk sigma over this timestep (deg)
        sigma = sigma_per_sec * (timestep ** 0.5)

        # TWO independent random walks
        theta += random.gauss(mu=0.0, sigma=sigma)   # pitch
        phi   += random.gauss(mu=0.0, sigma=sigma)   # roll

        # Build DCM from current angles
        T = dcm_from_angles(theta, phi)

        # Measured vector = DCM * reference vector
        v_meas = T @ v_true          # [vx, vy, vz]

        # Convert measured vector into a single tilt angle (deg)
        # This uses both vx and vy, so BOTH angles are represented.
        vx, vy, vz = v_meas
        horiz_mag = np.sqrt(vx**2 + vy**2)
        alpha = np.rad2deg(np.arctan2(horiz_mag, vz))   # tilt from +z

        alpha_vals.append(alpha)

        t += timestep

    all_meas_tilt.append(alpha_vals)

all_meas_tilt = np.array(all_meas_tilt)    # shape: (iterations, N_time)
t = np.arange(0, timespan, timestep)


# ------------------------------------------------------------------
# Plot: measured output (tilt from DCM) with ±3σ envelope
# ------------------------------------------------------------------
sns.set(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

rng = np.random.default_rng(0)
n_sample_traces = min(100, all_meas_tilt.shape[0])
sample_idx = rng.choice(all_meas_tilt.shape[0], size=n_sample_traces, replace=False)

# Plot a bunch of sample realizations in the background
for trace in all_meas_tilt[sample_idx]:
    ax.plot(t, trace, alpha=0.06, linewidth=0.8)

# Mean and ±3σ envelope of the measured tilt
mean_trace = all_meas_tilt.mean(axis=0)
std_trace  = all_meas_tilt.std(axis=0)
upper = mean_trace + 3 * std_trace
lower = mean_trace - 3 * std_trace

ax.plot(t, mean_trace, linewidth=2.0, label="Mean measured tilt")
ax.fill_between(t, lower, upper, alpha=0.25, label="±3σ envelope")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Measured tilt α (deg)")
ax.set_title("Monte Carlo Inclinometer – Output from DCM (two-angle random walk)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.4)
ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
ax.yaxis.set_major_locator(mtick.MaxNLocator(6))

plt.tight_layout()
plt.savefig("measured_tilt_from_dcm_3sigma.png", dpi=300, bbox_inches="tight")
plt.show()
