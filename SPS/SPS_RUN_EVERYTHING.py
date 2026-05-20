import subprocess

subprocess.run(["python", "-m", "SPS.SPS_samples"])
subprocess.run(["python", "-m", "SPS.SPS_gravity_var"])
subprocess.run(["python", "-m", "SPS.SPS_render"])
subprocess.run(["python", "-m", "SPS.SPS_process_images"])
subprocess.run(["python", "-m", "SPS.SPS_attitude_error"])
subprocess.run(["python", "-m", "SPS.SPS_gradient"])
