import spaceteams as st
import numpy as np


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # T1 = np.array([[1.0, 0.0, 0.0],
    #                [0.0, 1.0, 0.0],
    #                [0.0, 0.0, 1.0]])
    # T1_est = st.math.ModifiedRodriguesParameters(T1).ToMatrix()
    # print(f"T1     = {T1}")
    # print(f"T1_est = {T1_est}")

    # T2 = np.array([[np.cos(0.25 * np.pi), -np.sin(0.25 * np.pi), 0.0],
    #                [np.sin(0.25 * np.pi), np.cos(0.25 * np.pi), 0.0],
    #                [0.0, 0.0, 1.0]])
    # T2_est = st.math.ModifiedRodriguesParameters(T2).ToMatrix()
    # T2_quat = st.math.Quat_to_DCM(st.math.ModifiedRodriguesParameters(T2).ToQuat())
    # print(f"T2      = {T2}")
    # print(f"T2_est  = {T2_est}")
    # print(f"T2_quat = {T2_quat}")

    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = q1 / np.linalg.norm(q1)
    q1_est = st.math.ModifiedRodriguesParameters(st.math.Quat_to_DCM(q1)).ToQuat()
    print(f"q1     = {q1}")
    print(f"q1_est = {q1_est}")

    q2 = np.array([0.0, 0.707, -0.707, 0.0])
    q2 = q2 / np.linalg.norm(q2)
    q2_est = st.math.ModifiedRodriguesParameters(st.math.Quat_to_DCM(q2)).ToQuat()
    print(f"q2     = {q2}")
    print(f"q2_est = {q2_est}")
