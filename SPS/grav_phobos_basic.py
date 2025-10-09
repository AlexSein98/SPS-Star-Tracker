from SPS.grav_model import *


class grav_phobos_basic(grav_base):
    def __init__(self):
        name = "PHOBOS"
        spiceBodyFrame = "IAU_PHOBOS"
        maxDegree = 2
        maxOrder = 2
        mu = 7.0765e5
        omega = 0.000228032986482
        radius = 10993.0
        polarRadius = 10993.0

        super(grav_phobos_basic, self).__init__(name, spiceBodyFrame, maxDegree, maxOrder, mu, omega, radius, polarRadius)

        Clm = np.zeros((self.maxDegree + 1, self.maxOrder + 1))
        Slm = np.zeros((self.maxDegree + 1, self.maxOrder + 1))

        # FULLY NORMALIZED GRAVITY COEFFICIENTS (unitless)
        # Clm[2][0] = -0.1378
        # Clm[2][1] = 0.0024
        # Clm[2][2] = 0.0166

        # Slm[2][0] = 0.0
        # Slm[2][1] = -0.00077
        # Slm[2][2] = 0.00054

        Clm[2][0] = -0.0616260334599
        Clm[2][1] = 0.00185903200618
        Clm[2][2] = 0.0257166094188

        Slm[2][0] = 0.0
        Slm[2][1] = -0.000596439435316
        Slm[2][2] = 0.000836564402781

        self.Clm = Clm
        self.Slm = Slm
