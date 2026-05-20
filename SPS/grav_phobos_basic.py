from SPS.grav_model import *


class grav_phobos_basic(grav_base):
    def __init__(self):
        name = "PHOBOS"
        spiceBodyFrame = "IAU_PHOBOS"
        maxDegree = 4
        maxOrder = 4
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

        # Something else here?
        # Clm[2][0] = -0.0616260334599
        # Clm[2][1] = 0.00185903200618
        # Clm[2][2] = 0.0257166094188

        # Slm[2][0] = 0.0
        # Slm[2][1] = -0.000596439435316
        # Slm[2][2] = 0.000836564402781

        # From the Scheeres paper
        Clm[2][0] = -0.04660347700
        Clm[2][1] = 0.0
        Clm[2][2] = 0.02418427633
        Clm[3][0] = 0.002998797015
        Clm[3][1] = -0.004139321225
        Clm[3][2] = -0.008785040655
        Clm[3][3] = 0.001185163133
        Clm[4][0] = 0.006429537912
        Clm[4][1] = 0.003369680127
        Clm[4][2] = -0.002323017571
        Clm[4][3] = -0.003114272077
        Clm[4][4] = 0.0008212813403

        Slm[2][1] = 0.0
        Slm[2][2] = 0.0
        Slm[3][1] = 0.002045708945
        Slm[3][2] = 0.001045820499
        Slm[3][3] = -0.01320053160
        Slm[4][1] = -0.001010497508
        Slm[4][2] = -0.001589281757
        Slm[4][3] = 0.002661315051
        Slm[4][4] = 0.00007342710506

        self.Clm = Clm
        self.Slm = Slm
