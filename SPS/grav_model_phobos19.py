from SPS.grav_model import *


class grav_phobos19(grav_base):
    def __init__(self):
        name = "PHOBOS"
        spiceBodyFrame = "IAU_PHOBOS"
        maxDegree = 19
        maxOrder = 19
        mu = 0.7127e6
        omega = 0.000227897374386
        radius = 11000.0
        flattening = 0.0

        super(grav_phobos19, self).__init__(name, spiceBodyFrame, maxDegree, maxOrder, mu, omega, radius, flattening)