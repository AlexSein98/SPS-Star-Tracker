from py_src.star.python.transformations import *


class grav_base:
    def __init__(self, name: str, spiceBodyFrame: str, maxDegree: int, maxOrder: int, mu: float, 
                 omega: float, radius: float, polarRadius: float):
        self.name = name
        self.spiceBodyFrame = spiceBodyFrame
        self.maxDegree = maxDegree
        self.maxOrder = maxOrder
        self.mu = mu
        self.mass = self.mu / (UniversalConstants.G * 1e9)
        self.omega = omega
        self.radius = radius
        self.polarRadius = polarRadius
        self.eccentricity = np.sqrt(1.0 - (self.polarRadius / self.radius) ** 2)
        self.flattening = 1.0 - np.sqrt(1.0 - self.eccentricity ** 2)

        self.Clm = np.zeros((self.maxDegree + 1, self.maxOrder + 1))
        self.Slm = np.zeros((self.maxDegree + 1, self.maxOrder + 1))
