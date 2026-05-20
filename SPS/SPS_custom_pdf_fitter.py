from py_src.star.python.transformations import *
from SPS.gravity import *
from SPS.SPS_samples import ReadDEM, SampleGlobalDEM_LatLon
from SPS.global_config import *

import sys
import os
import csv
import copy
import time
import datetime

from matplotlib import pyplot as plt
import matplotlib
import spiceypy as spice

import scipy.stats as stats


class StarTrackerPDF:
    def __init__(self, _sigma: float, _weight_g: float, _weight_u: float):
        self.sigma = _sigma
        self.weight_g = _weight_g
        self.weight_u = _weight_u

        self.b = np.pi
        self.a = -np.pi
    
    def eval(self, x: float):
        g: float = 1.0 / (self.sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x / self.sigma) ** 2)
        u: float = 1.0 / ()
