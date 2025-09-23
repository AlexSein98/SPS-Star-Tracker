#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
install.py

This program is intended to install all
prerequisite packages for the star tracker
software on a computer running macOS

Note that this is useful for data processing
and analysis, but may not work for hardware-in-
the-loop testing.

'''

# Not been tested yet; this script is based on manually run commands

################################
#LOAD LIBRARIES
################################
import os
import sys
import time

################################
# Uses virtual environment to install dependencies
# Make sure terminal is in the parent directory (SPS-Star-Tracker/) before running script
################################

os.system('python3 -m venv star_tracker_venv')
os.system('source star_tracker_venv/bin/activate')

################################
#MAIN CODE
################################
# install/update python stuff
os.system('python3 -m pip3 install pip')
os.system('python3 -m pip3 install opencv-contrib-python')
os.system('python3 -m pip3 install psutil')
os.system('python3 -m pip3 install imageio')  # required for catalog creation
os.system('python3 -m pip3 install astropy')  # required for catalog creation
os.system('python3 -m pip3 install pandas')  # required for catalog creation
os.system('python3 -m pip3 install statistics')
os.system('python3 -m pip3 install astroquery')  # required for astrometry verification

# must install freetype2 dev pkg first??
os.system('python3 -m pip3 install matplotlib')
os.system('python3 -m pip3 install setuptools')
os.system('python3 -m pip3 install scipy')
os.system('python3 -m pip3 install spiceypy')  # required for planetary ephemerides
os.system('python3 -m pip install PyOpenGL')  # for OpenGL star rendering
os.system('python3 -m pip install PyOpenGL_accelerate')  # for OpenGL star rendering

# install module
home = os.getcwd()
os.chdir('./py_src/star_tracker')
os.system('python3 -m pip3 install .')
os.chdir(home)

# install/update stuff for IDS cam
os.system('python3 -m pip3 install pyueye')

print("\n\nInstallation complete.  Please restart the computer!") 
print("NOTE: no camera interfaces were installed during this process.  Other scripts/software may have to be run to install camera software interfaces and drivers.")

