# THIS COMMENT LINE SHOULD BE THE FIRST LINE OF THE FILE
# DON'T CHANGE ANY OF THE BELOW; NECESSARY FOR JOINING SIMULATION
import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
    exit(1)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)
import numpy as np
# DON'T CHANGE ANY OF THE ABOVE; NECESSARY FOR JOINING SIMULATION
################################################################


import numpy.typing as npt
import random


this = st.GetThisSystem()
pawn: st.Entity = this.GetParam(st.VarType.entityRef, "Pawn")
spsDevice: st.Entity = this.GetParam(st.VarType.entityRef, "SPS")

random.seed(100)

datasets: list[str] = this.GetParamArray(st.VarType.string, "Datasets")
priorities: list[float] = this.GetParamArray(st.VarType.double, "DatasetPriorities")

planetData = st.ProcPlanet.DataStore()
for i in range(len(datasets)):
    dataPath = st.path_utils.AssetPathToReal(st.AssetType.PlanetData, datasets[i])
    planetData.AddGeoBinAltimetryLayer(priorities[i], dataPath, st.ProcPlanet.GeoBin_Extra_Args())


def skew(vec: npt.NDArray):
    return np.array([[0.0, -vec[2], vec[1]],
                     [vec[2], 0.0, -vec[0]],
                     [-vec[1], vec[0], 0.0]])


def RotateVectorAroundAxis(vec: npt.NDArray, axis: npt.NDArray, angle: float) -> npt.NDArray:
    R: npt.NDArray = np.identity(3) * np.cos(angle) + \
        (1.0 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * skew(axis)
    return R @ vec


"""
Generate a new SPS testing location on the sun-lit side of a planet. 
The planet is specified by the "Planet" EntityRef in the payload ParamMap.
"""
def NewSPSLocation(paramMap: st.ParamMap, timestamp: st.timestamp):
    st.OnScreenAlert("Setting up new location...", "SPSGuessr Setup", st.Severity.Warning)

    planet: st.Entity = paramMap.GetParam(st.VarType.entityRef, "Planet")
    planetFrame = planet.GetBodyFixedFrame()
    radius: float = planet.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])
    
    sun: st.Entity = st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, ["CelestialObjects", "Sun"])
    sunLoc = sun.getLocation().WRT_ExprIn(planetFrame)
    sunDir = sunLoc / np.linalg.norm(sunLoc)

    axisOffset: float = random.uniform(0.0, 0.5 * np.pi)
    axisRotation: float = random.uniform(0.0, 2.0 * np.pi)
    
    # Choose axis to offset-rotate about
    axis1: npt.NDArray = np.array([0.0, 0.0, 1.0])
    if np.dot(sunDir, axis1) > 0.999:
        axis1 = np.array([1.0, 0.0, 0.0])
    
    # Calculate position unit vector
    rHat = RotateVectorAroundAxis(RotateVectorAroundAxis(sunDir, axis1, axisOffset), sunDir, axisRotation)
    loc, _ = st.ProcPlanet.SampleGround(planetData, radius * rHat, radius, 0.0, 22)

    st.OnScreenLogMessage(f"New location = {loc}", "SPSGuessr Setup", st.Severity.Info)
    
    newPayload = st.ParamMap()
    newPayload.AddParam(st.VarType.entityRef, "Planet", planet)
    newPayload.AddParam(st.VarType.doubleV3, "Location", loc)
    st.SimGlobals.Publish("SetupSPS", newPayload)


"""
Set up SPS at a new testing location on a planet. The planet is specified by 
the "Planet" EntityRef in the payload ParamMap, while the (assumed surface-snapped) 
location is specified by the "Location" doubleV3 parameter.
"""
def SetupSPS(paramMap: st.ParamMap, timestamp: st.timestamp):
    planet: st.Entity = paramMap.GetParam(st.VarType.entityRef, "Planet")
    planetFrame = planet.GetBodyFixedFrame()
    radius: float = planet.GetParam(st.VarType.double, ["#Planet", "General", "Radius_m"])

    newLoc: npt.NDArray = paramMap.GetParam(st.VarType.doubleV3, "Location")
    flu = st.PlanetUtils.ForwardLeftUpFromAzimuth(newLoc, 0.0, radius)
    rot = st.PlanetUtils.RotFromForwardLeftUp(flu)

    # locOverride = st.PhysEffect(st.PhysEffectType.LocationOverride, planetFrame)
    # locOverride.setVector(newLoc)

    # rotOverride = st.PhysEffect(st.PhysEffectType.RotationOverride, planetFrame)
    # rotOverride.setQuaternion(st.math.DCM_to_Quat(rot))

    # spsDevice.setPhysEffect("LocationOverride", locOverride)
    # spsDevice.setPhysEffect("RotationOverride", rotOverride)

    spsDevice.setLocation(st.frames.FramedLoc(newLoc, planetFrame))
    spsDevice.setRotation(st.frames.FramedRot(rot, planetFrame))

    # TODO: neither of these are working?
    # PhysEffect overrides
    # overrideLoc = st.PhysEffect(st.PhysEffectType.LocationOverride, planetFrame)
    # overrideRot = st.PhysEffect(st.PhysEffectType.RotationOverride, planetFrame)
    # overrideLoc.setVector(newLoc - 3.0 * flu.forward())
    # overrideRot.setQuaternion(st.math.DCM_to_Quat(rot))
    # pawn.setPhysEffect("OverrideLoc", overrideLoc)
    # pawn.setPhysEffect("OverrideRot", overrideRot)

    # Direct setting
    # pawn.setLocation(st.frames.FramedLoc(newLoc - 3.0 * flu.forward(), planetFrame))
    # pawn.setRotation(st.frames.FramedRot(rot, planetFrame))

    payload = st.ParamMap()
    payload.AddParam(st.VarType.entityRef, "Pawn", pawn)
    payload.AddParam(st.VarType.doubleV3, "LocationOverride", newLoc + 2.0 * flu.up() - 3.0 * flu.forward())
    payload.AddParam(st.VarType.doubleV4, "RotationOverride", st.math.DCM_to_Quat(rot))
    st.SimGlobals.Publish("OverrideLocationAndRotation", payload)
    st.SimGlobals.Publish("ResetPawnRotation", payload)

    st.OnScreenAlert("Moving to next location...", "SPSGuessr Action", st.Severity.Warning)


# Subscribe both SPS-setup-related functions
st.SimGlobals.Subscribe("NewSPSLocation", NewSPSLocation)
st.SimGlobals.Subscribe("SetupSPS", SetupSPS)


exitFlag = False
while not exitFlag:
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

st.leave_sim()
