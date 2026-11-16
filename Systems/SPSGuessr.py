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


this = st.GetThisSystem()
pawn: st.Entity = this.GetParam(st.VarType.entityRef, "Pawn")
spsDevice: st.Entity = this.GetParam(st.VarType.entityRef, "SPS")
planet: st.Entity = this.GetParam(st.VarType.entityRef, "Planet")
planetFrame: st.frames.Frame = planet.GetBodyFixedFrame()
maxLocations: int = spsDevice.GetParam(st.VarType.int32, "NumberOfLocations")
spsDevice.AddOrSetParam(st.VarType.bool, "GameComplete", False)

currentLocIdx: int = 1
spsDevice.AddOrSetParam(st.VarType.int32, "CurrentLocationIndex", currentLocIdx)
spsDevice.AddOrSetParam(st.VarType.int32, "Score", 0)


def GameComplete():
    score: int = spsDevice.GetParam(st.VarType.int32, "Score")
    st.OnScreenAlert(f"SPSGuessr game complete! Final score: {score}/25000", "SPSGuessr", st.Severity.Warning)
    spsDevice.SetParam(st.VarType.bool, "GameComplete", True)


def SingleRoundScore(paramMap: st.ParamMap, timestamp: st.timestamp):
    estimateLoc: npt.NDArray = paramMap.GetParam(st.VarType.doubleV3, "EstimateLocation")
    currentLocIdx: int = spsDevice.GetParam(st.VarType.int32, "CurrentLocationIndex")
    trueLoc = spsDevice.getLocation().WRT_ExprIn(planetFrame)
    
    dist: float = np.linalg.norm(estimateLoc - trueLoc)

    normDist: float = spsDevice.GetParam(st.VarType.double, ["Scoring", "NormalizationDistance"])
    minDist: float = spsDevice.GetParam(st.VarType.double, ["Scoring", "MinimumDistance"])
    exponentArg: float = 10.0 * max(dist - minDist, 0.0) / normDist

    st.OnScreenLogMessage(f"dist = {dist}", "SPSGuessrScoring", st.Severity.Info)
    st.OnScreenLogMessage(f"exponentArg = {exponentArg}", "SPSGuessrScoring", st.Severity.Info)

    thisScore: int = int(round(5000.0 * np.exp(-exponentArg)))
    # thisScore: int = 0
    st.OnScreenLogMessage(f"Score{currentLocIdx} = {thisScore}", "SPSGuessrScoring", st.Severity.Info)
    spsDevice.AddOrSetParam(st.VarType.int32, ["Scoring", f"Score{currentLocIdx}"], thisScore)
    
    score: int = spsDevice.GetParam(st.VarType.int32, "Score")
    score += thisScore
    spsDevice.SetParam(st.VarType.int32, "Score", score)

    if currentLocIdx > maxLocations:
        GameComplete()
        return


def NewSPSLocation(paramMap: st.ParamMap, timestamp: st.timestamp):
    currentLocIdx = spsDevice.GetParam(st.VarType.int32, "CurrentLocationIndex")
    currentLocIdx += 1
    spsDevice.SetParam(st.VarType.int32, "CurrentLocationIndex", currentLocIdx)

    st.SimGlobals.Publish("SPSEstimate", paramMap)


# Subscribe both SPS-setup-related functions
st.SimGlobals.Subscribe("NewSPSLocation", NewSPSLocation)
st.SimGlobals.Subscribe("SPSGuessrScore", SingleRoundScore)


exitFlag: bool = False
while not exitFlag:
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

    exitFlag = spsDevice.GetParam(st.VarType.bool, "GameComplete")

st.leave_sim()