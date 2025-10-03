from roboflow import Roboflow
rf = Roboflow(api_key="ZYUdahDEZ8kU1Ug660c3")
project = rf.workspace("earlydrowningdetection").project("near_drowning_detector-gxt56")
version = project.version(7)
dataset = version.download("coco")