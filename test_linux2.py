import sys
sys.path.append('/etri_workspace')
import SCAPI
import subprocess

inputFolder = "./Data/Sample2"

api = SCAPI.SCAPI()
m = api.SetInputFolder(inputFolder, 4)
#m = api.MakeLayer(inputFolder, 4)
print(m)
#api.SetInputFolder(inputFolder)
