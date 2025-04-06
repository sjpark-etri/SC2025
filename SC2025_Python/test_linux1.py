import SCAPI
import cv2
import os
inputFolder = "../Data/Sample1"
outputFolder = "../Data/RenderingResult3"

api = SCAPI.SCAPI()
api.SetInputFolder(inputFolder)
res = api.FullRendering(1.0, 10.0, 100)
for i in range(api.numView):    
    cv2.imwrite(os.path.join(outputFolder, "%03d.png" % i), res[i,...])
api.Finalize()