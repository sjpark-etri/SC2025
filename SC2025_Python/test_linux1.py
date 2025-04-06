import SCAPI
import cv2
import os
inputFolder = "../Data/Sample1"
outputFolder = "../Data/RenderingResult3"

api = SCAPI.SCAPI()
api.SetInputFolder(inputFolder)
#api.GetRenderPathFromFile(inputFolder, "path.txt")
api.GetRenderPath(inputFolder, 1.0, 10.0, 100)
for i in range(api.numView):
    image = api.Rendering(i)
    cv2.imwrite(os.path.join(outputFolder, "%03d.png" % i), image)

api.Finalize()