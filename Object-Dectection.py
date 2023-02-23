from ultralytics import YOLO, yolo
import os
from IPython.display import display, Image
from apiKey import API_Key
from IPython import display
display.clear_output()
# !yolo mode=checks

from roboflow import Roboflow
rf = Roboflow(api_key=API_Key)
project = rf.workspace().project("trees-ans9j")
model = project.version(2).model

# infer on a local image


# visualize your prediction
model.predict("testImage.jpg", confidence=60, overlap=70).save("prediction.jpg")