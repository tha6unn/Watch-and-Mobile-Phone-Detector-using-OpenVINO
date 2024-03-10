from openvino_detector_2022_3.model_api.performance_metrics import PerformanceMetrics
from time import perf_counter

import openvino as ov
core = ov.Core()

import cv2  # Install opencv-python
import numpy as np

class_names = open("labels.txt", "r").readlines()
model = core.read_model("tm.xml")
compiled_model = core.compile_model(model=model)

camera = cv2.VideoCapture(0)
metrics = PerformanceMetrics()
import time
while True:
    
    start_time = perf_counter()
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    metrics.update(start_time, image)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    output_key = compiled_model.output(0)

    op = compiled_model(image)[output_key]
    op = np.argmax(op)
    print(class_names[op])

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()