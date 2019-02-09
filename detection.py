#----------------------Dependencies----------------------------#
from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt

execution_path = os.getcwd()
color_index = {'car':'red','truck':'blue','motorcycle':'orange'}
#-------------------------Functions----------------------------------------------#
"""
def forFull(output_arrays, count_arrays, average_output_count):
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the entire video: ", average_output_count)
    print("------------END OF THE VIDEO --------------")
"""

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

#-------------------------------------------------------------------------------#
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path,"yolo.h5"))
detector.loadModel(detection_speed="fastest")
plt.show()

custom_objects = detector.CustomObjects(car=True,truck=True,motorcycle=True)

#--------------------------------Features---------------------------------------#
video_path = detector.detectCustomObjectsFromVideo(
minimum_percentage_probability=50,
custom_objects=custom_objects,
per_second_function=forSeconds,
display_percentage_probability=True,
display_object_name=True,
log_progress=True,
frames_per_second=60,
#--------------------Input-File--------------------------------------------#
input_file_path=os.path.join(execution_path,"cars.mp4"),
#--------------------Output-File-----------------------------------------#
output_file_path=os.path.join(execution_path, "car_detection"))
#---------------------Finish------------------------------------------#
print(video_path)
