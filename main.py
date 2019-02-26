from yoloParkingDetector import yoloParkingDetector

vid = "lotVid.avi"
classFile = "yolov3.txt"
weightsFile = "yolov3.weights"
configFile = "yolov3.cfg"
yml = "parking.yml"
json = "parkingData.json"



myDetector = yoloParkingDetector(vid, classFile, weightsFile, configFile, yml, json)
myDetector.run()