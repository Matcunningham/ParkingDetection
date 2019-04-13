from yoloParkingDetector import yoloParkingDetector

vid = "lotVid.avi"
classFile = "yolov3.txt"
weightsFile = "yolov3.weights"
configFile = "yolov3.cfg"
yml = "parking.yml"
json = "parkingData.json"


# Pass in zero as first argument for live feed
myDetector = yoloParkingDetector(vid, classFile, weightsFile, configFile, yml, json)
myDetector.run()
