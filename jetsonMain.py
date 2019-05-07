from yoloParkingDetectorJetson import yoloParkingDetector

vid = 0
classFile = "yolov3.txt"
weightsFile = "yolov3.weights"
configFile = "yolov3.cfg"
yml = "parking.yml"
json = "parkingData.json"


# Pass in zero as first argument for live feed
myDetector = yoloParkingDetector(vid, classFile, weightsFile, configFile, yml, json)
myDetector.run()
