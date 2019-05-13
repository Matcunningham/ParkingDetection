import cv2
import numpy as np
import yaml
from parkingSpacePoint import parkingSpacePoint
from os.path import isfile
from requests import post, put, get
import threading

cap = cv2.VideoCapture()
imgList = []
lock = threading.Lock()

# Used to get the latest frame, mainly for live feed
class CaptureThread(threading.Thread):

    def __init__(self):
        super(CaptureThread, self).__init__()
        self.start()

    def run(self):
        try:
            global cap
            while cap.isOpened():
                global imgList
                global lock
                lock.acquire()
                imgList[0] = cap.read()
                if not (imgList[0])[0]:
                    cap.release()
                    raise SystemExit
                lock.release()
        except:
            cap.release()
            

class yoloParkingDetector:
    
    def __init__(self, video, classFile, weightsFile, configFile, ymlFile, lotID, cameraNum):
        global cap
        self.video = video                                                  # Video file location

        self.classFile = classFile                                          # Used with classifier
        self.weightsFile = weightsFile
        self.configFile = configFile

        self.ymlFile = ymlFile                                              # Parking Space Point yml file


        self.URL = "http://54.186.186.248:3000/api/status"                                                    # Server URL and port
        self.camURL = "http://54.186.186.248:3000/api/camerastatus" 
        self.authenticationToken = "JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6InRlc3RDYW1lcmE1IiwiaWF0IjoxNTUxMjMzNDE4fQ.JvRJRkLppw1psQbroOHyURxPiyVJguP3p-JeY2vwjsw"                                       # Authentication Token file location, must be obtained from server. Used to prevent unauthorized posts
        self.cameraNum = cameraNum                                                 # Camera number used to identify where the info is coming from
        self.parkinglot_ID = lotID

        cap = cv2.VideoCapture(self.video)                             # CV2 open video file

        self.parkingSpaceData = None                                        # Stores all of our parking space points
        self.parkingStatus = []                                             # Stores the status of parking spaces
        self.parkingStatusInit = []                                         # Stores base status

        self.net = None                                                     # Net object used in classification
        self.classes = None                                                 # Classes read from classFile
        self.COLORS = None                                                  # Colors used for our classes

        self.isGetNecessary = True                                          # True for 1st iteration in run(), checks if we need to POST
        self.visualize = True                                               # Create and display window and draw rectangle

        self.threadToggle = False                                           # Toggle for threading, Set to FALSE for a video.

    def run(self):
        try:
            global imgList
            global cap
            if(self.threadToggle):
                imgList = [cap.read()]
                CaptureThread()
            
            self.openYML()  # Open ymlFile and save parking space boundary information

            with open(self.classFile, 'r') as f:                                              # Get classes from class file
                self.classes = [line.strip() for line in f.readlines()]
            self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))                      # Set colors for each class
            self.net = cv2.dnn.readNet(self.weightsFile, self.configFile)                           # Construct net object using our pre-trained classifier model and config files

            # Parking detection begins
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 1700)                                     # Jump to frame number specified

            while (cap.isOpened()):
                #currentPosition = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
                #currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Index of the frame to be decoded/captured next

                success = False
                if(self.threadToggle):
                    global lock
                    lock.acquire()
                    success, initialFrame = imgList[0] #self.cap.read()  # Capture frame
                    lock.release()
                else:
                    success, initialFrame = cap.read()

                if success:
                    frame = initialFrame
                else:
                    print("Video ended")
                    raise SystemExit


                Width = frame.shape[1]
                Height = frame.shape[0]
                scale = 0.00392

                # Prepare the frame to run through the deep neural network
                # blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)             # Create 4-dimensional blob from current frame
                blob = cv2.dnn.blobFromImage(frame, scale, (320, 320), (0, 0, 0), True, crop=False)             # Create 4-dimensional blob from current frame
                self.net.setInput(blob)                                                                         # Set frame blob as the input for out networ
                
                # Get output layers from network and then run interference through the network and gather predictions from output layers
                outs = self.net.forward(self.get_output_layers())

                # Initialize necessary data members
                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.4

                # For each detection from each output layer get the confidence, class id, and bounding box parameters
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:                                # Ignore all weak detections (less than 50% confidence)
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])
                # Apply non-max suppression  (removes boxes with high overlapping)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # Reset all spaces to empty before checking if vehicle in space
                for spaces in self.parkingSpaceData:
                    id = spaces['id']
                    self.updateStatus(id, 0)
                    
                # Go through boxes after non-max suppression and draw bounding boxes for vehicles detected inside of parking spaces
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]

                    x1 = x+w
                    y1 = y+h

                    # See rectangle contains any of the parking space points
                    for spaces in self.parkingSpaceData:
                        id = spaces['id']
                        point  = spaces['point']
                        xPoint = point[0]
                        yPoint = point[1]

                        # If rectangle contains a space point update status entry, draw the prediction, and break for loop
                        if(x < xPoint < x1 and y < yPoint < y1):
                            self.updateStatus(id, confidences[i])
                            if(self.visualize):
                                self.draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x1),round(y1), spaces['id'])
                            break

                print(self.parkingStatus)

                if self.isGetNecessary:
                    self.isGetNecessary = False
                    getReq = get(self.URL + "/" + str(self.parkinglot_ID))
                    if getReq.text == '':
                        self.postStatus()
                self.putStatus()

                self.parkingStatus = self.parkingStatusInit                             # Reset parking status for next frame
                if(self.visualize):
                    cv2.imshow("object detection", frame)  # Display the frame


                k = cv2.waitKey(1) # Time set low to speed up raspi, also get error on raspi using CAP_PROP... below
                if k == ord('q'):
                    break
                # elif k == ord('j'):
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame + 500)  # jump 500 frames forward
                # elif k == ord('u'):
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame - 500)  # jump 500 frames back
                # if cv2.waitKey(33) == 27:
                elif k == 27:
                    break
                cv2.imwrite("object-detection.jpg", frame)

            ### After breaking from while loop, release video input and close cv2 window
            cap.release()
            cv2.destroyAllWindows()
        except:
            cap.release()

    ### Get output layer names in the architecture ###
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    ### Draw prediction on frame ###
    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h, parkingID):
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, parkingID, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, "{0:.2f} %".format(confidence), (x - 10, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    ### Posts status updates to our server using HTTP POST ###
    def postStatus(self):
        print("Sending status data...")
        # head = {'Authorization': self.authenticationToken}
        status = self.parkingStatus
        body = {"parkinglot_ID": self.parkinglot_ID, "camera_ID": self.cameraNum, "status":status}
        print(body)
        try:
            msgResponse = post(self.URL, json=body)                         # Post message
            msgResponse2 = post(self.camURL, json=body)
            return msgResponse, msgResponse2
        except:
            raise ValueError('Status could not be posted')                                  # If post message fails catch error and print message

    def putStatus(self):
        print("Sending status data...")
        # head = {'Authorization': self.authenticationToken}
        status = self.parkingStatus
        body = {"parkinglot_ID": self.parkinglot_ID, "camera_ID": self.cameraNum, "status":status}
        putURL = self.URL+"/"+ str(self.parkinglot_ID)
        try:
            msgResponse = put(putURL, json=body)  # Post message
            return msgResponse
        except:
            raise ValueError('Status could not be posted')  # If post message fails catch error and print message

    def updateStatus(self, id, confidence):
        for entry in self.parkingStatus:
            if entry['id'] == id:
                entry['confidence'] = confidence

    ### Opens ymlFile and loads data, if ymlFile does not exist it prompts the user to define parking space points for detection ###
    def openYML(self):
        # Read YAML data (parking space polygons)
        if isfile(self.ymlFile):                                # If yml file exists open it and load parking space data
            with open(self.ymlFile, 'r') as stream:
                self.parkingSpaceData = yaml.load(stream)
        else:                                                   # Else create yml file then load it
            global cap
            success, image = cap.read()                    # Capture frame to mark parking spaces
            if success:                                         # If frame can be captured
                ymlImg = image
                mySpace = parkingSpacePoint(ymlImg, self.ymlFile)   # Create parkingSpaceBoundary object and pass it resized frame and destination yml file path
                mySpace.markSpaces()                                        # Run function to mark parking space boundaries
                del mySpace                                                 # Delete object once done

                with open(self.ymlFile, 'r') as stream:             # Open new yml file and load it
                    self.parkingSpaceData = yaml.load(stream)
            else:                                                           # If frame cannot be captured then print error message and exit
                print("Video could not be opened, parking boundaries cannot be established")
                raise SystemExit


        if self.parkingSpaceData != None:
            for spaces in self.parkingSpaceData:                            # Initialize parkingStatusInit
                id = spaces['id']
                entry = {'id':id, 'confidence': 0}
                self.parkingStatusInit.append(entry)
            self.parkingStatus = self.parkingStatusInit                     # Initialize parkingStatusInit


# ________________________€€€€€€€€€€€€€€€€€€€________
# ____________________€€€___________________€€€______
# ________________€€€_________________________€€€€___
# ______________€€______________________________€€€__
# ___________€€€_________________________________€€€_
# _________€€_____________________________________€€€
# ________€€_________€€€€€___________€€€€€_________€€
# ______€€__________€€€€€€__________€€€€€€_________€€
# _____€€___________€€€€____________€€€€___________€€
# ____€€___________________________________________€€
# ___€€___________________________________________€€_
# __€€€_____€____________________________€€______€€€_
# _€€€______€___________________________€€_______€€__
# _€€_________€________________________€€_______€€€__
# _€€€_________€€€____________________€€_______€€€___
# _€€____________€€€€____________€€€€€_________€€____
# _€€_______________€€€€€€€€€€€€€€€___________€€_____
# _€€_______________€€€€€€€€€€€€€€€__________€€______
# __€€____________________€€€€______________€€_______
# _€€€_____________________________________€€________
# __€€€__________________________________€€__________
# ___€€€_______________________________€€____________
# ____€€€€__________________________€€€______________
# _______€€€€€_                  _€€€€€______________
# _____________€€€€€€€€€€€€€€€€€_____________________
