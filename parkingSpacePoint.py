import cv2
import yaml
import easygui
class parkingSpacePoint:
    def __init__(self, img, file):
        self.image = img
        self.filePath = file
        self.parkingSpace = []
        self.id = 1
        self.data = []


    def dumpYML(self):
        with open(self.filePath, "a") as yml:
            yaml.dump(self.data, yml)


    def definePoints(self, event, x, y, flags, param):
        currentSpace = {'id': None, 'point': None}                                      # Initialize dictionary for 1st parking space
        if event == cv2.EVENT_LBUTTONDBLCLK:                                        # If a point on the image is double left clicked
            self.parkingSpace.append((x,y))                                         # Append the point to parkingSpace
            currentSpace['point']=(list(self.parkingSpace[0]))                      # Get point
            id = easygui.enterbox("Enter ID number for this spot")                  # Get ID
            currentSpace['id'] = id
            self.data.append(currentSpace)
            self.parkingSpace = []

            cv2.putText(self.image, id, (x, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, (0, 0, 255), 2)


    def markSpaces(self):
        cv2.namedWindow("Double click to mark points")                                  # Name window
        cv2.imshow("Double click to mark points", self.image)                                  # Set captured frame and show
        cv2.setMouseCallback("Double click to mark points", self.definePoints)           # Set double left click action

        while True:                                                                     # Set parking space boundaries and loop until ESC is pressed
            cv2.imshow("Double click to mark points", self.image)
            key = cv2.waitKey(1) & 0xFF                                                 # 0xFF to ensure we only get the last 8 bits of ASCII character input
            if cv2.waitKey(33) == 27:                                                   # If ESC key is pressed, break
                break

        if self.data != []:                                                                  # After breaking loop, dump collected parking data if not null
            self.dumpYML()
        cv2.destroyAllWindows()                                                         # Close parking boundary window

