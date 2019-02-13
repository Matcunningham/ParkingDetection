
import cv2
import numpy as np


video = "lotVidEnd.avi"
classFile = "yolov3.txt"
weightsFile = "yolov3.weights"
configFile = "yolov3.cfg"

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(img, "{0:.2f} %".format(confidence), (x-10,y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




############### MAIN ###############

cap = cv2.VideoCapture(video)

while(cap.isOpened()):
    currentPosition = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Index of the frame to be decoded/captured next

    success, image = cap.read()

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classFile, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weightsFile, configFile)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        print(str(class_ids[i]) + " " + str(confidences[i]) + " X: " + str(x) + " Y: " + str(y) + " W: " + str(w) + " H: " + str(h))
    cv2.imshow("object detection", image)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('j'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame + 500)  # jump 500 frames forward
    elif k == ord('u'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame - 500)  # jump 500 frames back
    # if cv2.waitKey(33) == 27:
    elif k == 27:
        break

    cv2.imwrite("object-detection.jpg", image)
cap.release()
cv2.destroyAllWindows()
