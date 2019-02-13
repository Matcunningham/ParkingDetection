# import required packages
import cv2
import argparse
import numpy as np

# handle command line arguments
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')

ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')

ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')

ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')

args = ap.parse_args()

# read input image
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

# read class names from text file
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(args.weights, args.config)

# create input blob
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
net.setInput(blob)