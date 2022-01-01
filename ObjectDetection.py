import numpy as np
import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def detect(img, conf, nms):
    # display original image
    st.subheader("Original Image")
    plt.imshow(img)
    st.pyplot()

    # get yolo model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # get classes for detection
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # get layer from yolo
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1]
                     for i in net.getUnconnectedOutLayers()]

    # convert img
    img = cv2.cvtColor(np.array(img.convert('RGB')), 1)
    height, width, channels = img.shape

    # convert to blob
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), swapRB=True, crop=False)

    # get output from yolo model
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # get the coordinates of object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, conf, nms)

    # draw rectangle and label
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str.upper((classes[class_ids[i]]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(img, label, (x+5, y+25),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))

    # display detected img
    st.subheader("Object-Detected Image")
    plt.imshow(img)
    st.pyplot()


if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("cai gi do Detection")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.00, 1.00, 0.5, 0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

    # upload image
    imgfile = st.file_uploader(
        "Upload Image to Detect", type=['jpg', 'png', 'jpeg'])

    # detect
    if imgfile is not None:
        img = Image.open(imgfile)
        detect(img, conf_threshold, nms_threshold)
