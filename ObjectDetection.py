import numpy as np
import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x+5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]

    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(
        outputs, confidence, width, height)

    # nms
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs


def detect(img, conf, nms):
    # get classes
    with open('yolo copy.names', 'r') as f:
        classes = f.read().splitlines()

    # load yolo model
    net = cv2.dnn.readNetFromDarknet(
        'yolov3_custom_train.cfg', 'yolov3_custom_train_600.weights')

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1]
                   for i in net.getUnconnectedOutLayers()]

    # detection
    boxes, confidences, classIDs, idxs = make_prediction(
        net, layer_names, classes, img, conf, nms)

    # draw bounding boxes
    img = draw_bounding_boxes(
        img, boxes, confidences, classIDs, idxs, classes)

    # display detected img
    st.subheader("Object-Detected Image")
    plt.imshow(img)
    st.pyplot()


if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Aquarium Detection")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.00, 1.00, 0.5, 0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

    # upload image
    imgfile = st.file_uploader(
        "Upload Image to Detect", type=['jpg', 'png', 'jpeg'])

    # detect
    if imgfile is not None:
        img = Image.open(imgfile)

        # display original image
        st.subheader("Original Image")
        plt.imshow(img)
        st.pyplot()

        # convert to np array
        pil_image = img.convert('RGB')
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        detect(open_cv_image, conf_threshold, nms_threshold)
