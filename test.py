import cv2

test = ['./images/zebra (66).jpg', './images/buffalo (15).jpg', './images/elephant (48).jpg',
        './images/rhino (21).jpg', './images/raccoon-1_jpg.rf.4735f8b019bd0bfa86593b474aa7a5fa.jpg']

img = cv2.imread(test[3])
print((img))
with open('wild.names', 'r') as f:
    classes = f.read().splitlines()
print(classes)

net = cv2.dnn.readNetFromDarknet(
    'yolov3_custom_train.cfg', 'yolov3_custom_train.backup')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

classIds, scores, boxes = model.detect(
    img, confThreshold=0.1, nmsThreshold=0.4)

for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2],
                  box[1] + box[3]), color=(0, 255, 0), thickness=2)
    text = '%s: %.2f' % (classes[classId], score)
    cv2.putText(img, text, (box[0] + 5, box[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
