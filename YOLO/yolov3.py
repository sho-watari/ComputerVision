import cv2
import numpy as np
import random

classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]

conf_threshold = 0.5
nms_threshold = 0.15
img_width = 416
img_height = 416
is_flip = False

color = {c: [random.randint(0, 255) for _ in range(3)] for c in range(len(classes))}
thickness = 2


def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def draw(frame, class_id, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), color[class_id], thickness=thickness)

    prob = "%s %.2f" % (classes[class_id], conf)
    label_size, base_line = cv2.getTextSize(prob, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.rectangle(frame, (left - 1, top + label_size[1] + base_line), (left + label_size[0], top),
                  color[class_id], cv2.FILLED)
    cv2.putText(frame, prob, (left, top + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


def postprocessing(frame, out):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    class_ids = []
    confidences = []
    boxes = []
    for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > conf_threshold:
            center_x = int(detection[0] * frame_width)
            center_y = int(detection[1] * frame_height)
            width = int(detection[2] * frame_width)
            height = int(detection[3] * frame_height)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            class_ids.append(classId)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box[0], box[1], box[2], box[3]
        draw(frame, class_ids[i], confidences[i], left, top, left + width, top + height)


if __name__ == "__main__":
    #
    # yolo
    #
    net = cv2.dnn.readNetFromDarknet("./yolov3.cfg", "./yolov3.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    #
    # web camera
    #
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    while True:
        _, frame = cap.read()

        #
        # yolo inference
        #
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (img_width, img_height), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        out = net.forward(get_outputs_names(net))[0]
        postprocessing(frame, out)

        if is_flip:
            frame = cv2.flip(frame, 1)

        cv2.imshow("YOLOv3", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
