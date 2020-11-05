import cntk as C
import cv2
import numpy as np
import random
import time

category = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

img_channel = 3
img_height = 416
img_width = 416

num_bboxes = 1330

conf_threshold = 0.6
nms_threshold = 0.1

thickness = 2
fontScale = 1.0
color = {c: [random.randint(0, 255) for _ in range(3)] for c in category}


def intersection_over_union(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin, xmax = max(box1[0], box2[0]), min(box1[2], box2[2])
    ymin, ymax = max(box1[1], box2[1]), min(box1[3], box2[3])

    intersection = np.maximum((xmax - xmin), 0) * np.maximum((ymax - ymin), 0)
    union = area1 + area2 - intersection

    return intersection / union


def non_maximum_suppression(img, bboxes, threshold):
    while bboxes != []:
        box1 = bboxes[0].copy()
        bboxes.pop(0)
        bboxes_cpy = bboxes.copy()
        for box2 in bboxes_cpy:
            iou = intersection_over_union(box1[2:], box2[2:])
            if iou >= threshold:
                bboxes.remove(box2)

        prob = "%s %.2f" % (box1[0], box1[1])
        (x, y), base = cv2.getTextSize(prob, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
        cv2.rectangle(img, (box1[2] + thickness, box1[3] + y + base + thickness), (box1[2] + x, box1[3]),
                      color[box1[0]], cv2.FILLED)
        cv2.putText(img, prob, (box1[2] + thickness, box1[3] + y + thickness), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                    (255, 255, 255))
        cv2.rectangle(img, (box1[2], box1[3]), (box1[4], box1[5]), color[box1[0]], thickness=thickness)

    return img


if __name__ == "__main__":
    model = C.load_model("./ssmd.model")

    cap_height = 1080
    cap_width = 1920

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)

    #
    # video demo
    #
    while True:
        _, frame = cap.read()

        start = time.perf_counter()

        output = model.eval({model.arguments[0]: np.ascontiguousarray(
            cv2.resize(frame, (img_width, img_height)).transpose(2, 0, 1), dtype="float32")})[0]

        bbox = output[:, :4]
        conf = output[:, 4]
        prob = C.softmax(np.ascontiguousarray(output[:, 5:], dtype="float32"), axis=1).eval().max(axis=1)
        label = output[:, 5:].argmax(axis=1)

        bboxes_dict = {}
        for i in range(num_bboxes):
            if conf[i] >= conf_threshold:
                xmin, ymin = bbox[i, 0] - bbox[i, 2] / 2, bbox[i, 1] - bbox[i, 3] / 2
                xmax, ymax = bbox[i, 0] + bbox[i, 2] / 2, bbox[i, 1] + bbox[i, 3] / 2

                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                if xmax > 1: xmax = 1
                if ymax > 1: ymax = 1

                bboxes_dict.setdefault(label[i], []).append([category[label[i]], prob[i],
                                                             int(xmin * cap_width), int(ymin * cap_height),
                                                             int(xmax * cap_width), int(ymax * cap_height)])

        for bboxes_list in bboxes_dict.values():
            bboxes_list.sort(reverse=True)
            frame = non_maximum_suppression(frame, bboxes_list, nms_threshold)

        cv2.imshow("Single Shot Multi Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        end = time.perf_counter()
        
        print("FPS %.1f" % (1.0 / (end - start)))

    cap.release()
    cv2.destroyAllWindows()
    
