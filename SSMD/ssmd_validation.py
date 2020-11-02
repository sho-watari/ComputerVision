import cntk as C
import cv2
import json
import numpy as np
import os
import pickle
import random

from sklearn.metrics import average_precision_score

categories = {"1": [0, "person"], "2": [1, "bicycle"], "3": [2, "car"], "4": [3, "motorcycle"], "5": [4, "airplane"],
              "6": [5, "bus"], "7": [6, "train"], "8": [7, "truck"], "9": [8, "boat"], "10": [9, "traffic light"],
              "11": [10, "fire hydrant"], "13": [11, "stop sign"], "14": [12, "parking meter"], "15": [13, "bench"],
              "16": [14, "bird"], "17": [15, "cat"], "18": [16, "dog"], "19": [17, "horse"], "20": [18, "sheep"],
              "21": [19, "cow"], "22": [20, "elephant"], "23": [21, "bear"], "24": [22, "zebra"], "25": [23, "giraffe"],
              "27": [24, "backpack"], "28": [25, "umbrella"], "31": [26, "handbag"], "32": [27, "tie"],
              "33": [28, "suitcase"], "34": [29, "frisbee"], "35": [30, "skis"], "36": [31, "snowboard"],
              "37": [32, "sports ball"], "38": [33, "kite"], "39": [34, "baseball bat"], "40": [35, "baseball glove"],
              "41": [36, "skateboard"], "42": [37, "surfboard"], "43": [38, "tennis racket"], "44": [39, "bottle"],
              "46": [40, "wine glass"], "47": [41, "cup"], "48": [42, "fork"], "49": [43, "knife"], "50": [44, "spoon"],
              "51": [45, "bowl"], "52": [46, "banana"], "53": [47, "apple"], "54": [48, "sandwich"],
              "55": [49, "orange"], "56": [50, "broccoli"], "57": [51, "carrot"], "58": [52, "hot dog"],
              "59": [53, "pizza"], "60": [54, "donut"], "61": [55, "cake"], "62": [56, "chair"], "63": [57, "couch"],
              "64": [58, "potted plant"], "65": [59, "bed"], "67": [60, "dining table"], "70": [61, "toilet"],
              "72": [62, "tv"], "73": [63, "laptop"], "74": [64, "mouse"], "75": [65, "remote"], "76": [66, "keyboard"],
              "77": [67, "cell phone"], "78": [68, "microwave"], "79": [69, "oven"], "80": [70, "toaster"],
              "81": [71, "sink"], "82": [72, "refrigerator"], "84": [73, "book"], "85": [74, "clock"],
              "86": [75, "vase"], "87": [76, "scissors"], "88": [77, "teddy bear"], "89": [78, "hair drier"],
              "90": [79, "toothbrush"]}

category = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

dir_file = "../COCO/COCO"
data_file = "val2014"

img_channel = 3
img_height = 416
img_width = 416

num_bboxes = 1330
num_classes = 80
num_channel = 5 + num_classes

anchor_boxes = np.load("./anchor_boxes.npy")

conf_threshold = 0.5
nms_threshold = 0.1

thickness = 2
fontScale = 0.3
color = {c: [random.randint(0, 255) for _ in range(3)] for c in category}


def intersection_over_union(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin, xmax = max(box1[0], box2[0]), min(box1[2], box2[2])
    ymin, ymax = max(box1[1], box2[1]), min(box1[3], box2[3])

    intersection = np.maximum((xmax - xmin), 0) * np.maximum((ymax - ymin), 0)
    union = area1 + area2 - intersection
    return intersection / union


def non_maximum_suppression(img, bboxes, threshold, nms_list):
    while bboxes != []:
        box1 = bboxes[0].copy()
        nms_list.append(box1)
        bboxes.pop(0)
        bboxes_cpy = bboxes.copy()
        for box2 in bboxes_cpy:
            iou = intersection_over_union(box1[2:], box2[2:])
            if iou >= threshold:
                bboxes.remove(box2)

        prob = "%s %.2f" % (box1[0], box1[1])
        (x, y), base = cv2.getTextSize(prob, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
        cv2.rectangle(img, (box1[2] + thickness, box1[3] + y + base + thickness), (box1[2] + x, box1[3]),
                      (0, 0, 255), cv2.FILLED)
        cv2.putText(img, prob, (box1[2] + thickness, box1[3] + y + thickness), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                    (255, 255, 255))
        cv2.rectangle(img, (box1[2], box1[3]), (box1[4], box1[5]), color[box1[0]], thickness=thickness)

    return img, nms_list


def ssmd_meanAP(iou_score):
    instance_file = "%s/annotations/instances_%s.json" % (dir_file, data_file)

    if not os.path.exists(data_file + "_ssmd_meanAP.pkl"):
        with open(instance_file, "rb") as file:
            dataset = json.load(file)

        annotations = dataset["annotations"]

        bbox_dict = {}
        for ann in annotations:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            bbox = ann["bbox"]
            bbox.append(categories[str(category_id)][0])

            bbox_dict.setdefault(image_id, []).append(bbox)

        val_dict = {}
        for image_id, bbox_list in bbox_dict.items():
            filename = "{:s}/{:s}/COCO_{:s}_{:0>12s}.jpg".format(dir_file, data_file, data_file, str(image_id))

            img = cv2.imread(filename)
            height, width, _ = img.shape

            target_dict = {}
            for bbox in bbox_list:
                box = [(bbox[0] / width) * img_width, (bbox[1] / height) * img_height,
                       ((bbox[0] + bbox[2]) / width) * img_width, ((bbox[1] + bbox[3]) / height) * img_height]
                target_dict.setdefault(bbox[4], []).append(box)

            val_dict[filename] = target_dict

        with open("./" + data_file + "_ssmd_meanAP.pkl", "wb") as file:
            pickle.dump(val_dict, file)
        print("Saved " + data_file + "_ssmd_meanAP.pkl")
    else:
        with open("./" + data_file + "_ssmd_meanAP.pkl", "rb") as file:
            val_dict = pickle.load(file)

    model = C.load_model("./ssmd.model")

    #
    # mean average precision
    #
    meanAP = dict()
    for label in category:
        meanAP[label] = []

    for file, true_dict in val_dict.items():
        img = cv2.resize(cv2.imread(file), (img_width, img_height))
        output = model.eval({model.arguments[0]: np.ascontiguousarray(img.transpose(2, 0, 1), dtype="float32")})[0]

        bbox = output[:, :4]
        conf = output[:, 4]
        prob = C.softmax(np.ascontiguousarray(output[:, 5:], dtype="float32"), axis=1).eval().max(axis=1)
        label = output[:, 5:].argmax(axis=1)

        #
        # objectness processing
        #
        pred_dict = {}
        for i in range(num_bboxes):
            if conf[i] >= conf_threshold:
                xmin, ymin = bbox[i, 0] - bbox[i, 2] / 2, bbox[i, 1] - bbox[i, 3] / 2
                xmax, ymax = bbox[i, 0] + bbox[i, 2] / 2, bbox[i, 1] + bbox[i, 3] / 2

                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                if xmax > 1: xmax = 1
                if ymax > 1: ymax = 1

                pred_dict.setdefault(label[i], []).append([category[label[i]], prob[i],
                                                           int(xmin * img_width), int(ymin * img_height),
                                                           int(xmax * img_width), int(ymax * img_height)])
        #
        # non maximum suppression
        #
        nms_dict = {}
        for key, pred_bboxes in pred_dict.items():
            nms_list = []
            pred_bboxes.sort(reverse=True)
            img, nms_list = non_maximum_suppression(img, pred_bboxes, nms_threshold, nms_list)
            nms_dict[key] = nms_list

        #
        # average precision
        #
        for true_label, true_bboxes in true_dict.items():
            if true_label in nms_dict:
                for true_bbox in true_bboxes:
                    y_true = []
                    y_score = []
                    for pred_bbox in nms_dict[true_label]:
                        iou_mAP = intersection_over_union(true_bbox, pred_bbox[2:])
                        if iou_mAP > 0:
                            y_score.append(pred_bbox[1])
                            if iou_mAP >= iou_score:
                                y_true.append(1)
                            else:
                                y_true.append(0)

                    if y_true != []:
                        average_precision = average_precision_score(y_true, y_score)
                        if np.isnan(average_precision):
                            average_precision = 0.0
                    else:
                        average_precision = 0.0

                    meanAP[category[true_label]].append(average_precision)
            else:
                average_precision = 0.0
                meanAP[category[true_label]].append(average_precision)

    meanAP_score = dict()
    for key, value in meanAP.items():
        meanAP_score[key] = np.array(value).mean()

    print("mAP@%d %.1f" % (iou_score * 100, np.array(list(meanAP_score.values())).mean() * 100))


if __name__ == "__main__":
    ssmd_meanAP(0.5)
    
