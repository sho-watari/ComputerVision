import cv2
import numpy as np

#
# [category, confidence, xmin, ymin, xmax, ymax]
#
bboxes_dict = {"bicycle": [["bicycle", 0.4,   5, 100, 320, 280],
                           ["bicycle", 0.9,  10,  80, 340, 330],
                           ["bicycle", 0.3,  50,  72, 330, 257]],
               "car": [["car", 0.8, 250,  40, 416, 128],
                       ["car", 0.5, 243,  75, 400, 126]],
               "dog":[["dog", 0.75,  22, 213, 135, 385],
                      ["dog", 0.8,  17, 161, 140, 380]]}
fontScale = 0.3
nms_threshold = 0.1


def intersection_over_union(box1, box2):  # [xmin, ymin, xmax, ymax]
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

            if iou > threshold:
                bboxes.remove(box2)

        prob = "%s %.2f" % (box1[0], box1[1])
        (x, y), base = cv2.getTextSize(prob, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
        cv2.rectangle(img, (box1[2] - 1, box1[3] + y + base), (box1[2] + x, box1[3]), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, prob, (box1[2], box1[3] + y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255))
        cv2.rectangle(img, (box1[2], box1[3]), (box1[4], box1[5]), (0, 0, 255), thickness=2)
    return img


if __name__ == "__main__":
    img = cv2.imread("./bicycle-car-dog.jpg")

    #
    # Nun Maximum Suppression
    #
    for bboxes_list in bboxes_dict.values():
        bboxes_list.sort(reverse=True)
        img = non_maximum_suppression(img, bboxes_list, nms_threshold)

    cv2.imshow("Non Maximum Suppression", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
