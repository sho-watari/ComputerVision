import cv2
import json
import numpy as np

from sklearn.cluster import k_means

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

dir_file = "../COCO/COCO"
data_file = "train2014"

img_channel = 3
img_height = 416
img_width = 416
num_classes = 80
num_anchors = 5


def dimension_clustering(anchor_boxes, num_anchors):
    centroid, label, _ = k_means(anchor_boxes, num_anchors)

    np.save("anchor_boxes.npy", centroid)
    print("\nSaved anchor_boxes.npy")


if __name__ == "__main__":
    """ Bounding Box and Category Label from COCO dataset """
    instance_file = "%s/annotations/instances_%s.json" % (dir_file, data_file)

    with open(instance_file, "rb") as file:
        dataset = json.load(file)

    annotations = dataset["annotations"]

    #
    # image_id and bbox dictionary
    #
    bbox_dict = {}
    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]
        bbox.append(categories[str(category_id)][0])

        bbox_dict.setdefault(image_id, []).append(bbox)

    #
    # mapfile and bounding-box file
    #
    anchor_boxes = np.zeros((len(annotations), 2), dtype="float32")

    num_samples = 0
    num_bboxes = 0

    map_file = open("./" + data_file + "_ssmd_images.txt", "w")
    box_file = open("./" + data_file + "_ssmd_bboxes.txt", "w")
    for image_id, bbox_list in bbox_dict.items():
        filename = "{:s}/{:s}/COCO_{:s}_{:0>12s}.jpg".format(dir_file, data_file, data_file, str(image_id))
        map_file.write("%s\t%d\n" % (filename, 0))

        img = cv2.imread(filename)
        height, width, _ = img.shape

        for bbox in bbox_list:
            box = [(bbox[0] + bbox[2] / 2) / width, (bbox[1] + bbox[3] / 2) / height, bbox[2] / width, bbox[3] / height]
            box_file.write("{}\t|bbox {}\t|label {}:1\n".format(num_samples, " ".join(["%e" % b for b in box]), bbox[4]))

            anchor_boxes[num_bboxes, :] = box[2:]
            num_bboxes += 1

        num_samples += 1
        if num_samples % 10000 == 0:
            print("Now %d samples..." % num_samples)

    map_file.close()
    box_file.close()

    print("\nNumber of samples", num_samples)

    #
    # Dimension Clustering
    #
    dimension_clustering(anchor_boxes, num_anchors)
    
