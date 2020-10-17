import argparse
import cntk as C
import cv2
import h5py
import numpy as np
import random
import time

from layers import conv, batch_norm, max_pool

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
img_height = 608
img_width = 608

num_cell = 19
num_bboxes = 5

anchors_boxes = np.array([[0.57273, 0.677385],
                          [1.87446, 2.06253],
                          [3.33843, 5.47434],
                          [7.88282, 3.52778],
                          [9.77052, 9.16828]], dtype="float32")
conf_threshold = 0.5
nms_threshold = 0.1
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
        cv2.rectangle(img, (box1[2] - 1, box1[3] + y + base), (box1[2] + x, box1[3]), color[box1[0]], cv2.FILLED)
        cv2.putText(img, prob, (box1[2], box1[3] + y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255))
        cv2.rectangle(img, (box1[2], box1[3]), (box1[4], box1[5]), color[box1[0]], thickness=2)

    return img


def center_offset(num_boxes, num_cell):
    cx, cy = [], []
    x_cur, y_cur = -1, -1
    x_div, y_div = num_boxes, num_boxes * num_cell
    for i in range(num_boxes * num_cell * num_cell):
        if i % x_div == 0: x_cur += 1
        if i % y_div == 0: y_cur += 1
        if x_cur == num_cell: x_cur = 0
        cx.append([x_cur])
        cy.append([y_cur])
    return np.ascontiguousarray(np.concatenate((np.asarray(cx), np.asarray(cy)), axis=1), dtype="float32")


def prior_anchor(num_boxes, num_cell, anchor_boxes):
    pwh = []
    for i in range(num_boxes):
        pwh.append([anchor_boxes[i][0], anchor_boxes[i][1]])
    return np.ascontiguousarray(np.asarray(pwh * num_cell * num_cell), dtype="float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", nargs="?", dest="height", default=1080, help="camera height (default=1080)")
    parser.add_argument("--w", nargs="?", dest="width", default=1920, help="camera width (default=1920)")
    parser.add_argument("--f", nargs="?", dest="flip", default=False, help="fliplr (default=False)")

    args = parser.parse_args()

    #
    # yolo v2
    #
    h = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32")

    with h5py.File("./yolov2.h5", "r") as f:
        for l in range(20):
            h = conv(f["params/conv%d/weights" % (l + 1)][()])(h)
            h = batch_norm(f["params/bn%d/scale" % (l + 1)][()], f["params/conv%d/bias" % (l + 1)][()],
                           f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h)
            h = C.leaky_relu(h, alpha=0.1)

            if l in [0, 1, 4, 7, 12]:
                if l == 12:
                    h12 = h
                h = max_pool(h, ksize=2, stride=2)

        l += 1
        h12 = conv(f["params/conv%d/weights" % (l + 1)][()])(h12)
        h12 = batch_norm(f["params/bn%d/scale" % (l + 1)][()], f["params/conv%d/bias" % (l + 1)][()],
                         f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h12)
        h12 = C.leaky_relu(h12, alpha=0.1)

        reorg = C.transpose(C.reshape(h12, (64, num_cell, 2, num_cell, 2)), (0, 1, 3, 2, 4))
        reorg = C.transpose(C.reshape(reorg, (64, num_cell * num_cell, -1)), (0, 2, 1))
        reorg = C.reshape(h12, (256, num_cell, num_cell))

        h = C.splice(reorg, h, axis=0)

        l += 1
        h = conv(f["params/conv%d/weights" % (l + 1)][()])(h)
        h = batch_norm(f["params/bn%d/scale" % (l + 1)][()], f["params/conv%d/bias" % (l + 1)][()],
                       f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h)
        h = C.leaky_relu(h, alpha=0.1)

        l += 1
        h = conv(f["params/conv%d/weights" % (l + 1)][()], f["params/conv%d/bias" % (l + 1)][()].reshape(-1, 1, 1))(h)

    #
    # web camera
    #
    cap_height = int(args.height)
    cap_width = int(args.width)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)

    while True:
        _, frame = cap.read()

        start = time.perf_counter()

        x_img = np.ascontiguousarray(cv2.resize(frame, (img_width, img_height))[..., ::-1].transpose(2, 0, 1) / 255,
                                     dtype="float32")
        y = h.eval({h.arguments[0]: x_img})[0]

        bboxes_dict = {}
        for i in range(num_bboxes):
            obj_x, obj_y = np.where(y[4 + 85 * i, :, :] > conf_threshold)
            if len(obj_x) == 0:
                continue
            else:
                for ix, iy in zip(obj_x, obj_y):
                    y[85 * i:2 + 85 * i, ix, iy]
                    cx, cy = (C.sigmoid(y[85 * i:2 + 85 * i, ix, iy]).eval() + np.array([ix, iy])) * 32
                    ph, pw = C.exp(y[2 + 85 * i:4 + 85 * i, ix, iy]).eval() * anchors_boxes[i] * 32
                    prob = C.softmax(y[5 + 85 * i:85 * (i + 1), ix, iy]).eval()

                    if args.flip:
                        bboxes_dict.setdefault(category[prob.argmax()], []).append([
                            category[prob.argmax()], prob.max(),
                            cap_width - int((cy + ph / 2) * (cap_width / img_width)),
                            int((cx - pw / 2) * (cap_height / img_height)),
                            cap_width - int((cy - ph / 2) * (cap_width / img_width)),
                            int((cx + ph / 2) * (cap_height / img_height))])
                    else:
                        bboxes_dict.setdefault(category[prob.argmax()], []).append([
                            category[prob.argmax()], prob.max(),
                            int((cy - ph / 2) * (cap_width / img_width)),
                            int((cx - pw / 2) * (cap_height / img_height)),
                            int((cy + ph / 2) * (cap_width / img_width)),
                            int((cx + ph / 2) * (cap_height / img_height))])

        if args.flip:
            frame = cv2.flip(frame, 1)

        for bboxes_list in bboxes_dict.values():
            bboxes_list.sort(reverse=True)
            frame = non_maximum_suppression(frame, bboxes_list, nms_threshold)

        cv2.imshow("YOLOv2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        end = time.perf_counter()

        print("FPS %.1f" % (1.0 / (end - start)))
        
    cap.release()
    cv2.destroyAllWindows()

