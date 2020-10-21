import cntk as C
import cv2
import h5py
import numpy as np
import os
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

cap_width = 640
cap_height = 800

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
conf_threshold = 0.6
nms_threshold = 0.1
fontScale = 0.5
color = {c: [random.randint(0, 255) for _ in range(3)] for c in category}
is_flip = False

layers = [[3, 32, 3],
          [32, 64, 3],
          [64, 128, 3],
          [128, 64, 1],
          [64, 128, 3],
          [128, 256, 3],
          [256, 128, 1],
          [128, 256, 3],
          [256, 512, 3],
          [512, 256, 1],
          [256, 512, 3],
          [512, 256, 1],
          [256, 512, 3],
          [512, 1024, 3],
          [1024, 512, 1],
          [512, 1024, 3],
          [1024, 512, 1],
          [512, 1024, 3],
          [1024, 1024, 3],
          [1024, 1024, 3]]


def extract_parameter():
    file = open("./yolov2.weights", "rb")

    with h5py.File("./yolov2.h5", "w") as f:
        data = np.fromfile(file, dtype="float32")[4:]

        offset = 0
        for i, l in enumerate(layers):
            in_ch, out_ch, ksize = l[0], l[1], l[2]

            f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/bn%d/scale/" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/bn%d/mean" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/bn%d/variance" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/conv%d/weights" % (i + 1),
                             data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                                 out_ch, in_ch, ksize, ksize))
            offset += (out_ch * in_ch * ksize * ksize)

        in_ch, out_ch, ksize, i = 512, 64, 1, 20

        f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/scale/" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/mean" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/variance" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/conv%d/weights" % (i + 1),
                         data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                             out_ch, in_ch, ksize, ksize))
        offset += (out_ch * in_ch * ksize * ksize)

        in_ch, out_ch, ksize, i = 1280, 1024, 3, 21

        f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/scale/" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/mean" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/variance" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/conv%d/weights" % (i + 1),
                         data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                             out_ch, in_ch, ksize, ksize))
        offset += (out_ch * in_ch * ksize * ksize)

        in_ch, out_ch, ksize, i = 1024, 425, 1, 22

        f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/conv%d/weights" % (i + 1),
                         data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                             out_ch, in_ch, ksize, ksize))
        offset += (out_ch * in_ch * ksize * ksize)


def convolution(weights, pad=True, stride=1, name=''):
    W = C.Constant(value=weights, name='W')

    @C.BlockFunction('Convolution2D', name)
    def conv2d(x):
        return C.convolution(W, x, strides=[stride, stride], auto_padding=[False, pad, pad])

    return conv2d


def batch_normalization(scale, bias, mean, variance, spatial=True, name=''):
    scale = C.Constant(value=scale, name='scale')
    bias = C.Constant(value=bias, name='bias')
    mu = C.Constant(value=mean, name='aggreagate_mean')
    sigma = C.Constant(value=variance, name='aggregate_variance')

    @C.BlockFunction('BatchNormalization', name)
    def batch_norm(x):
        return C.batch_normalization(x, scale, bias, mu, sigma, spatial=spatial, running_count=C.constant(5000))

    return batch_norm


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


def yolov2(h):
    with h5py.File("./yolov2.h5", "r") as f:
        for l in range(20):
            h = convolution(f["params/conv%d/weights" % (l + 1)][()])(h)
            h = batch_normalization(f["params/bn%d/scale" % (l + 1)][()], f["params/conv%d/bias" % (l + 1)][()],
                                    f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h)
            h = C.leaky_relu(h, alpha=0.1)

            if l in [0, 1, 4, 7, 12]:
                if l == 12:
                    h12 = h
                h = C.layers.MaxPooling((2, 2), strides=2, pad=True)(h)

        l += 1
        h12 = convolution(f["params/conv%d/weights" % (l + 1)][()])(h12)
        h12 = batch_normalization(f["params/bn%d/scale" % (l + 1)][()], f["params/conv%d/bias" % (l + 1)][()],
                                  f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h12)
        h12 = C.leaky_relu(h12, alpha=0.1)

        reorg = C.reshape(h12, (256, num_cell, num_cell))

        h = C.splice(reorg, h, axis=0)

        l += 1
        h = convolution(f["params/conv%d/weights" % (l + 1)][()])(h)
        h = batch_normalization(f["params/bn%d/scale" % (l + 1)][()], f["params/conv%d/bias" % (l + 1)][()],
                                f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h)
        h = C.leaky_relu(h, alpha=0.1)

        l += 1
        h = convolution(f["params/conv%d/weights" % (l + 1)][()])(h)
        h += C.Constant(f["params/conv%d/bias" % (l + 1)][()].reshape(-1, 1, 1))

    return h


if __name__ == "__main__":
    if not os.path.exists("./yolov2.h5"):
        extract_parameter()

    #
    # yolo v2
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32")
    net = yolov2(input)

    #
    # web camera
    #
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_height)

    while True:
        _, frame = cap.read()

        start = time.perf_counter()

        x_img = np.ascontiguousarray(cv2.resize(frame, (img_width, img_height))[..., ::-1].transpose(2, 0, 1) / 255,
                                     dtype="float32")
        y = net.eval({net.arguments[0]: x_img})[0]

        bboxes_dict = {}
        for i in range(num_bboxes):
            obj_x, obj_y = np.where(y[4 + 85 * i, :, :] > conf_threshold)
            if len(obj_x) == 0:
                continue
            else:
                for ix, iy in zip(obj_x, obj_y):
                    cx, cy = (C.sigmoid(y[85 * i:2 + 85 * i, ix, iy]).eval() + np.array([ix, iy])) * 32
                    ph, pw = C.exp(y[2 + 85 * i:4 + 85 * i, ix, iy]).eval() * anchors_boxes[i] * 32
                    prob = C.softmax(y[5 + 85 * i:85 * (i + 1), ix, iy]).eval()

                    if is_flip:
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

        if is_flip:
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
    
