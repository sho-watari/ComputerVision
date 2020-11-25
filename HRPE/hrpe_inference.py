import cntk as C
import cv2
import numpy as np
import time

from scipy import ndimage
from sklearn import cluster

img_channel = 3
img_height = 320
img_width = 480
num_keypoint = 16

all_joint = [[0, 1], [1, 2], [5, 4], [4, 3], [2, 6],
             [3, 6], [6, 7], [7, 8], [8, 9], [10, 11],
             [11, 12], [15, 14], [14, 13], [12, 8], [13, 8]]

color = [(255, 128, 0), (255, 128, 0), (128, 255, 0), (128, 255, 0), (255, 128, 255),
         (255, 128, 255), (0, 255, 255), (0, 255, 255), (0, 0, 255), (255, 0, 0),
         (255, 0, 0), (0, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0)]

conf_threshold = 0.1
nms_threshold = 0.05
psn_threshold = 0.75

sigma = 2
thickness = 3


def non_maximum_suppression(heatmap, height, width):
    map_left = np.zeros((height, width, num_keypoint))
    map_right = np.zeros((height, width, num_keypoint))
    map_top = np.zeros((height, width, num_keypoint))
    map_bottom = np.zeros((height, width, num_keypoint))

    map_left[1:, :] = heatmap[:-1, :]
    map_right[:-1, :] = heatmap[1:, :]
    map_top[:, 1:] = heatmap[:, :-1]
    map_bottom[:, :-1] = heatmap[:, 1:]

    peaks_binary = np.logical_and.reduce((heatmap >= map_left, heatmap >= map_right, heatmap >= map_top,
                                          heatmap >= map_bottom, heatmap >= nms_threshold))

    return peaks_binary * heatmap


if __name__ == "__main__":
    model = C.load_model("./hrpe.model")

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
        frame = cv2.resize(frame, (img_width, img_height))

        start = time.perf_counter()

        output = model.eval({model.arguments[0]: np.ascontiguousarray(frame.transpose(2, 0, 1), dtype="float32")})
        kps, psn = output[model.outputs[0]][0], output[model.outputs[1]][0][0]

        #
        # postprocessing
        #
        if psn > psn_threshold:
            #
            # resize and blur
            #
            heatmap = cv2.resize(kps.transpose(1, 2, 0), (img_width, img_height), interpolation=cv2.INTER_CUBIC)
            for i in range(num_keypoint):
                heatmap[..., i] = ndimage.gaussian_filter(heatmap[..., i], sigma=sigma)

            #
            # non maximum suppression
            #
            heatmap = non_maximum_suppression(heatmap, img_height, img_width)

            N = round(psn)
            conf_list = []
            for i in range(num_keypoint):
                value = heatmap[..., i].flatten()
                value.sort()
                values = value[::-1][:N]
                for n in range(N):
                    if values[n] > conf_threshold:
                        k, h, w = np.where([heatmap[:, :, i] == values[n]])
                    else:
                        continue

                    conf_list.append([i, int(h[0]), int(w[0])])

            if not conf_list:
                N = 0
                continue
            conf = np.array(conf_list)

            #
            # k-means clustering
            #
            kms = cluster.KMeans(n_clusters=N, init="k-means++", n_init=10, max_iter=10, tol=1e-4)
            label = kms.fit_predict(conf[:, 1:])

            kps_list = []
            for n in range(N):
                kxy = conf[label == n]
                kps_list.append({int(k): [int(x), int(y)] for k, x, y in kxy})

            for kps in kps_list:
                add_key = set(range(num_keypoint)) - kps.keys()
                for key in add_key:
                    kps[key] = [-1, -1]
                jnt = [[y, x] for k, (x, y) in sorted(kps.items())]
                for j, joint in enumerate(all_joint):
                    if jnt[joint[0]] == [-1, -1] or jnt[joint[1]] == [-1, -1]:
                        continue
                    cv2.line(frame, tuple(jnt[joint[0]]), tuple(jnt[joint[1]]), color[j], thickness=thickness)

        else:  # no person
            N = 0

        cv2.imshow("Higher Resolution Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        end = time.perf_counter()

        print("%d person FPS %.1f" % (N, 1.0 / (end - start)))

    cap.release()
    cv2.destroyAllWindows()
    
