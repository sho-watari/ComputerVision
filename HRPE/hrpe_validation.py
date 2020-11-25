import cntk as C
import cv2
import numpy as np
import random

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

num_samples = 2611


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


class HigherResolutionPoseEstimation:
    def __init__(self, map_file, is_train):
        self.sample_count = 0
        self.minibatch_count = 0
        with open(map_file) as f:
            self.map_list = f.readlines()
        if is_train:
            random.shuffle(self.map_list)

    def next_minibatch(self, minibatch_size):
        self.minibatch_count = minibatch_size

        batch_img = np.zeros((minibatch_size, img_channel, img_height, img_width), dtype="float32")
        batch_kps = np.zeros((minibatch_size, num_keypoint, img_height // 4, img_width // 4), dtype="float32")
        batch_psn = np.zeros((minibatch_size, 1), dtype="float32")

        batch_file = self.map_list[self.sample_count: self.sample_count + minibatch_size]
        for i, line in enumerate(batch_file):
            img_file, kps_file, psn = line[:-1].split("\t")

            batch_img[i] = np.ascontiguousarray(cv2.imread(img_file).transpose(2, 0, 1), dtype="float32")
            batch_kps[i] = np.load(kps_file)
            batch_psn[i] = psn

        self.sample_count += minibatch_size

        return batch_img, batch_kps, batch_psn

    def randomize(self):
        random.shuffle(self.map_list)


def hrpe_PCKh(threshold):
    #
    # minibatch reader
    #
    valid_reader = HigherResolutionPoseEstimation("./val_hrpe320x480_map.txt", is_train=False)

    model = C.load_model("./hrpe.model")

    correct_keypoint, num_person = 0, 0

    for _ in range(num_samples):
        batch_img, batch_kps, batch_psn = valid_reader.next_minibatch(1)

        output = model.eval({model.arguments[0]: batch_img})
        kps, psn = output[model.outputs[0]][0], output[model.outputs[1]][0][0]
        img = np.ascontiguousarray(batch_img[0].transpose(1, 2, 0), dtype="uint8")

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
                    cv2.line(img, tuple(jnt[joint[0]]), tuple(jnt[joint[1]]), color[j], thickness=thickness)

            #
            # percentage of correct key-points head
            #
            head_list = []
            for kps in kps_list:
                head_list.append(kps[9])

            nms_map = non_maximum_suppression(batch_kps[0].transpose(1, 2, 0), img_height // 4, img_width // 4)
            head_map, neck_map = nms_map[..., 9], nms_map[..., 8]

            _, head_h, head_w = np.where([head_map > 0])
            _, neck_h, neck_w = np.where([neck_map > 0])

            try:
                head_bone_link = np.sqrt(np.square(head_h - neck_h) + np.square(head_w - neck_w)) * threshold
            except ValueError:
                continue

            tmp = np.zeros((img_height // 4, img_width // 4), dtype="uint8")
            for i in range(int(batch_psn)):
                try:
                    cv2.circle(tmp, (int(head_w[i]), int(head_h[i])), int(head_bone_link[i]), (255, 0, 0), -1)
                except IndexError:
                    continue

            h_true = cv2.resize(tmp, (img_width, img_height))
            _, kps_h, kps_w = np.where([h_true > 0])

            for head in head_list:
                if head == [-1, -1]:
                    continue

                if head[0] in kps_h and head[1] in kps_w:
                    correct_keypoint += 1

        num_person += int(batch_psn)

    pckh = correct_keypoint / num_person

    print("PCKh@%.1f %.1f" % (threshold, pckh * 100))


if __name__ == "__main__":
    hrpe_PCKh(0.5)
    
