import cv2
import glob
import numpy as np
import os
import pickle
import random
import tarfile

from scipy import io

joint_map = {0: "r_ankle", 1: "r_knee", 2: "r_hip", 3: "l_hip", 4: "l_knee", 5: "l_ankle", 6: "pelvis", 7: "thorax",
             8: "upper_neck", 9: "head_top", 10: "r_wrist", 11: "r_elbow", 12: "r_shoulder", 13: "l_shoulder",
             14: "l_elbow", 15: "l_wrist"}

map_joint = {val: key for key, val in joint_map.items()}

data_file = "./mpii_human_pose_v1"

img_height = 320
img_width = 480
num_keypoint = 16

sigma = 2


def mpii_human_pose_targz(data_file):
    with tarfile.open("%s.tar.gz" % data_file, "r:gz") as f:
        f.extractall(path=data_file)


if __name__ == "__main__":
    #
    # filename and keypoint dictionary
    #
    if not os.path.exists("./mpii_human_pose_v1/mpii_human_pose_keypoint.pkl"):
        mat = io.loadmat("%s/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat" % data_file)

        ann_list, train_list = mat["RELEASE"]["annolist"][0, 0][0], mat["RELEASE"]["img_train"][0, 0][0]
        kps_dict = {}
        for i, (ann, is_train) in enumerate(zip(ann_list, train_list)):
            filename = ann["image"]["name"][0, 0][0]

            if not is_train:
                continue

            if not "annopoints" in str(ann["annorect"].dtype):
                continue

            annopoints = ann["annorect"]["annopoints"][0]
            for annopoint in annopoints:
                if not "point" in str(annopoint.dtype):
                    continue

                annopoint = annopoint["point"][0, 0]
                joint_idx = [str(j[0, 0]) for j in annopoint["id"][0]]
                x = [int(x[0, 0]) for x in annopoint["x"][0]]
                y = [int(y[0, 0]) for y in annopoint["y"][0]]

                joint_pos = {}
                for _j_idx, (_x, _y) in zip(joint_idx, zip(x, y)):
                    joint_pos[joint_map[int(_j_idx)]] = [_x, _y]

                kps_dict.setdefault(filename, []).append(joint_pos)

        with open("./mpii_human_pose_v1/mpii_human_pose_keypoint.pkl", "wb") as f:
            pickle.dump(kps_dict, f)

    else:
        with open("./mpii_human_pose_v1/mpii_human_pose_keypoint.pkl", "rb") as f:
            kps_dict = pickle.load(f)

    #
    # image and heatmap
    #
    if not os.path.exists(data_file + "/image"):
        os.makedirs(data_file + "/image")

    if not os.path.exists(data_file + "/heatmap"):
        os.makedirs(data_file + "/heatmap")

    psn_list = []
    for i, (filename, kps_list) in enumerate(kps_dict.items()):
        img = cv2.imread(data_file + "/images/" + filename)
        height, width, _ = img.shape
        resize = cv2.resize(img, (img_width, img_height))

        cv2.imwrite(data_file + "/image/image_{:0>5d}.jpg".format(i), resize)

        #
        # heatmap of keypoint
        #
        heatmap = np.zeros((num_keypoint, img_height // 4, img_width // 4), dtype="float32")
        for kps in kps_list:
            for joint, xy in kps.items():
                x, y = xy[0] / width, xy[1] / height

                if int(y * (img_height // 4)) >= img_height // 4:
                    heatmap[map_joint[joint], img_height // 4 - 1, int(x * (img_width // 4))] = 1
                elif int(x * (img_width // 4)) >= img_width // 4:
                    heatmap[map_joint[joint], int(y * (img_height // 4)), img_width // 4 - 1] = 1
                else:
                    heatmap[map_joint[joint], int(y * (img_height // 4)), int(x * (img_width // 4))] = 1

        #
        # blur and normalization
        #
        for k in range(num_keypoint):
            if heatmap[k].max() == 0:
                continue
            else:
                blur = cv2.GaussianBlur(heatmap[k], ksize=(3, 3), sigmaX=sigma, sigmaY=sigma)
                heatmap[k] = blur / blur.max()

        np.save(data_file + "/heatmap/heatmap_{:0>5d}".format(i), heatmap)

        #
        # number of person
        #
        psn_list.append(len(kps_list))

    #
    # image, heatmap, and person
    #
    img_list = glob.glob(data_file + "/image/*.jpg")
    kps_list = glob.glob(data_file + "/heatmap/*.npy")

    val_list = random.sample(range(len(img_list)), int(len(img_list) * 0.15))

    train_file = open("./train_hrpe320x480_map.txt", "w")
    valid_file = open("./val_hrpe320x480_map.txt", "w")
    for i, (img_file, kps_file, psn) in enumerate(zip(img_list, kps_list, psn_list)):
        if i in val_list:
            valid_file.write("%s\t%s\t%s\n" % (img_file, kps_file, psn))
        else:
            train_file.write("%s\t%s\t%s\n" % (img_file, kps_file, psn))
    train_file.close()
    valid_file.close()

    print("Number of training sample", len(img_list) - len(val_list))
    print("Number of validation sample", len(val_list))
    
