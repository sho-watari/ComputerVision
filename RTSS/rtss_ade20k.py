import cv2
import glob
import numpy as np
import os

dir_file = "./ADEChallengeData2016/"
data_file = "training"

images_file = "images/%s" % data_file
annotations_file = "annotations/%s" % data_file

img_height = 320
img_width = 480
num_classes = 150 + 1  # 150 categories + background


if __name__ == "__main__":
    img_list = glob.glob(dir_file + images_file + "/*.jpg")
    ann_list = glob.glob(dir_file + annotations_file + "/*.png")

    if not os.path.exists(dir_file + "%s/images" % data_file):
        os.makedirs(dir_file + "%s/images" % data_file)

    if not os.path.exists(dir_file + "%s/labels" % data_file):
        os.makedirs(dir_file + "%s/labels" % data_file)

    for i, (img_file, ann_file) in enumerate(zip(img_list, ann_list)):
        print("%d\t%s\t%s" % (i, img_file, ann_file))
        #
        # resize image
        #
        img = cv2.resize(cv2.imread(img_file), (img_width, img_height))

        cv2.imwrite(dir_file + "%s/images/" % data_file + img_file.rsplit("\\")[1], img)

        #
        # annotation label
        #
        ann = cv2.resize(cv2.imread(ann_file), (img_width, img_height))
        idx = ann.mean(axis=2).astype(np.int8)

        label = np.zeros((num_classes, img_height, img_width), dtype=np.int8)
        for i in range(img_height):
            for j in range(img_width):
                label[idx[i, j], i, j] = 1

        np.save(dir_file + "%s/labels/" % data_file + ann_file.rsplit(".", 1)[0].rsplit("\\")[1], label)

    #
    # image and label file
    #
    img_list = glob.glob(dir_file + "%s/images/*.jpg" % data_file)
    ann_list = glob.glob(dir_file + "%s/labels/*.npy" % data_file)
    if data_file == "training":
        with open("./train_rtss320x480_map.txt", "w") as map_file:
            for i, (img_file, ann_file) in enumerate(zip(img_list, ann_list)):
                map_file.write("%s\t%s\n" % (img_file, ann_file))
    else:
        with open("./val_rtss320x480_map.txt", "w") as map_file:
            for i, (img_file, ann_file) in enumerate(zip(img_list, ann_list)):
                map_file.write("%s\t%s\n" % (img_file, ann_file))
                
