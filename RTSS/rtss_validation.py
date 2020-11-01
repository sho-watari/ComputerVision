import cntk as C
import cv2
import numpy as np
import random

img_channel = 3
img_height = 320
img_width = 480
num_classes = 151

num_samples = 2000


class RealTimeSemanticSegmentation:
    def __init__(self, map_file, is_train):
        self.sample_count = 0
        self.minibatch_count = 0
        with open(map_file) as f:
            self.map_list = f.readlines()
        if is_train:
            random.shuffle(self.map_list)

    def next_minibatch(self, minibatch_size):
        self.minibatch_count = minibatch_size

        image_data = np.zeros((minibatch_size, img_channel, img_height, img_width), dtype="float32")
        label_data = np.zeros((minibatch_size, num_classes, img_height, img_width), dtype="float32")

        batch_file = self.map_list[self.sample_count: self.sample_count + minibatch_size]
        for i, line in enumerate(batch_file):
            img_file, ann_file = line[:-1].split("\t")

            image_data[i] = np.ascontiguousarray(cv2.imread(img_file).transpose(2, 0, 1), dtype="float32")
            label_data[i] = np.load(ann_file)

        self.sample_count += minibatch_size

        return image_data, label_data

    def randomize(self):
        random.shuffle(self.map_list)


def rtss_meanIOU():
    model = C.load_model("./rtss.model")

    #
    # mean intersection over union
    #
    valid_reader = RealTimeSemanticSegmentation("./val_rtss320x480_map.txt", is_train=False)

    iou = np.zeros((num_samples, num_classes))
    count = np.zeros((num_samples, num_classes))
    for n in range(num_samples):
        batch_image, batch_label = valid_reader.next_minibatch(1)

        predict = model.eval({model.arguments[0]: batch_image})[0].argmax(axis=0)

        predict_label = np.empty((num_classes, img_height, img_width), dtype=np.int8)
        for i in range(img_height):
            for j in range(img_width):
                predict_label[predict[i, j], i, j] = 1

        intersection = predict_label * batch_label[0]
        union = predict_label + batch_label[0] - intersection

        iou[n] = intersection.sum(axis=(1, 2)) / (union.sum(axis=(1, 2)) + 1e-5)
        count[n] = np.identity(num_classes)[list(np.nonzero(batch_label[0].sum(axis=(1, 2)))[0])].sum(axis=0)

    mean_iou = iou.sum(axis=0) / (count.sum(axis=0) + 1e-5)
    print("meanIOU %.1f" % (mean_iou.sum() / np.count_nonzero(mean_iou) * 100))


if __name__ == "__main__":
    rtss_meanIOU()
    
