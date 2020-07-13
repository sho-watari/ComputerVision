import cntk as C
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re

img_channel = 3
img_height = 300
img_width = 300

num_feature = 1024
num_word = 12212
num_hidden = 512

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["font.size"] = 15

BOS = "<BOS>"
EOS = "<EOS>"
MAX = 35

with open("id2word.pkl", "rb") as f:
    id2word = pickle.load(f)


def convolution(weights, name=''):
    W = C.Constant(value=weights, name='W')

    @C.BlockFunction('Convolution2D', name)
    def conv2d(x):
        return C.convolution(W, x, strides=[1, 1], auto_padding=[False, True, True])

    return conv2d


def batch_normalization(scale, bias, mean, variance, spatial=True, name=''):
    scale = C.Constant(value=scale, name='scale')
    bias = C.Constant(value=bias, name='scale')
    mu = C.Constant(value=mean, name='aggreagate_mean')
    sigma = C.Constant(value=variance, name='aggregate_variance')

    @C.BlockFunction('BatchNormalization', name)
    def batch_norm(x):
        return C.batch_normalization(x, scale, bias, mu, sigma, spatial=spatial, running_count=C.constant(5000))

    return batch_norm


def coco21(h, filename="../COCO/coco21.h5"):
    with h5py.File(filename, "r") as f:
        for l in range(20):
            h = convolution(f["params/conv%d/weights" % (l + 1)][()])(h)
            h = C.elu(h)
            h = batch_normalization(f["params/bn%d/scale" % (l + 1)][()], f["params/bn%d/bias" % (l + 1)][()],
                                    f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h)
            if l in [1, 3, 6, 9, 14]:
                h = C.pooling(h, C.MAX_POOLING, pooling_window_shape=(3, 3), strides=(2, 2), auto_padding=[False, True, True])
        h = C.layers.GlobalAveragePooling()(h)
    return h


if __name__ == "__main__":
    #
    # input, coco21, and model
    #
    x = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32")
    cnn = coco21(x / 255.0)
    model = C.load_model("./nics.model")

    #
    # Image Caption Generator
    #
    filename = input("filename: ")
    
    image = cv2.resize(cv2.imread(filename), (img_width, img_height)).transpose(2, 0, 1)
    feature = np.reshape(cnn.eval({x: np.ascontiguousarray(image, dtype="float32")}), (1, 1, num_feature))
    word = np.identity(num_word, dtype="float32")[1].reshape(1, 1, num_word)

    for _ in range(MAX):
        prob = model.eval({model.arguments[0]: feature, model.arguments[1]: word})[0]
        pred = np.identity(num_word, dtype="float32")[prob.argmax(axis=1)[-1]].reshape(1, 1, num_word)
        word = np.concatenate((word, pred), axis=1)
        if id2word[prob.argmax(axis=1)[-1]] == EOS:
            break

    caption = ""
    for idx in word[0].argmax(axis=1):
        caption += id2word[idx]

    plt.figure()
    plt.imshow(image[::-1].transpose(1, 2, 0))
    plt.axis("off")
    plt.title(re.sub("<BOS>|<EOS>", "", caption))
    plt.show()
    
