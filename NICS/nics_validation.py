import cntk as C
import cv2
import h5py
import json
import nltk
import numpy as np
import os
import pickle

from nltk.translate.bleu_score import sentence_bleu

img_channel = 3
img_height = 300
img_width = 300

num_feature = 1024
num_word = 12212
num_hidden = 512

BOS = "<BOS>"  # begin of sequence
EOS = "<EOS>"  # end of sequence
UNK = "<UNK>"  # unknown

M = 35

with open("id2word.pkl", "rb") as f:
    id2word = pickle.load(f)

with open("word2id.pkl", "rb") as f:
    word2id = pickle.load(f)


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


def coco21(h, filename="../COCO/coco21.h5"):
    with h5py.File(filename, "r") as f:
        for l in range(20):
            h = convolution(f["params/conv%d/weights" % (l + 1)][()])(h)
            h = C.elu(h)
            h = batch_normalization(f["params/bn%d/scale" % (l + 1)][()], f["params/bn%d/bias" % (l + 1)][()],
                                    f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h)
            if l in [1, 3, 6, 9, 14]:
                h = C.layers.MaxPooling((3, 3), strides=2, pad=True)(h)

        h = C.layers.GlobalAveragePooling()(h)
    return h


def nics_bleu():
    dir_file = "../COCO/COCO"
    data_file = "val2014"
    caption_file = "./stair_captions/stair_captions_v1.2/stair_captions_v1.2_val_tokenized.json"

    if not os.path.exists(data_file + "_nics_bleu.pkl"):
        with open(caption_file, "rb") as file:
            dataset = json.load(file)

        annotations = dataset["annotations"]

        val_dict = {}
        for ann in annotations:
            image_id = ann["image_id"]
            filename = "{:s}/{:s}/COCO_{:s}_{:0>12s}.jpg".format(dir_file, data_file, data_file, str(image_id))

            word_list = ann["tokenized_caption"].split()
            word_list = [word if word in word2id.keys() else UNK for word in word_list]

            tokenized_caption = [BOS]
            tokenized_caption.extend(word_list)
            tokenized_caption.append(EOS)

            val_dict.setdefault(filename, []).append(tokenized_caption)

        num_samples = len(val_dict)
        print("Number of samples", num_samples)

        with open("./" + data_file + "_nics_bleu.pkl", "wb") as f:
            pickle.dump(val_dict, f)
        print("Saved " + data_file + "_nics_bleu.pkl")
    else:
        with open("./" + data_file + "_nics_bleu.pkl", "rb") as f:
            val_dict = pickle.load(f)

    #
    # input, coco21, and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32")
    CNN = coco21(input / 255.0)
    model = C.load_model("./nics.model")

    #
    # bilingual evaluation understudy
    #
    method = nltk.translate.bleu_score.SmoothingFunction()
    bleu4 = []
    bleu1 = []
    for filename, reference in val_dict.items():
        image = cv2.resize(cv2.imread(filename), (img_width, img_height)).transpose(2, 0, 1)
        feature = np.reshape(CNN.eval({input: np.ascontiguousarray(image, dtype="float32")}), (1, 1, num_feature))
        word = np.identity(num_word, dtype="float32")[1].reshape(1, 1, num_word)

        for _ in range(M):
            prob = model.eval({model.arguments[0]: feature, model.arguments[1]: word})[0]
            pred = np.identity(num_word, dtype="float32")[prob.argmax(axis=1)[-1]].reshape(1, 1, -1)
            word = np.concatenate((word, pred), axis=1)
            if id2word[prob.argmax(axis=1)[-1]] == "<EOS>":
                break

        candidate = []
        for idx in word[0].argmax(axis=1):
            candidate.append(id2word[idx])

        bleu4.append(sentence_bleu(reference, candidate, smoothing_function=method.method3))
        bleu1.append(sentence_bleu(reference, candidate, weights=(1,), smoothing_function=method.method3))

    bleu4_score = np.array(bleu4)
    bleu1_score = np.array(bleu1)

    print("BLEU-4 Score {:.2f}".format(bleu4_score.mean() * 100))
    print("BLEU-1 Score {:.2f}".format(bleu1_score.mean() * 100))


if __name__ == "__main__":
    nics_bleu()
    
