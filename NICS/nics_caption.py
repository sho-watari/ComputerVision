import cntk as C
import cntk.io.transforms as xforms
import h5py
import json
import numpy as np
import pickle

from collections import Counter, OrderedDict

dir_file = "../COCO/COCO"
data_file = "train2014"
caption_file = "./stair_captions/stair_captions_v1.2/stair_captions_v1.2_train_tokenized.json"

img_channel = 3
img_height = 300
img_width = 300

BOS = "<BOS>"  # begin of sequence
EOS = "<EOS>"  # end of sequence
UNK = "<UNK>"  # unknown


def create_word2id(captions):
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Number of total words:", len(counter))
    word_counts = [x for x in counter.items() if x[1] >= 4]  # less 4 count word is not common word
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Number of words:", len(word_counts) + 1)  # plus 1 is <UNK>
    word_list = [x[0] for x in word_counts]
    word_list.remove(BOS)
    word_list.remove(EOS)
    word_list.insert(0, BOS)
    word_list.insert(0, EOS)

    word_list.append(UNK)
    word2id = dict([(x, y) for (y, x) in enumerate(word_list)])
    id2word = dict([(x, y) for (x, y) in enumerate(word_list)])
    return word2id, id2word


def create_reader(map_file):
    transforms = [xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear")]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        images=C.io.StreamDef(field="image", transforms=transforms),
        labels=C.io.StreamDef(field="label", shape=1))), randomize=False)


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
    """ Caption text and 300x300 image from COCO dataset """
    with open(caption_file, "rb") as file:
        dataset = json.load(file)

    annotations = dataset["annotations"]

    captions_dict = OrderedDict()
    for ann in annotations:
        image_id = ann["image_id"]
        filename = "{:s}/{:s}/COCO_{:s}_{:0>12s}.jpg".format(dir_file, data_file, data_file, str(image_id))

        tokenized_caption = [BOS]
        tokenized_caption.extend(ann["tokenized_caption"].split())
        tokenized_caption.append(EOS)

        captions_dict.setdefault(filename, []).append(tokenized_caption)

    with open("./" + data_file + "_nics_images.txt", "w") as map_file:
        for key in caption_dict.keys():
            map_file.write("%s\t%d\n" % (key, 0))

    num_samples = len(captions_dict)

    #
    # word2id and id2word dictionary
    #
    captions_list = []
    for valus in captions_dict.values():
        for value in values:
            captions_list.append(value)

    word2id, id2word = create_word2id(captions_list)

    with open("word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    print("Saved word2id.pkl\n")

    with open("id2word.pkl", "wb") as f:
        pickle.dump(id2word, f)
    print("Saved id2word.pkl\n")

    #
    # encoding CNN features
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32")
    model = coco21(input / 255.0)

    image_reader = create_reader("./" + data_file + "_nics_images.txt")

    minibatch_size = 64
    sample_count = 0
    features = []    
    while sample_count < num_samples:
        data = image_reader.next_minibatch(min(minibatch_size, num_samples - sample_count))
        features.append(np.squeeze(model.eval({input: list(data.values())[0]})))
        sample_count += minibatch_size

    features = np.concatenate((np.array(features[:-1], dtype="float32").reshape(-1, 1024),
                               np.array(features[-1], dtype="float32")), axis=0)

    print("\nAll image are encoded feature vector -> {}\n".format(features.shape))

    #
    # word, target, and features
    #
    num_samples = 0
    with open("./" + data_file + "_nics_captions.txt", "w", encoding="utf-8") as cap_file:
        for i, word_list in enumerate(captions_dict.values()):
            for words in word_list:
                words = [word if word in word2id else UNK for word in words]

                image = " ".join(["%e" % n for n in features[i]])
                cap_file.write("%d\t|word %d:1 |# %s\t|target %d:1 |# %s\t|feature %s\n" % (  # first row
                    num_samples, word2id[BOS], BOS, word2id[words[1]], words[1], image))
                for w, word in enumerate(words[1:-1]):
                    cap_file.write("%d\t|word %d:1 |# %s\t|target %d:1 |# %s\n" % (
                        num_samples, word2id[word], word, word2id[words[w + 2]], words[w + 2]))

                num_samples += 1
                if num_samples % 10000 == 0:
                    print("Now %d samples..." % num_samples)

    print("\nNumber of samples", num_samples)
    
