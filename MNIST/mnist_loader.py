import gzip
import numpy as np

files = {"train": ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"],
         "test": ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]}

num_classes = 10


def load_mnist(normalize=False, dataset={}):
    """ Fashion-MNIST has 10 categories.

    https://github.com/zalandoresearch/fashion-mnist

    0 - T-shirt/Top
    1 - Trouser
    2 - Pullover
    3 - Dress
    4 - Coat
    5 - Sandal
    6 - Shirt
    7 - Sneaker
    8 - Bag
    9 - Ankle Boot

    """
    for file in files.keys():
        with gzip.open("./dataset/" + files[file][0], "rb") as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
            dataset[file + "_image"] = images.reshape(-1, 1, 28, 28).astype(np.float32)
        with gzip.open("./dataset/" + files[file][1], "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
            dataset[file + "_label"] = np.identity(num_classes)[labels].astype(np.float32)

    if normalize is True:
        for key in ["train_image", "test_image"]:
            dataset[key] /= 255.0

    return dataset["train_image"], dataset["train_label"], dataset["test_image"], dataset["test_label"]
    
