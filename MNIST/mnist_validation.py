import cntk as C

from mnist_loader import load_mnist

num_classes = 10

num_samples = 10000


if __name__ == "__main__":
    #
    # load Fashion-MNIST dataset
    #
    train_image, train_label, test_image, test_label = load_mnist()

    #
    # load Fashion-MNIST model
    #
    model = C.load_model("./mnist.model")
    label = C.input_variable(shape=num_classes, dtype="float32")

    errs = C.classification_error(model, label)

    #
    # Test Accuracy
    #
    minibatch_size = 512
    sample_count = 0
    error_count = 0
    while sample_count < num_samples:
        data = {model.arguments[0]: test_image[sample_count: sample_count + minibatch_size],
                label: test_label[sample_count: sample_count + minibatch_size]}

        error_count += errs.eval(data).sum()

        sample_count += minibatch_size

    print("Test Accuracy {:.2f}%".format((num_samples - error_count) / num_samples * 100))
    
