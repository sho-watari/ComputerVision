import cntk as C
import pandas as pd

from cntk.layers import BatchNormalization, Convolution2D, GlobalAveragePooling, MaxPooling
from cntkx.learners import CyclicalLearningRate

from mnist_loader import load_mnist

img_channel = 1
img_height = 28
img_width = 28
num_classes = 10

epoch_size = 25
minibatch_size = 256
num_samples = 60000

step_size = num_samples // minibatch_size * 10
weight_decay = 0.0005


def convnet(h):
    with C.layers.default_options(activation=C.relu, init=C.he_normal(), pad=True, strides=1, bias=False,
                                  map_rank=1, use_cntk_engine=True):
        h = Convolution2D((3, 3), 128)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 128)(h)
        h = BatchNormalization()(h)

        h = MaxPooling((3, 3), strides=2)(h)

        h = Convolution2D((3, 3), 256)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((1, 1), 128)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 256)(h)
        h = BatchNormalization()(h)

        h = MaxPooling((3, 3), strides=2)(h)

        h = Convolution2D((3, 3), 512)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((1, 1), 256)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 512)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((1, 1), 256)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 512)(h)
        h = BatchNormalization()(h)

        h = Convolution2D((1, 1), num_classes, activation=None, bias=True)(h)
        h = GlobalAveragePooling()(h)
        h = C.reshape(h, -1)

        return h


if __name__ == "__main__":
    #
    # load Fashion-MNIST dataset
    #
    train_image, train_label, test_image, test_label = load_mnist()

    #
    # input, label, and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    label = C.input_variable(shape=num_classes, dtype="float32")

    model = convnet((input - train_image.mean(axis=0)) / 255.0)

    #
    # loss function and error metrics
    #
    loss = C.cross_entropy_with_softmax(model, label)
    errs = C.classification_error(model, label)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.momentum_sgd(model.parameters, lr=0.1, momentum=0.9, l2_regularization_weight=l2_decay)
    clr = CyclicalLearningRate(learner, base_lrs=1e-4, max_lrs=0.6, minibatch_size=minibatch_size, step_size=step_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")
    trainer = C.Trainer(model, (loss, errs), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    #
    # training
    #
    logging = {"epoch": [], "loss": [], "error": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            data = {input: train_image[sample_count: sample_count + minibatch_size],
                    label: train_label[sample_count: sample_count + minibatch_size]}

            trainer.train_minibatch(data)

            clr.batch_step()

            sample_count += minibatch_size
            epoch_loss += trainer.previous_minibatch_loss_average
            epoch_metric += trainer.previous_minibatch_evaluation_average

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / (num_samples / minibatch_size))
        logging["error"].append(epoch_metric / (num_samples / minibatch_size))

        trainer.summarize_training_progress()

    model.save("./mnist.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./mnist.csv", index=False)
    print("Saved logging.")
    
