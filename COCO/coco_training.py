import cntk as C
import cntk.io.transforms as xforms
import os

from cntk.layers import BatchNormalization, Convolution2D, GlobalAveragePooling, MaxPooling
from cntkx.learners import CyclicalLearningRate
from pandas import DataFrame

img_channel = 3
img_height = 224
img_width = 224
num_classes = 80

epoch_size = 100
minibatch_size = 32
num_samples = 548448

step_size = num_samples // minibatch_size * 10
weight_decay = 0.0005


def create_reader(map_file, mean_file, is_train):
    transforms = []
    if is_random:
        transforms += [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)]
        transforms += [xforms.crop(crop_type="randomside", side_ratio=0.875)]
    transforms += [xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear"),
                   xforms.mean(mean_file)]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        images=C.io.StreamDef(field="image", transforms=transforms),
        labels=C.io.StreamDef(field="label", shape=num_classes))), randomize=is_train)


def coco21(h):
    with C.layers.default_options(activation=C.elu, init=C.he_normal(), pad=True, strides=1, bias=False,
                                  map_rank=1, use_cntk_engine=True):
        h = Convolution2D((3, 3), 32)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 32)(h)
        h = BatchNormalization()(h)

        h = MaxPooling((3, 3), strides=2)(h)

        h = Convolution2D((3, 3), 64)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 64)(h)
        h = BatchNormalization()(h)

        h = MaxPooling((3, 3), strides=2)(h)

        h = Convolution2D((3, 3), 128)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((1, 1), 64)(h)
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

        h = MaxPooling((3, 3), strides=2)(h)

        h = Convolution2D((3, 3), 1024)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((1, 1), 512)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 1024)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((1, 1), 512)(h)
        h = BatchNormalization()(h)
        h = Convolution2D((3, 3), 1024)(h)
        h = BatchNormalization()(h)

        h = Convolution2D((1, 1), num_classes, activation=None, bias=True)(h)
        h = GlobalAveragePooling()(h)
        h = C.reshape(h, -1)

        return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train2014_coco256x256_map.txt", "./train2014_coco256x256_mean.xml", is_train=True)

    #
    # input, label, and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    label = C.input_variable(shape=num_classes, dtype="float32")

    input_map = {input: train_reader.streams.images, label: train_reader.streams.labels}

    model = coco21(input / 255.0)

    #
    # loss function and error metrics
    #
    loss = C.cross_entropy_with_softmax(model, label)
    errs = C.classification_error(model, label)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.momentum_sgd(model.parameters, lr=0.1, momentum=0.9, l2_regularization_weight=weight_decay)
    clr = CyclicalLearningRate(learner, base_lrs=1e-4, max_lrs=0.1, minibatch_size=minibatch_size, step_size=step_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, errs), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    #
    # train model
    #
    logging = {"epoch": [], "loss": [], "error": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            data = train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

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

    #
    # save model and logging
    #
    model.save("./coco21.model")
    print("Saved model.")

    df = DataFrame(logging)
    df.to_csv("./coco21.csv", index=False)
    print("Saved logging.")
    
