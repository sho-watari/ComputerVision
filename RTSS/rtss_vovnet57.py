import cntk as C
import cntk.io.transforms as xforms
import cntkx as Cx
import os
import pandas as pd

from cntk.layers import BatchNormalization, Convolution2D, Dense, GlobalAveragePooling, MaxPooling
from cntkx.learners import CyclicalLearningRate

img_channel = 3
img_height = 224
img_width = 224
num_classes = 1000

epoch_size = 100
minibatch_size = 64
num_samples = 775983

step_size = num_samples // minibatch_size * 10
weight_decay = 0.0005


def create_reader(map_file, is_train):
    transforms = []
    if is_train:
        transforms += [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)]
        transforms += [xforms.crop(crop_type="randomside", side_ratio=0.875)]
    transforms += [xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear")]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        images=C.io.StreamDef(field="image", transforms=transforms),
        labels=C.io.StreamDef(field="label", shape=num_classes))), randomize=is_train)


def osa_module(h0, num_filters, num_outputs):
    with C.layers.default_options(init=C.he_normal(), pad=True, strides=1, bias=False,
                                  map_rank=1, use_cntk_engine=True):
        h1 = Cx.mish(BatchNormalization()(Convolution2D((3, 3), num_filters)(h0)))
        h2 = Cx.mish(BatchNormalization()(Convolution2D((3, 3), num_filters)(h1)))
        h3 = Cx.mish(BatchNormalization()(Convolution2D((3, 3), num_filters)(h2)))
        h4 = Cx.mish(BatchNormalization()(Convolution2D((3, 3), num_filters)(h3)))
        h5 = Cx.mish(BatchNormalization()(Convolution2D((3, 3), num_filters)(h4)))

        h = C.splice(h1, h2, h3, h4, h5, axis=0)

        return Cx.mish(BatchNormalization()(Convolution2D((1, 1), num_outputs)(h)))


def vovnet57(h):
    with C.layers.default_options(init=C.he_normal(), pad=True, strides=1, bias=False,
                                  map_rank=1, use_cntk_engine=True):
        h = Cx.mish(BatchNormalization()(Convolution2D((3, 3), 64, strides=2)(h)))
        h = Cx.mish(BatchNormalization()(Convolution2D((3, 3), 64)(h)))
        h = Cx.mish(BatchNormalization()(Convolution2D((3, 3), 128)(h)))

        h = MaxPooling((3, 3), strides=2)(h)

        h = osa_module(h, 128, 256)

        h = MaxPooling((3, 3), strides=2)(h)

        h = osa_module(h, 160, 512)

        h = MaxPooling((3, 3), strides=2)(h)

        h = osa_module(h, 192, 768)
        h = osa_module(h, 192, 768)
        h = osa_module(h, 192, 768)
        h = osa_module(h, 192, 768)

        h = MaxPooling((3, 3), strides=2)(h)

        h = osa_module(h, 224, 1024)
        h = osa_module(h, 224, 1024)
        h = osa_module(h, 224, 1024)

        h = GlobalAveragePooling()(h)

        h = Dense(num_classes, activation=None, init=C.glorot_uniform(), bias=True)(h)

        return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train_imagenet_map.txt", is_train=True)

    #
    # input, label, and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    label = C.input_variable(shape=num_classes, dtype="float32")

    input_map = {input: train_reader.streams.images, label: train_reader.streams.labels}

    model = vovnet57(input / 255.0)

    #
    # loss function and error metrics
    #
    loss = C.cross_entropy_with_softmax(model, label)
    errs = C.classification_error(model, label)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.momentum_sgd(model.parameters, lr=0.1, momentum=0.9, l2_regularization_weight=weight_decay)
    clr = CyclicalLearningRate(learner, base_lr=1e-4, max_lr=0.1, ramp_up_step_size=step_size,
                               minibatch_size=minibatch_size)
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
    model.save("./vovnet57.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./vovnet57.csv", index=False)
    print("Saved logging.")
    
