import cntk as C
import cntk.io.transforms as xforms
import cntkx as Cx
import numpy as np
import os
import pandas as pd

from cntk.layers import BatchNormalization, Convolution2D, Dense, GlobalAveragePooling
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


def cbm(h, num_channel, kernel_size=3, stride=1, pad=True, activation=True):
    with C.layers.default_options(init=C.he_normal(), bias=False, map_rank=1, use_cntk_engine=True):
        h = Convolution2D((kernel_size, kernel_size), num_channel, strides=stride, pad=pad)(h)
        h = BatchNormalization()(h)
        if activation:
            h = Cx.mish(h)

        return h


def residual_block(x, num_filters):
    h = cbm(x, num_filters // 2, kernel_size=1)
    h = cbm(h, num_filters, kernel_size=3, activation=False)

    return Cx.mish(h + x)


def upconv(h, num_channel, factor=2):
    for _ in range(int(np.log2(factor))):
        h = Cx.upsample(h, 2)

    return cbm(h, num_channel, kernel_size=1)


def downconv(h, num_channel, factor=2):
    for _ in range(int(np.log2(factor))):
        h = cbm(h, num_channel, stride=2)

    return h


def hrnet(h):
    h = cbm(h, 64, stride=2)
    h = cbm(h, 64, stride=2)

    #
    # stage 1
    #
    h11 = residual_block(h, 64)  # 56x56

    h21 = cbm(h11, 64)
    h22 = downconv(h11, 128, 2)

    #
    # stage 2
    #
    h21 = residual_block(h21, 64)   # 56x56
    h22 = residual_block(h22, 128)  # 28x28

    h31 = cbm(C.splice(h21, upconv(h22, 128, 2), axis=0), 64)
    h32 = cbm(C.splice(downconv(h21, 64, 2), h22, axis=0), 128)
    h33 = cbm(C.splice(downconv(h21, 64, 4), downconv(h22, 128, 2), axis=0), 256)

    #
    # stage 3
    #
    h31 = residual_block(h31, 64)   # 56x56
    h32 = residual_block(h32, 128)  # 28x28
    h33 = residual_block(h33, 256)  # 14x14

    h41 = cbm(C.splice(h31, upconv(h32, 128, 2), upconv(h33, 256, 4), axis=0), 64)
    h42 = cbm(C.splice(downconv(h31, 64, 2), h32, upconv(h33, 256, 2), axis=0), 128)
    h43 = cbm(C.splice(downconv(h31, 64, 4), downconv(h32, 128, 2), h33, axis=0), 256)
    h44 = cbm(C.splice(downconv(h31, 64, 8), downconv(h32, 128, 4), downconv(h33, 256, 2), axis=0), 512)

    #
    # stage 4
    #
    h41 = residual_block(h41, 64)   # 56x56
    h42 = residual_block(h42, 128)  # 28x28
    h43 = residual_block(h43, 256)  # 14x14
    h44 = residual_block(h44, 512)  # 7x7

    hr = C.splice(cbm(h41, 128), upconv(h42, 128, 2), upconv(h43, 256, 4), upconv(h44, 512, 8), axis=0, name="higher")

    h41 = cbm(hr, 64)
    h42 = cbm(C.splice(downconv(h41, 64, 2), h42, upconv(h33, 256, 2), upconv(h44, 512, 4), axis=0), 128)
    h43 = cbm(C.splice(downconv(h41, 64, 4), downconv(h42, 128, 2), h43, upconv(h44, 512, 2), axis=0), 256)
    h44 = cbm(C.splice(downconv(h41, 64, 8), downconv(h42, 128, 4), downconv(h43, 256, 2),  h44, axis=0), 512)

    #
    # classification
    #
    h = cbm(downconv(downconv(downconv(h41, 128) + h42, 256) + h43, 512) + h44, 1024, kernel_size=1)
    h = GlobalAveragePooling()(h)
    h = Dense(num_classes, activation=None, init=C.glorot_uniform(), bias=True)(h)

    return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("../RTSS/train_imagenet_map.txt", is_train=True)

    #
    # input, label and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    label = C.input_variable(shape=(num_classes,), dtype="float32")

    input_map = {input: train_reader.streams.images, label: train_reader.streams.labels}

    model = hrnet(input / 255.0)

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
    model.save("./hrnet.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./hrnet.csv", index=False)
    print("Saved logging.")
    
