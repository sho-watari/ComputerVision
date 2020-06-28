import cntk as C
import os
import pandas as pd
import pickle

from cntk.layers import Dense, Embedding, LayerNormalization, LSTM, Recurrence, RecurrenceFrom
from cntkx.learner import CyclicalLearingRate

num_feature = 1024
num_word = 12212
num_hidden = 512

epoch_size = 100
minibatch_size = 1024
num_samples = 413915

sample_size = 64
step_size = num_samples // sample_size * 10


def create_reader(path, train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        words=C.io.StreamDef(field="word", shape=num_word,  is_sparse=True),
        targets=C.io.StreamDef(field="target", shape=num_word, is_sparse=True),
        features=C.io.StreamDef(field="feature", shape=num_feature, is_sparse=False))),
                                randomize=train, max_sweeps=C.io.INFINITELY_REPEAT if train else 1)


def neural_image_caption_system(features, words):
    with C.layers.default_options(enable_self_stabilization=True):
        h0 = Dense(num_hidden, bias=False)(features)
        h0 = LayerNormalization()(h0)

        h = Embedding(num_hidden)(words)
        h = LayerNormalization()(h)

        h1, c1 = Recurrence(LSTM(num_hidden), return_full_state=True)(h0).outputs
        h2, c2 = Recurrence(LSTM(num_hidden), return_full_state=True)(LayerNormalization()(h1)).outputs
        h3, c3 = Recurrence(LSTM(num_hidden), return_full_state=True)(LayerNormalization()(h2)).outputs

        h = RecurrenceFrom(LSTM(num_hidden))(h1, c1, h)
        h = LayerNormalization()(h)
        h = RecurrenceFrom(LSTM(num_hidden))(h2, c2, h)
        h = LayerNormalization()(h)
        h = RecurrenceFrom(LSTM(num_hidden))(h3, c3, h)
        h = LayerNormalization()(h)

        h = Dense(num_word)(h)

        return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train2014_nics_captions.txt", True)

    #
    # features, words, and targets
    #
    features = C.sequence.input_variable(shape=(num_feature,), is_sparse=False, sequence_axis=C.Axis("I"))
    words = C.sequence.input_variable(shape=(num_word,))
    targets = C.sequence.input_variable(shape=(num_word,))

    input_map = {features: train_reader.streams.features, words: train_reader.streams.words,
                 targets: train_reader.streams.targets}

    #
    # model, loss function, and error metrics
    #
    model = neural_image_caption_system(features, words)

    loss = C.cross_entropy_with_softmax(model, targets)
    errs = C.classification_error(model, targets)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=0.01, momentum=0.9, gradient_clipping_threshold_per_sample=sample_size,
                     gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lrs=1e-4, max_lrs=0.01, minibatch_size=sample_size, step_size=step_size)
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

            minibatch_count = data[features].num_samples
            sample_count += minibatch_count
            epoch_loss += trainer.previous_minibatch_loss_average * minibatch_count
            epoch_metric += trainer.previous_minibatch_evaluation_average * minibatch_count

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / num_samples)
        logging["error"].append(epoch_loss / num_samples)

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    model.save("./nics.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./nics.csv", index=False)
    print("Saved logging.")
    
