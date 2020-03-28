import cntk as C
import cntk.io.transforms as xforms
import json

img_channel = 3
img_height = 224
img_width = 224
num_classes = 80

num_samples = 89299


def create_reader(map_file, mean_file, is_train):
    transforms = []
    if is_train:
        transforms += [xforms.crop(crop_type="randomside", side_ratio=0.875)]
        transforms += [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)]
    transforms += [xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear"),
                   xforms.mean(mean_file)]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        images=C.io.StreamDef(field="image", transforms=transforms),
        labels=C.io.StreamDef(field="label", shape=num_classes))), randomize=is_train)


if __name__ == "__main__":
    #
    # built-in reader
    #
    valid_reader = create_reader("./val2014_coco256x256_map.txt", "./train2014_coco256x256_mean.xml", is_train=False)

    #
    # model and label
    #
    model = C.load_model("./coco21.model")
    label = C.input_variable(shape=num_classes, dtype="float32")

    input_map = {model.arguments[0]: valid_reader.streams.images, label: valid_reader.streams.labels}

    #
    # Top-3, Top-1 Accuracy
    #
    top1 = C.classification_error(model, label)
    top3 = C.classification_error(model, label, topN=3)

    minibatch_size = 64
    sample_count = 0
    top1_error = 0
    top3_error = 0
    while sample_count < num_samples:
        data = valid_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

        top1_error += top1.eval(data).sum()  # Top-1
        top3_error += top3.eval(data)  # Top-3

        sample_count += minibatch_size

    print("Top-1 Accuracy {:.2f}%".format((num_samples - top1_error) / num_samples * 100))
    print("Top-3 Accuracy {:.2f}%".format((num_samples - top3_error) / num_samples * 100))
    
