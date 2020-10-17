import h5py
import numpy as np

layers = [[3, 32, 3],
          [32, 64, 3],
          [64, 128, 3],
          [128, 64, 1],
          [64, 128, 3],
          [128, 256, 3],
          [256, 128, 1],
          [128, 256, 3],
          [256, 512, 3],
          [512, 256, 1],
          [256, 512, 3],
          [512, 256, 1],
          [256, 512, 3],
          [512, 1024, 3],
          [1024, 512, 1],
          [512, 1024, 3],
          [1024, 512, 1],
          [512, 1024, 3],
          [1024, 1024, 3],
          [1024, 1024, 3]]


if __name__ == "__main__":
    file = open("./yolov2.weights", "rb")

    with h5py.File("./yolov2.h5", "w") as f:
        data = np.fromfile(file, dtype="float32")[4:]

        offset = 0
        for i, l in enumerate(layers):
            in_ch, out_ch, ksize = l[0], l[1], l[2]

            f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/bn%d/scale/" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/bn%d/mean" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/bn%d/variance" % (i + 1), data=data[offset: offset + out_ch])
            offset += out_ch
            f.create_dataset("params/conv%d/weights" % (i + 1),
                             data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                                 out_ch, in_ch, ksize, ksize))
            offset += (out_ch * in_ch * ksize * ksize)

        in_ch, out_ch, ksize, i = 512, 64, 1, 20

        f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/scale/" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/mean" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/variance" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/conv%d/weights" % (i + 1),
                         data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                             out_ch, in_ch, ksize, ksize))
        offset += (out_ch * in_ch * ksize * ksize)

        in_ch, out_ch, ksize, i = 1280, 1024, 3, 21

        f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/scale/" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/mean" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/bn%d/variance" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/conv%d/weights" % (i + 1),
                         data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                             out_ch, in_ch, ksize, ksize))
        offset += (out_ch * in_ch * ksize * ksize)

        in_ch, out_ch, ksize, i = 1024, 425, 1, 22

        f.create_dataset("params/conv%d/bias" % (i + 1), data=data[offset: offset + out_ch])
        offset += out_ch
        f.create_dataset("params/conv%d/weights" % (i + 1),
                         data=data[offset: offset + (out_ch * in_ch * ksize * ksize)].reshape(
                             out_ch, in_ch, ksize, ksize))
        offset += (out_ch * in_ch * ksize * ksize)
        
