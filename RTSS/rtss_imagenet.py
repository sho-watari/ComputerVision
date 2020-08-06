import cv2
import glob
import os
import requests

from requests.exceptions import Timeout

dir_file = "./ImageNet/"
imagenet_url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}"
timeout = 3


def download_imagenet():
    with open("./imagenet1000.txt") as f:
        image_id_list = f.readlines()
        
    if not os.path.exists(dir_file):
        os.mkdir(dir_file)

    for image_id in image_id_list:
        #
        # get image url
        #
        print(image_id)
        image_id = image_id.split()[0]

        if not os.path.exists(image_id):
            os.mkdir(dir_file + image_id)

        resp = requests.get(imagenet_url.format(image_id))

        image_url_list = resp.content.decode().split()

        for i, image_url in enumerate(image_url_list):
            #
            # save image file
            #
            savename = dir_file + image_id + "/" + image_id + "_{}.jpg".format(i)

            try:
                resp = requests.get(image_url, timeout=timeout)

                img = resp.content
                with open(savename, "wb") as f:
                    f.write(img)
                print(savename)

            except Timeout:
                print("Timeout ->", image_url)
                continue

            except:
                print("Error ->", image_url)
                continue


def cleaning_imagenet(dir_file):
    file_list = glob.glob(dir_file + "*/*.jpg")
    for file in file_list:
        try:
            img = cv2.imread(file)
            img.shape
        except AttributeError:
            print("Delete ->", file)
            os.remove(file)


def vovnet57_imagenet(dir_file):
    dir_list = glob.glob(dir_file + "*")

    with open("./train_imagenet_map.txt", "w") as map_file:
        for i, dir_name in enumerate(dir_list):
            file_list = glob.glob(dir_name + "/*.jpg")

            for file in file_list:
                map_file.write("%s\t%d\n" % (file, i))


if __name__ == "__main__":
    download_imagenet()

    cleaning_imagenet(dir_file)

    vovnet57_imagenet(dir_file)
    
