import cntk as C
import cv2
import numpy as np
import random
import time

is_fast = False

category = ["background", "wall", "building, edifice", "sky", "floor, flooring", "tree", "ceiling", "road, route",
            "bed", "windowpane, window", "grass", "cabinet", "sidewalk, pavement",
            "person, individual, someone, somebody, mortal, soul", "earth, ground", "door, double door", "table",
            "mountain, mount", "plant, flora, plant life", "curtain, drape, drapery, mantle, pall", "chair",
            "car, auto, automobile, machine, motorcar", "water", "painting, picture", "sofa, couch, lounge","shelf",
            "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair", "seat", "fence, fencing", "desk",
            "rock, stone", "wardrobe, closet, press", "lamp", "bathtub, bathing tub, bath, tub", "railing, rail",
            "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
            "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper",
            "fireplace, hearth, open fireplace", "refrigerator, icebox", "grandstand, covered stand", "path",
            "stairs, steps", "runway", "case, display case, showcase, vitrine",
            "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase",
            "river", "bridge, span", "bookcase", "blind, screen", "coffee table, cocktail table",
            "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop",
            "stove, kitchen stove, range, kitchen range, cooking stove", "palm, palm tree", "kitchen island",
            "computer, computing machine, computing device, data processor, electronic computer, information processing system",
            "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty",
            "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle",
            "towel", "light, light source", "truck, motortruck", "tower", "chandelier, pendant, pendent",
            "awning, sunshade, sunblind", "streetlight, street lamp", "booth, cubicle, stall, kiosk",
            "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box",
            "airplane, aeroplane, plane", "dirt track", "apparel, wearing apparel, dress, clothes", "pole",
            "land, ground, soil", "bannister, banister, balustrade, balusters, handrail",
            "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle",
            "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship",
            "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy",
            "washer, automatic washer, washing machine", "plaything, toy", "swimming pool, swimming bath, natatorium",
            "stool", "barrel, cask", "basket, handbasket", "waterfall, falls", "tent, collapsible shelter", "bag",
            "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
            "trade name, brand name, brand, marque", "microwave, microwave oven", "pot, flowerpot",
            "animal, animate being, beast, brute, creature, fauna", "bicycle, bike, wheel, cycle ", "lake",
            "dishwasher, dish washer, dishwashing machine", "screen, silver screen, projection screen",
            "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
            "traffic light, traffic signal, stoplight", "tray",
            "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
            "fan", "pier, wharf, wharfage, dock", "crt screen", "plate", "monitor, monitoring device",
            "bulletin board, notice board", "shower", "radiator", "glass, drinking glass", "clock", "flag"]

img_channel = 3
img_height = 320
img_width = 480
num_classes = 150 + 1  # 150 categories + background

color = {0: [0, 0, 0]}
color.update({c: [random.randint(0, 255) for _ in range(3)] for c in range(1, num_classes)})


if __name__ == "__main__":
    model = C.load_model("./rtss.model")

    cap_height = 1080
    cap_width = 1920

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)

    #
    # video demo
    #
    if is_fast:
        while True:
            _, frame = cap.read()

            start = time.perf_counter()

            output = model.eval({model.arguments[0]: np.ascontiguousarray(
                cv2.resize(frame, (img_width, img_height)).transpose(2, 0, 1), dtype="float32")})[0]

            predict = output.argmax(axis=0)

            cv2.imshow("Real-Time Semantic Segmentation", predict.astype("uint8"))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            end = time.perf_counter()

            print("FPS %.1f" % (1.0 / (end - start)))

        cap.release()
        cv2.destroyAllWindows()

    else:
        display = np.zeros((img_height, img_width, img_channel), dtype="uint8")
        while True:
            _, frame = cap.read()

            start = time.perf_counter()

            output = model.eval({model.arguments[0]: np.ascontiguousarray(
                cv2.resize(frame, (img_width, img_height)).transpose(2, 0, 1), dtype="float32")})[0]

            predict = output.argmax(axis=0)
            for i in range(img_height):
                for j in range(img_width):
                    display[i, j] = color[predict[i, j]]

            cv2.imshow("Real-Time Semantic Segmentation", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            end = time.perf_counter()

            print("FPS %.1f" % (1.0 / (end - start)))

        cap.release()
        cv2.destroyAllWindows()
        
