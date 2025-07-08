import os
import json
import numpy as np


# all meta file loaders
def get_mapillary_vistas_meta(root, file):
    with open(os.path.join(root, file)) as f:
        config = json.load(f)
    MAPILLARY_VISTAS_SEM_SEG_CATEGORIES = config["labels"]

    stuff_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    evaluate = [True if ((k["evaluate"]) and not ("void" in k["name"])) else False for k in
                MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "evaluate": evaluate
    }
    return ret


def city_meta(root, file):
    with open(os.path.join(root, file)) as f:
        config = json.load(f)
    CITYSCAPES_SEM_SEG_CATEGORIES = config

    # assume this matches with txt file!
    stuff_classes = [k["name"] for k in CITYSCAPES_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_SEM_SEG_CATEGORIES]
    evaluate = [True if ((not k["ignoreInEval"]) and (k["trainId"] != 255)) else False
                for k in CITYSCAPES_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "evaluate": evaluate
    }
    return ret

def pascal_voc_meta(root, file):
    with open(os.path.join(root, file)) as f:
        stuff_classes = f.read().split()

    # assume this matches with txt file!
    stuff_colors = color_map_voc()
    evaluate = [False if "void" in c else True for c in stuff_classes]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "evaluate": evaluate
    }
    return ret

def mmor_meta(root, file):
    print(os.path.join(root, file))
    with open(os.path.join(root, file)) as f:
        data = json.load(f)

    stuff_classes = list(data.keys())
    stuff_colors = [v["color"] for _,v in data.items()]
    evaluate = [False if ("void" in c) or ("background" in c) else True for c in stuff_classes]


    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "evaluate": evaluate
    }
    print(ret)
    return ret

def synmedi_meta(root, file):
    with open(os.path.join(root, file)) as f:
        data = json.load(f)
    
    print(data)

    stuff_classes = list(data.keys())[1:]
    stuff_colors = [v["color"] for _,v in data.items()][1:]
    evaluate = [False if ("void" in c) or ("UNLABELLED" in c) else True for c in stuff_classes][1:]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "evaluate": evaluate
    }
    print(ret)
    return ret


def color_map_voc(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def endovis_meta(root, file):
    with open(os.path.join(root, file)) as f:
        config = json.load(f)
    MEDICAL_SEM_SEG_CATEGORIES = config

    # assume this matches with txt file!
    stuff_classes = [k["name"] for k in MEDICAL_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in MEDICAL_SEM_SEG_CATEGORIES]
    evaluate = [True if not ("void" in k["name"]) else False for k in
                MEDICAL_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "evaluate": evaluate
    }
    return ret