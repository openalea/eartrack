# coding: utf-8

import os
import re
import cv2
import json
import types
import numpy

import openalea.eartrack.binarisation as bin


def init(param_folder, input_folder, output_folder, log):
    files = os.listdir(input_folder)

    pattern = '^plant\-([0-9]*)\_task\-([0-9]*)\_(s|t)v([0-9]*)\_(cabin\-1|2)\.png$'

    img_desc = dict()
    for f in files:
        m = re.match(pattern, f)
        if m:
            plant = int(m.group(1))
            task = int(m.group(2))
            view = 'top' if m.group(3) == 't' else 'side'
            angle = int(m.group(4))
            cabin = m.group(5)

            if plant not in img_desc.keys():
                img_desc[plant] = dict()

            if task not in img_desc[plant].keys():
                img_desc[plant][task] = dict({'top': dict(), 'side': dict(),
                                              'cabin': cabin})

            # Create output folders
            if log:
                plant_folder = os.path.join(output_folder, str(plant))
                if not os.path.isdir(plant_folder):
                    os.mkdir(plant_folder)

                result_folder = os.path.join(plant_folder, 'results')
                if not os.path.isdir(result_folder):
                    os.mkdir(result_folder)

                task_folder = os.path.join(plant_folder, str(task))
                if not os.path.isdir(task_folder):
                    os.mkdir(task_folder)

            img_desc[plant][task][view][angle] = os.path.join(input_folder, f)

        else:
            raise ValueError("Error in filename " + f)

    # get binarisation parameters
    # param_folder = os.path.join(input_folder, 'parameters')
    param_file = os.path.join(param_folder, 'parameters.json')
    parameters = json.load(open(param_file))

    # Get background and masks images
    for cabin in parameters.keys():
        for view in parameters[cabin].keys():
            for param, value in parameters[cabin][view].items():
                if type(value) in types.StringTypes and '.png' in value:
                    method = cv2.IMREAD_GRAYSCALE if 'mask' in param else \
                        cv2.IMREAD_COLOR
                    parameters[cabin][view][param] = \
                        cv2.imread(os.path.join(param_folder, value), method)
                if type(value) == types.ListType:
                    parameters[cabin][view][param] = \
                        tuple(parameters[cabin][view][param])

    return img_desc, parameters


def read_images(img_desc, plant, task):
    images = dict({'top': dict(), 'side': dict()})
    for a in img_desc[plant][task]['top'].keys():
        images['top'][a] = cv2.imread(img_desc[plant][task]['top'][a],
                                      cv2.IMREAD_COLOR)
    for a in img_desc[plant][task]['side'].keys():
        images['side'][a] = cv2.imread(img_desc[plant][task]['side'][a],
                                       cv2.IMREAD_COLOR)
    return images


def binaries_calculation(images, cabin, param):
    binaries = dict({'top': dict(), 'side': dict()})
    for a in images['top'].keys():
        binaries['top'][a] = bin.color_tree(images['top'][a],
                                            cabin=cabin,
                                            mask_pot=param[cabin]["top"]["mask_top_pot"],
                                            mask_rails=param[cabin]["top"]["mask_top_rails"],
                                            empty_img=param[cabin]["top"]["background"])

    mask_top_center = numpy.zeros(binaries['top'][a].shape)
    height = int(mask_top_center.shape[0]/3)
    width = int(mask_top_center.shape[1]/3)
    mask_top_center[height:height*2, width:width*2] = 255

    mean_image = bin.mean_image([images['side'][angle] for angle in
                                 images['side'].keys()])
    for a in images['side'].keys():
        binaries['side'][a] = bin.meanshift_hsv(images['side'][a],
                                                mean_image,
                                                threshold=param[cabin]["side"]["meanshift_threshold"],
                                                hsv_min=param[cabin]["side"]["hsv_threshold_min"],
                                                hsv_max=param[cabin]["side"]["hsv_threshold_max"],
                                                mask_mean_shift=param[cabin]["side"]["mask_mean_shift"],
                                                mask_hsv=param[cabin]["side"]["mask_hsv"])

    return binaries, mask_top_center
