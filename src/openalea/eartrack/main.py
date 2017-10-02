#!/usr/bin/python
# coding: utf-8
""" Main executable of eartrack.

"""
import os
import argparse

import cv2
import numpy as np

import openalea.eartrack
import openalea.eartrack.eartrack as et_lib
import openalea.eartrack.binarisation_folder as bin_f
from openalea.deploy.shared_data import shared_data


def print_o(path, text, log, verbose):
    if log:
        f = open(path, 'a')
        f.write(text)
        f.close()
    if verbose:
        print text


def main():
    """ This function is called by eartrack

    To obtain specific help, type::

        eartrack --help
    """
    param_folder = os.path.join(shared_data(openalea.eartrack), "parameters")
    input_folder = os.path.join(shared_data(openalea.eartrack), "images")
    output_folder = os.path.join(os.path.expanduser('~'),
                                 'ear_tracking_results')
    log = False
    verbose = False
    usage = """
eartrack detect and track ear position on image of maize plants.

Example:

       eartrack -i %s -o %s
"""%(input_folder, output_folder)

    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("-i", type=str,
                        help="Select input folder (default : %s)"
                             % input_folder)
    parser.add_argument("-l", "--log", action="store_true",
                        help="Log output in file and save images in selected "
                             "output folder")
    parser.add_argument("-o", type=str,
                        help="Select output folder (default : %s)"
                             % output_folder)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Output verbosity")
    args = parser.parse_args()
    if args.i:
        input_folder = args.i
    if args.log:
        log = True
    if args.o:
        output_folder = args.o
    if args.verbose:
        verbose = True
    if log:
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    # Init parameters and images
    img_desc, parameters = bin_f.init(param_folder, input_folder,
                                      output_folder, log)

    # Read images, calculate binaries and ear tracking
    for plant in img_desc.keys():

        results_folder = os.path.join(output_folder, str(plant), 'results')

        for task in img_desc[plant].keys():
            cabin = img_desc[plant][task]['cabin']
            task_folder = os.path.join(output_folder, str(plant), str(task))
            log_file = os.path.join(task_folder, "log.txt")
            print_o(log_file, "\n++++++++++++++++++++++++++++++++++++++++++\n "
                              "Analysis logs of plant " + str(plant) +
                              " in task " + str(task) + "\n", log, verbose)

            # Read images corresponding to 1 plant and 1 imaging task
            images = bin_f.read_images(img_desc, plant, task)

            # Calculate binaries images
            binaries, mask_top_center = bin_f.binaries_calculation(images,
                                                                   cabin,
                                                                   parameters)

            # Calculate ear tracking
            existing_angles = binaries["side"].keys()
            if isinstance(binaries["top"][0], type(None)):
                print_o(log_file, "Missing top binary image\n", log,
                            verbose)
                raise ValueError("Missing top binary image")
            angles_to_keep, result_img, top_log = (
                et_lib.top_analyse(binaries["top"][0], existing_angles,
                                   mask_top_center))
            print_o(log_file, top_log, log, verbose)
            if log:
                cv2.imwrite(os.path.join(task_folder, "top0.png"), result_img)

            kept_positions = np.empty([0, 3], 'int')
            useful_kept_images = np.empty([0], 'int')
            for angle in angles_to_keep:
                if isinstance(binaries["side"][angle], type(None)):
                    print_o(log_file, "Missing side " + str(angle) +
                                " binary image\n", log, verbose)
                    continue
                positions, useful_img, side_log, img_debug = (
                    et_lib.side_analyse(binaries["side"][angle],
                                        images["side"][angle],
                                        angle,
                                        parameters[cabin]["side"]["pot_height"],
                                        parameters[cabin]["side"]["pot_width"],))
                print_o(log_file, side_log, log, verbose)
                if log:
                    for name, img in img_debug.items():
                        cv2.imwrite(os.path.join(task_folder, name), img)

                kept_positions = np.append(kept_positions, positions, axis=0)
                useful_kept_images = np.append(useful_kept_images, useful_img,
                                               axis=0)
            print_o(log_file, "\n\n******************************\n", log,
                    verbose)
            values = np.empty([0, 2], 'int')
            im = np.empty([0], 'int')
            result_text = ""
            pos_to_record = list()
            if kept_positions.shape[0]:
                for i in range(len(kept_positions)):
                    for j in range(kept_positions[i, 2]):
                        values = np.append(values, [kept_positions[i, 0:2]],
                                           axis=0)
                        im = np.append(im, useful_kept_images[i])
                mean_pos, finales_positions, final_kept_images = \
                    et_lib.robust_mean(values, im)
                finales_positions = np.append(finales_positions,
                                              np.array([final_kept_images]).T,
                                              1)
                if not (mean_pos == [-1, -1]).all():
                    log_text = "Ear position : (" + str(mean_pos[1]) + ", " + \
                               str(mean_pos[0]) + ")\n"
                    log_text += "Initials values : \n" + \
                                np.array_str(kept_positions) + "\n"
                    log_text += "Finals values : \n" + \
                                np.array_str(finales_positions) + "\n"
                    print_o(log_file, log_text, log, verbose)
                    if log:
                        finale_img = images["side"][finales_positions[0, 2]].copy()
                        pos = finales_positions[0, 0:2]
                        finale_img[pos[0] - 10:pos[0] + 11,
                                   pos[1] - 10:pos[1] + 11, :] = [0, 255, 255]
                        img_path = os.path.join(results_folder,
                                                str(task) + "_side_" +
                                                str(finales_positions[0, 2]) +
                                                "_finale.png")
                        cv2.imwrite(img_path, finale_img)
                    result_text += str(task) + " : " + \
                                   ";".join(map(str, finales_positions[0, 0:2])) + "\n"
                    pos_to_record.append(finales_positions[0, :])
                elif finales_positions.shape[0] == 2:
                    log_text = "No ear found but 2 probables positions " \
                               "has been kept:\n" + \
                               np.array_str(finales_positions) + "\n"
                    log_text += "Initials values : \n" + \
                                np.array_str(kept_positions) + "\n"
                    print_o(log_file, log_text, log, verbose)
                    if log:
                        finale_img = images["side"][finales_positions[0, 2]].copy()
                        for pos in finales_positions:
                            finale_img[pos[0] - 10:pos[0] + 11,
                                       pos[1] - 10:pos[1] + 11, :] = \
                                [0, 255, 255]
                        img_path = os.path.join(results_folder,
                                                str(task) + "_side_" +
                                                str(finales_positions[0, 2]) +
                                                "_finale.png")
                        cv2.imwrite(img_path, finale_img)
                    result_text += str(task) + " : "
                    result_text += " and ".join(";".join(map(str, [pos[0], pos[1]]))
                                            for pos in finales_positions)
                    result_text += "\n"
                    pos_to_record.append(finales_positions[0, :])
                    pos_to_record.append(finales_positions[1, :])
                else:
                    log_text = "No ear found after side views images " \
                               "analysis\n"
                    log_text += "Initials values : \n" + \
                                np.array_str(kept_positions) + "\n"
                    print_o(log_file, log_text, log, verbose)
                    result_text += str(task) + ";" + str(mean_pos[0]) + ";" + \
                                  str(mean_pos[1]) + "\n"
                    pos_to_record.append(None)
            else:
                print_o(log_file,"Not any side view allow to detect ear \n",
                        log, verbose)
                result_text += str(task) + " : No result\n"
                pos_to_record.append(None)

            print result_text
