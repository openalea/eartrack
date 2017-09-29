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


def main():
    """ his function is called by eartrack

    To obtain specific help, type::

        eartrack --help
    """
    param_folder = os.path.join(shared_data(openalea.eartrack), "parameters")
    input_folder = os.path.join(shared_data(openalea.eartrack), "images")
    output_folder = os.path.join(os.path.expanduser('~'),
                                 'ear_tracking_results')

    usage = """
eartrack detect and track ear position on image of maize plants.

Example:

       eartrack -i %s -o %s
"""%(input_folder, output_folder)


    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("-i", type=str,
                        help="Select input folder (default : %s)"%(input_folder))
    parser.add_argument("-o", type=str,
                        help="Select output folder (default : %s)"%(output_folder))

    args = parser.parse_args()
    if args.i:
        input_folder = args.i
    if args.o:
        output_folder = args.o


    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Init parameters and images
    img_desc, parameters = bin_f.init(param_folder, input_folder, output_folder)

    # Read images, calculate binaries and ear tracking
    for plant in img_desc.keys():

        results_folder = os.path.join(output_folder, str(plant), 'results')

        for task in img_desc[plant].keys():
            cabin = img_desc[plant][task]['cabin']
            task_folder = os.path.join(output_folder, str(plant), str(task))

            # Read images corresponding to 1 plant and 1 imaging task
            images = bin_f.read_images(img_desc, plant, task)

            # Calculate binaries images
            binaries, mask_top_center = bin_f.binaries_calculation(images,
                                                                   cabin,
                                                                   parameters)

            # Calculate ear tracking
            # TODO: The directory may be not writable. Use with cmd instead.
            log_file = open(os.path.join(task_folder, "log.txt"), 'a')
            existing_angles = binaries["side"].keys()
            if isinstance(binaries["top"][0], type(None)):
                log_file.write("Missing top binary image\n")
                raise ValueError("Missing top binary image")
            angles_to_keep, result_img, top_log = (
                et_lib.top_analyse(binaries["top"][0], existing_angles,
                                   mask_top_center))
            log_file.write("Analysis logs of plant " + str(plant) + "\n\n")
            log_file.write(top_log)
            cv2.imwrite(os.path.join(task_folder, "top0.png"), result_img)

            # print str(finTop - begin_time)
            kept_positions = np.empty([0, 3], 'int')
            useful_kept_images = np.empty([0], 'int')
            for angle in angles_to_keep:
                # debutSide = time.time()
                if isinstance(binaries["side"][angle], type(None)):
                    log_file.write("Missing side " + str(angle) +
                                   " binary image\n")
                    continue
                positions, useful_img, debug = (
                    et_lib.side_analyse(binaries["side"][angle],
                                        images["side"][angle],
                                        angle,
                                        task_folder,
                                        parameters[cabin]["side"]["pot_height"],
                                        parameters[cabin]["side"]["pot_width"],))
                kept_positions = np.append(kept_positions, positions, axis=0)
                useful_kept_images = np.append(useful_kept_images, useful_img,
                                               axis=0)
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
                    log_file.write("Ear position : (" + str(mean_pos[1]) +
                                   ", " + str(mean_pos[0]) + ")\n")
                    log_file.write("Initials values : \n")
                    log_file.write(np.array_str(kept_positions) + "\n")
                    log_file.write("Finals values : \n")
                    log_file.write(np.array_str(finales_positions) + "\n")

                    imageFinale = images["side"][finales_positions[0, 2]].copy()
                    imageFinale[mean_pos[0] - 10:mean_pos[0] + 11,
                    mean_pos[1] - 10:mean_pos[1] + 11, :] = [0, 255, 255]
                    cv2.imwrite(os.path.join(results_folder, str(task) +
                                             "_side_" +
                                             str(finales_positions[0, 2]) +
                                             "_finale.png"),
                                imageFinale)
                    result_text += str(task) + ";" + str(mean_pos[0]) + ";" + \
                                  str(mean_pos[1]) + "\n"
                    # pos_to_record.append(finales_positions[0,:])
                    mean_pos = np.append(mean_pos, finales_positions[0, 2])
                    pos_to_record.append(mean_pos)
                elif finales_positions.shape[0] == 2:
                    log_file.write("No ear found but 2 probables positions has "
                                   "been kept:\n")
                    log_file.write(np.array_str(finales_positions) + "\n")
                    log_file.write("Initials values : \n")
                    log_file.write(np.array_str(kept_positions) + "\n")
                    imageFinale = images["side"][finales_positions[0, 2]].copy()
                    result_text += str(task)
                    for pos in finales_positions:
                        imageFinale[pos[0] - 10:pos[0] + 11,
                        pos[1] - 10:pos[1] + 11, :] = [0, 255, 255]
                        result_text += ";" + str(pos[0]) + ";" + str(pos[1])
                    cv2.imwrite(os.path.join(results_folder,
                                             str(task) + "_side_" +
                                             str(finales_positions[0, 2]) +
                                             "_finale.png"),
                                imageFinale)
                    result_text += "\n"
                    pos_to_record.append(finales_positions[0, :])
                    pos_to_record.append(finales_positions[1, :])
                else:
                    log_file.write("No ear found after side views images "
                                   "analysis\n")
                    log_file.write("Initials values : \n")
                    log_file.write(np.array_str(kept_positions) + "\n")
                    result_text += str(task) + ";" + str(mean_pos[0]) + ";" + \
                                  str(mean_pos[1]) + "\n"
                    pos_to_record.append(None)
            else:
                log_file.write("Not any side view allow to detect ear \n")
                pos_to_record.append(None)
