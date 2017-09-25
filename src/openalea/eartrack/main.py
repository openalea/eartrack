#!/usr/bin/python
# coding: utf-8
""" Main executable of eartrack.

"""
import os
import cv2
import numpy as np

#import ear_tracking_library_standalone as et_lib
import openalea.eartrack.eartrack as et_lib
import openalea.eartrack.binarisation_folder as st_lib


output_folder = os.path.join(os.path.expanduser('~'), 'ear_tracking_results')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# Read images' information and create output folders
# TODO read images from zenodo
images_folder = os.path.join(os.path.expanduser('~'), 'Bureau',
                             'ear_tracking_input')

img_desc, parameters = st_lib.init(images_folder, output_folder)

# Read images, calculate binaries and ear tracking
for plant in img_desc.keys():

    results_folder = os.path.join(output_folder, str(plant), 'results')

    for task in img_desc[plant].keys():
        cabin = img_desc[plant][task]['cabin']
        task_folder = os.path.join(output_folder, str(plant), str(task))

        # Read images corresponding to 1 plant and 1 imaging task
        images = st_lib.read_images(img_desc, plant, task)

        # Calculate binaries images
        binaries, mask_top_center = st_lib.binaries_calculation(images,
                                                                cabin,
                                                                parameters)

        # Calculate ear tracking
        log_file = open(os.path.join(task_folder, "log.txt"), 'a')
        existing_angles = binaries["side"].keys()
        if isinstance(binaries["top"][0], type(None)):
            log_file.write("Missing top binary image\n")
            raise "Missing top binary image"
        angles_to_keep, result_img, top_log = \
            et_lib.top_analyse(binaries["top"][0], existing_angles,
                               mask_top_center)
        log_file.write("Analysis logs of plant " + str(plant) + "\n\n")
        log_file.write(top_log)
        cv2.imwrite(os.path.join(task_folder, "top0.png"), result_img)

        # print str(finTop - begin_time)
        kept_positions = np.empty([0, 3], 'int')
        useful_kept_images = np.empty([0], 'int')
        for angle in angles_to_keep:
            # debutSide = time.time()
            if isinstance(binaries["side"][angle], type(None)):
                log_file.write("Missing side " + str(angle) + " binary image\n")
                continue
            positions, imagesUtiles, debug = et_lib.side_analyse(binaries["side"][angle],
                                                                 images["side"][angle],
                                                                 angle,
                                                                 task_folder,
                                                                 parameters[cabin]["side"]["pot_height"],
                                                                 parameters[cabin]["side"]["pot_width"],)
            kept_positions = np.append(kept_positions, positions, axis=0)
            useful_kept_images = np.append(useful_kept_images, imagesUtiles,
                                           axis=0)
            # print "fin side " + str(angle) + " : " + str(time.time() - debutSide)
            # ~ for imageUtile in imagesUtiles:
            # ~ useful_kept_images.append(imageUtile)
        values = np.empty([0, 2], 'int')
        # ~ im = list()
        im = np.empty([0], 'int')
        # mean_pos = np.array([-1,-1])
        resultText = ""
        pos_to_record = list()
        if kept_positions.shape[0]:
            for i in range(len(kept_positions)):
                for j in range(kept_positions[i, 2]):
                    values = np.append(values, [kept_positions[i, 0:2]],
                                       axis=0)
                    # ~ im.append(useful_kept_images[i])
                    im = np.append(im, useful_kept_images[i])
                    # ~ imagesUtiles.append(imagesUtiles[i])
            mean_pos, finales_positions, final_kept_images = \
                et_lib.robust_mean(values, im)
            finales_positions = np.append(finales_positions,
                                          np.array([final_kept_images]).T, 1)
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
                cv2.imwrite(os.path.join(results_folder, str(task) + "_side_" +
                                         str(finales_positions[0, 2]) +
                                         "_finale.png"),
                            imageFinale)
                resultText += str(task) + ";" + str(mean_pos[0]) + ";" + str(
                    mean_pos[1]) + "\n"
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
                resultText += str(task)
                for pos in finales_positions:
                    imageFinale[pos[0] - 10:pos[0] + 11,
                    pos[1] - 10:pos[1] + 11, :] = [0, 255, 255]
                    resultText += ";" + str(pos[0]) + ";" + str(pos[1])
                cv2.imwrite(os.path.join(results_folder,
                                         str(task) + "_side_" +
                                         str(finales_positions[0, 2]) +
                                         "_finale.png"),
                            imageFinale)
                resultText += "\n"
                pos_to_record.append(finales_positions[0, :])
                pos_to_record.append(finales_positions[1, :])
            else:
                log_file.write("No ear found after side view images analysis\n")
                log_file.write("Initials values : \n")
                log_file.write(np.array_str(kept_positions) + "\n")
                resultText += str(task) + ";" + str(mean_pos[0]) + ";" + \
                              str(mean_pos[1]) + "\n"
                pos_to_record.append(None)
        else:
            log_file.write("Not any side view allow to detect ear \n")
            pos_to_record.append(None)