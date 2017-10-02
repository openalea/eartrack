# -*- python -*-
# -*- coding: utf-8 -*-
"""eartrack library provides useful functions to track ear on maize
"""

import os
import math as m
import multiprocessing as mp

import numpy as np
import cv2

from skimage import measure
from skimage.morphology import skeletonize, medial_axis, label
from skimage import graph

# import matplotlib.pyplot as plt

import openalea.eartrack.binarisation as bin

writing_semaphore = mp.BoundedSemaphore()


def top_analyse(top_binary_img, existing_angles, center_mask):
    """
    This function analyses top binary image to determine best side view images
    to see the stem and find ear

    :param top_binary_img: (numpy array of uint8) representing binary image
    :param existing_angles: (list of int) list of existing angle for this
    snapshot
    :param center_mask: (numpy array of uint8) mask representing  the center of
    image to know if a leave can be considered as obstructing
    :return:
        (list of int) informative angles of view to analyse
        (numpy array of uint8) result image for log
        (string) log to write
    """

    log = ""

    # Determination of most informative angles for ear tracking
    result_img, alpha90, alpha270, exclusions = \
        get_view_angles(top_binary_img[::-1, ::-1], center_mask)
    if alpha90 == -1 and alpha270 == -1:
        log += "Binarisation error for top view image\n\n"
        return list(), result_img, log

    # TODO : refaire cette méthode plus proprement
    existing_angles.sort()

    angles_to_keep = list()
    for angle in (alpha90, alpha270):
        if angle > 345:
            angle -= 360
        for i in range(len(existing_angles)):
            if abs(existing_angles[i] - angle) <= 10:
                if i > 0:
                    angles_to_keep.append(existing_angles[i-1])
                else:
                    angles_to_keep.append(existing_angles[len(existing_angles)-1])

                angles_to_keep.append(existing_angles[i])

                if i < len(existing_angles)-1:
                    angles_to_keep.append(existing_angles[i+1])
                else:
                    angles_to_keep.append(existing_angles[0])
                break
            elif abs(existing_angles[i] - angle) <= 15:
                angles_to_keep.append(existing_angles[i])
                if existing_angles[i] < angle:
                    if i < len(existing_angles)-1:
                        angles_to_keep.append(existing_angles[i+1])
                    else:
                        angles_to_keep.append(existing_angles[0])
                else:
                    if i > 0:
                        angles_to_keep.append(existing_angles[i-1])
                    else:
                        angles_to_keep.append(existing_angles[len(existing_angles)-1])
    angles_to_keep.sort()

    # Exclude some angles which could have leaves hamper the stem detection
    excluded_angles = list()
    for exclude_angle in exclusions:
        exclude_negatives_angles = 1000
        if exclude_angle > 335:
            exclude_negatives_angles = exclude_angle - 360
        i = 0
        while i < len(angles_to_keep):
            if abs(exclude_angle - angles_to_keep[i]) < 25:
                excluded_angles.append(angles_to_keep.pop(i))
            elif abs(exclude_negatives_angles - angles_to_keep[i]) < 25:
                excluded_angles.append(angles_to_keep.pop(i))
            else:
                i += 1

    # Log intermediate and final data on top images analyse
    log += "Top view analyzed : alpha = " + str(alpha90) + ", alpha2 = " + \
             str(alpha270)
    if len(exclusions):
        log += "\nExcluded angles : " + ";".join(map(str, exclusions))
    log += "\n\n"
    log += "Interesting angles : " + \
           ";".join(map(str, sorted(angles_to_keep + excluded_angles))) + "\n"
    log += "Final kept angles : " + ";".join(map(str, angles_to_keep)) + "\n"
    if not len(angles_to_keep):
        log += "All views has been excluded\n"

    return angles_to_keep, result_img, log


def side_analyse(binary_img, color_img, angle, pot_height, pot_center):
    """ Perform the side analyse an image of side view maize plant

    :param binary_img: (numpy array of uint8) binary image
    :param color_img: (numpy array of uint8) color image in BGR matrix
    :param angle: (int) view angle of the image
    :param pot_height: (int) height position of the top of the pot
    :param pot_center: (int) width position of the center of the pot
    :return: positions: (np array of uint numpy array) Kept position(s) as
      probable(s) ear(s), each position as [x, y, angle]
             useful_images: (np array of str) ids of images corresponding to
      each position
             log: (string) log to write
             img_debug: (list of numpy array) list of output images from
      differents stages of calculation
    """
    positions = np.empty([0, 3], 'int')
    useful_images = np.empty([0], 'int')
    log = ""
    img_debug = dict()

    image_name = "side_" + str(angle) + ".png"
    log += "\n-----------------------------\n"
    log += "Loading " + image_name + "\n"
    ''' LOADING BINARY AND ORIGINAL IMAGE '''

    binary_img = bin.close(binary_img, iterations=4)

    name, ext = os.path.splitext(image_name)

    # Get the biggest region
    biggest_binary_region = binary_biggest_region(binary_img)

    # Extract skeleton of plant
    output_skeleton_img = get_skeleton(biggest_binary_region)

    # Extract distance transform
    dist_trans_img = distance_transform(biggest_binary_region)

    # skimage's graph library and skeleton cleaning
    begin, end = get_endpoints(output_skeleton_img, pot_center, pot_height)
    if begin == [-1, -1]:
        log += "Error in bottom's stem detection\n\n"
        return positions, useful_images, log, img_debug
    output_skeleton_img = skeleton_cleaning(output_skeleton_img, begin)
    route = find_cross_route(output_skeleton_img, begin)
    route.reverse()

    # Make color image with distance transform '''
    output_dt_img = dist_trans_img*255/dist_trans_img.max()
    output_dt_img = output_dt_img.astype(int)

    # Make image binary and skeletons
    output_binary_img = np.zeros(color_img.shape, 'uint8')
    output_binary_img[:, :, 0] = biggest_binary_region
    output_binary_img[:, :, 1] = biggest_binary_region
    output_binary_img[:, :, 2] = biggest_binary_region
    for pix in route:
        output_binary_img[pix[0], pix[1]-2:pix[1]+2, :] = (0, 0, 255)

    # Get main direction of stem, rotate the stem and adapt on it the
    # following derivation algorithme
    init_stem = np.zeros(biggest_binary_region.shape, 'uint8')
    for pix in route:
        mask = dist_trans_img[pix[0], pix[1]]
        init_stem[pix[0]-mask:pix[0]+mask+1, pix[1]-mask:pix[1]+mask+1] = 255
    output_stem_img, a, b, r_xy, alpha = majors_axes_regression_line(init_stem)

    # Perform derivation on route to
    diff, x, y = derivate(route)

    # Eliminate noise on derivation curve
    indices = differential_cleaning(diff, x, y, 10, 5, 5)

    # Delete extrema error
    i = len(indices)-1
    while indices[i][2] == 0:
        i -= 1
    if x[len(y)-1] == x[indices[i][0]] or \
            abs(float(y[len(y)-1] - y[indices[i][0]])/float(x[len(y)-1] -
                                                    x[indices[i][0]]) - a) > 1:
        for j in range(len(indices)-1, i-1, -1):
            route = route[:indices[i][0]]
            indices.pop(len(indices)-1)
    i = 0
    while indices[i][2] == 0:
        i += 1
    if x[indices[i][1]] == x[0] or \
            abs(float(y[indices[i][1]] - y[0])/float(x[indices[i][1]] -
                                                    x[0]) - a) > 1:
        for j in range(i+1):
            route = route[indices[0][1]:]
            indices.pop(0)

    # Stem reconstruction
    cleaned_stem = np.zeros(biggest_binary_region.shape, 'uint8')
    for pix in route:
        mask = dist_trans_img[pix[0], pix[1]]
        cleaned_stem[pix[0]-mask:pix[0]+mask+1, pix[1]-mask:pix[1]+mask+1] = \
            biggest_binary_region[pix[0]-mask:pix[0]+mask+1,
            pix[1]-mask:pix[1]+mask+1]

    output_stem_img, a, b, r_xy, alpha = majors_axes_regression_line(cleaned_stem)

    if r_xy > 30:
        log += "Stem detection error\n\n"
        img_debug[name + "_stem_error" + ext] = output_stem_img
        return positions, useful_images, log, img_debug

    skeleton_stem = np.zeros(binary_img.shape, 'uint8')
    for pixel in route:
        skeleton_stem[pixel] = 1
    begin, end = get_endpoints(skeleton_stem, pot_center, pot_height)
    if begin == [-1, -1] or end == [-1, -1]:
        log += "Error in bottom or top of stem detection after cleaning stem"
        return positions, useful_images, log, img_debug
    route = find_route(skeleton_stem, begin, end)

    # Statistics on distances curve to detect probable ear position
    distances = get_distances(route, dist_trans_img)
    distances_length = float(len(distances))

    part_1 = int(round(len(distances)/2.5))

    position = 0

    solutions, stems, pics, poses = ear_detection(distances)
    minus_pos = poses[0]
    stem_pos_after_ear = poses[1]

    kept_solutions = -1
    for i in range(len(solutions)):
        if solutions[i][1] > 0:
            positions = np.append(positions, [[route[solutions[i][0]][0],
                                               route[solutions[i][0]][1],
                                               solutions[i][1]]], axis=0)
            useful_images = np.append(useful_images, angle)
            if kept_solutions < 0:
                kept_solutions = i
                position = solutions[i][0]
            elif solutions[i][1] > solutions[kept_solutions][1]:
                kept_solutions = i
                position = solutions[i][0]

    log += "Stem width bellow the ear = " + str(distances[minus_pos]) + "\n"
    if kept_solutions >= 0:
        log += "Stem with up to the ear = " + \
               str(distances[stem_pos_after_ear]) + "\n"

        log += "Probable ear position : " + str(route[position][0]) + "\n"

    else:
        log += "Ear detection error\n"

    log += "Solutions : \n"
    for sol in solutions:
        log += "\tsolution : " + str(route[sol[0]][0]) + ", weight : " + \
                 str(sol[1]) + "\n"
    log += "Peaks (leaves) : \n"
    for pic in pics:
        log += "\tpeak : " + str(route[pic[0]][0]) + ", begin : " + \
                 str(route[pic[1]][0]) + ", end : " + str(route[pic[2]][0]) + \
                 ", relative length : " + \
                 str(float(pic[2] - pic[1])*100./distances_length) + "\n"
    log += "Troughs (stem part) : \n"
    for stem in stems:
        log += "\tbegin : " + str(route[stem[0]][0]) + ", end : " + \
                 str(route[stem[1]][0]) + ", relative length : " + \
                 str(float(stem[1] - stem[0])*100./distances_length) + "\n"
    log += "\n"

    # BUG X11 to generate figure
    # plt.clf()
    # plt.plot(distances)
    # plt.plot(minus_pos, -1,'r*')
    # plt.plot(stem_pos, 0,'g*')
    # plt.plot(position, -1,'b*')
    # for i in pics[:,0]:
    #    plt.plot(i,distances[i], 'r*')
    # for stem in stems:
    #    plt.plot(range(stem[0],stem[1]),distances[stem[0]:stem[1]],'r')
    # plt.savefig(os.path.join(output_folder,"courbe_"+name+ext))

    # yellow square on solution
    output_results_img = color_img.copy()
    output_results_img[route[position][0]-5:route[position][0]+5,
                  route[position][1]-5:route[position][1]+5, :] = (0, 255, 255)

    for i in range(len(route)):
        output_binary_img[route[i][0], route[i][1]-2:route[i][1]+2, :] = (0, 255, 0)
        if i < stem_pos_after_ear:
            mask = distances[minus_pos]
        else:
            mask = distances[stem_pos_after_ear]
        if distances[i] == distances[minus_pos] and i < part_1:
            output_results_img[route[i][0], route[i][1]-mask:route[i][1]+mask+1, 0] \
                = 255
        elif distances[i] == distances[stem_pos_after_ear] \
                and i >= part_1 and position:
            output_results_img[route[i][0], route[i][1]-mask:route[i][1]+mask+1, 1] \
                = 255
        else:
            output_results_img[route[i][0], route[i][1]-mask:route[i][1]+mask+1, 2] \
                = 255
    output_binary_img[route[position][0]-5:route[position][0]+5,
                      route[position][1]-5:route[position][1]+5, :] = (0, 0, 255)
    output_binary_img[route[minus_pos][0]-5:route[minus_pos][0]+5,
                      route[minus_pos][1]-5:route[minus_pos][1]+5, :] = (255, 0, 0)

    # save images
    img_debug[name + "_distance_transform" + ext] = output_dt_img
    img_debug[name + "_result" + ext] = output_results_img
    img_debug[name + "_binary" + ext] = output_binary_img
    img_debug[name + "_cleaned_stem" + ext] = output_stem_img
    img_debug[name + "_skeleton" + ext] = output_skeleton_img * 255

    # Log distance values
    log += "Stem width : " + ';'.join(map(str, distances)) + "\n"
    return positions, useful_images, log, img_debug


def get_skeleton(binary_image):
    """ This function use skimage medial axis to perform skeleton on binary
    image

    :param binary_image: (numpy 2D array of binary uint8) binary image to
    perform skeleton
    :return: (numpy 2D array of binary uint8) binary image of skeleton
    """
    return (medial_axis(binary_image > 0)).astype(int)


def distance_transform(binary_image, distance_type=1, mask_size=5):
    """ This function perform opencv distance transform on binary image

    :param binary_image: (numpy 2D array of binary uint8) binary image to
    perform distance transorm
    :param distance_type: see cv::DistanceTypes
    :param mask_size: see cv::DistanceTransformMasks
    :return: (numpy 2D array of uint8) binary image transformed in distances
    """
    return (cv2.distanceTransform(binary_image,
                                  distance_type, mask_size)).astype(int)


def binary_biggest_region(binary_image):
    """ Look for the biggest object on a binary image

    :param binary_image: (numpy 2D array of binary uint8) binary image to
    analyse
    :return: (numpy 2D array of binary uint8) binary image containing only the
    biggest object
    """
    biggest = 0
    lab = 0

    labelled_img = measure.label(binary_image, neighbors=8)
    for region in measure.regionprops(labelled_img):
        if region['area'] > biggest and \
                binary_image[region['coords'][0][0], region['coords'][0][1]]:
            biggest = region['area']
            lab = region['label']

    return binary_image * (labelled_img == lab)


def get_endpoints(skeleton, center, height):
    """ Try to find the bottom and upper node of the stem in a maize plant

    :param skeleton: (numpy 2D array of binary uint8) representing the skeleton
    of side view image of a maize plant
    :param center: (int) pixel in the width center of the pot
    (depending on the plateform and the calibration)
    :param height: (int) pixel in the height top of the pot
    (depending on the plateform and the calibration)
    :return: (list of 2 int) pixel of the bottom of the stem
             (list of 2 int) pixel of the top of the stem
    """
    mini = skeleton.shape[0]
    down_look_for = height
    skelet = np.where(skeleton > 0)
    up_node = [-1, -1]
    down_node = [-1, -1]
    for i in range(0, skelet[0].shape[0]):
        node = tuple([skelet[0][i], skelet[1][i]])
        if node[0] < mini or \
                (node[0] == mini and
                        skeleton[node[0], node[1]-1:node[1]+2].all()):
            mini = node[0]
            up_node = node

    loop_again = 1
    while loop_again:
        if skeleton[down_look_for, :].any():
            indices = np.where(skeleton[down_look_for, :])
            loop_again = 0
            for y in indices[0]:
                if abs(y - center) < abs(down_node[1] - center):
                    down_node = [down_look_for, y]
        down_look_for -= 1
        if down_look_for < 1500:
            loop_again = 0
    return down_node, up_node


def skeleton_cleaning(skeleton, begin):
    """ Clean the skeleton

    :param skeleton: (numpy 2D array of binary uint8) representing the skeleton
    of side view image of maize plant
    :param begin: bottm of stem
    :return: (numpy 2D array of binary uint8) representing cleaned skeleton
    """
    cleaned_skeleton = np.array(skeleton)
    skeleton_inverted = np.array(skeleton, 'float')
    skeleton_inverted[skeleton_inverted == 0] = np.Inf
    mcp = graph.MCP(skeleton_inverted)
    cc, t = mcp.find_costs([begin])

    s = np.where(skeleton)
    cross = list()
    ends = list()
    for i in range(len(s[0])):
        pattern = skeleton[s[0][i] - 1:s[0][i] + 2, s[1][i] - 1:s[1][i] + 2]
        if len(np.where(pattern > 0)[0]) > 3:
            cross.append([s[0][i], s[1][i]])
        elif len(np.where(pattern > 0)[0]) < 3:
            ends.append([s[0][i], s[1][i]])
    for end in ends:
        temp = list()
        current = end
        prec_in_cross = 0
        loop_again = 1
        while loop_again:
            direction = t[current[0], current[1]]
            if direction == -1:
                break
            temp.append(current)
            a = np.zeros([8])
            a[direction] = 1
            a = np.insert(a, 4, 0)
            a = a[::-1]
            a = a.reshape([3, 3])
            next_one = np.where(a == 1)
            current = [current[0] + next_one[0][0]-1,
                       current[1] + next_one[1][0]-1]
            if current in cross:
                prec_in_cross = 1
            else:
                if prec_in_cross:
                    temp.pop()
                    loop_again = 0

        if len(temp) < 100:
            for pixel in temp:
                cleaned_skeleton[pixel[0], pixel[1]] = 0
    return cleaned_skeleton


def find_route(skeleton, begin, end):
    """ Find the shortest route on a skeleton between 2 pixels

    :param skeleton: (numpy 2D array of binary uint8) representing the skeleton
    of side view image of a maize plant
    :param begin: (list of 2 int) pixel of the bottom of the stem
    :param end: (list of 2 int) pixel of the top of the stem
    :return: (list of list of 2 int) list of all the pixels to follow to get
    the shortest path between begin and end
    """
    skeleton_inverted = np.array(skeleton)
    skeleton_inverted[skeleton_inverted == 0] = 255
    return graph.route_through_array(skeleton_inverted, begin, end)[0]


def find_cross_route(skeleton, begin):
    """ Find the shortest route on a skeleton between a beginning pixel and the
    upper cross on the skeleton

    :param skeleton: (numpy 2D array of binary uint8) representing the skeleton
    of side view image of a maize plant
    :param begin: (list of 2 int) pixel of the bottom of the stem
    :return: (list of list of 2 int) list of all the pixels to follow to get
    the shortest path between begin and upper cross
    """
    s = np.where(skeleton)
    end = skeleton.shape
    for i in range(len(s[0])):
        if len(np.where(skeleton[s[0][i]-1:s[0][i]+2,
                                 s[1][i]-1:s[1][i]+2] > 0)[0]) > 3:
            if s[0][i] < end[0]:
                end = [s[0][i], s[1][i]]
    skeleton_inverted = np.array(skeleton)
    skeleton_inverted[skeleton_inverted == 0] = 255
    return graph.route_through_array(skeleton_inverted, begin, end)[0]


def get_distances(route, distance_transform_img):
    """ Find the distances values from the selected route and the distance
    transform's image

    :param route: (list of list of 2 int) list of all the pixels to follow a
    route on image
    :param distance_transform_img: (numpy 2D array of uint8) binary image
    transformed in distances
    :return: (list of int) representing the distances values all along the route
    """
    distances = list()
    for pixel in route:
        distances.append(distance_transform_img[pixel])
    return distances


def derivate(route):
    """ derivate a route in order to analyse increases and decreases
    (variation of directions)

    :param route: (list of list of 2 int) list of all the pixels to follow a
    route on image
    :return: diff: (list of int) values in [-1, 0, 1] representing the variation
    of the route
             x: (list of int) x original position of each diff value
             y: (list of int) y original position of each diff value
    """
    longueur = len(route)
    x = np.zeros([1, 0], 'int')
    y = np.zeros([1, 0], 'int')
    diff = np.zeros([1, 0], 'float')
    i = 0
    while i < longueur-1:
        superior_index = 1
        x = np.append(x, route[i][0])
        y = np.append(y, route[i][1])
        while route[i+superior_index][0] >= route[i][0]:
            x = np.append(x, route[i+superior_index][0])
            y = np.append(y, route[i+superior_index][1])
            diff = np.append(diff, float(route[i+superior_index][1] -
                                         route[i+superior_index-1][1]))
            superior_index += 1
            if i + superior_index == len(route):
                break
        if i + superior_index < len(route):
            if i == 0:
                diff = np.append(diff, float(route[superior_index][1] -
                                             route[0][1]) /
                                 float(route[superior_index][0] - route[0][0]))
            else:
                if superior_index == 1:
                    diff = np.append(diff, float(route[i+superior_index][1] -
                                                 route[i][1]) /
                                     float(route[i+superior_index][0] -
                                           route[i][0]))
                else:
                    diff = np.append(diff,
                                     np.sign(float(route[i+superior_index][1] -
                                                   route[i][1]) /
                                             float(route[i+superior_index][0] -
                                                   route[i][0]))*np.Inf)
        else:
            diff = np.append(diff, 1.)
        i += superior_index

    if not route[longueur-1][0] == route[longueur-2][0]:
        x = np.append(x, route[longueur-1][0])
        y = np.append(y, route[longueur-1][1])
        diff = np.append(diff, float(route[longueur-1][1] -
                                     route[longueur-2][1]) /
                         float(route[longueur-1][0] - route[longueur-2][0]))
    return diff, x, y


def differential_cleaning(diff, x, y, max_space, min_length, min_height):
    """ Analyse derivatives values to keep only the significant variations

    :param diff: (list of int) values in [-1, 0, 1] representing the variation
    of a route
    :param x: (list of int) x original position of each diff value
    :param y: (list of int) y original position of each diff value
    :param max_space: (int) max length (in pixels) of diff null to reckon that
     the increase or decrease is no longer the same variation
    :param min_length: (int) minimum length of variation to reckon that the
    variation is significant
    :param min_height: minimum height of variation to reckon that the
    variation is significant
    :return: (list of 3 int list) describing the diff values by parts of same
    variation [[begin, end, variation]]
    """

    # first loop to separate variations
    indices = list()
    begin = -1
    end = -1
    direction = 0
    for i in range(0, len(diff)):
        if not diff[i] == 0:
            if begin > -1 and direction*diff[i] > 0:
                end = i
                if i == len(diff)-1:
                    indices.append(list([begin, end, direction]))
            else:
                if begin > -1 and direction*diff[i] < 0:
                    indices.append(list([begin, end+1, direction]))
                if diff[i] > 0:
                    direction = 1
                else:
                    direction = -1
                begin = i
                end = i
        else:
            if end > -1:
                if abs(x[i] - x[end]) > max_space or i == len(diff)-1:
                    indices.append(list([begin, end+1, direction]))
                    begin = -1
                    end = -1
                    direction = 0

    # second loop to group sames variations together
    good_index = list()
    end = 0
    for i in indices:
        if end < i[0]:
            if len(good_index) > 0 and good_index[len(good_index)-1][2] == 0:
                good_index[len(good_index)-1][1] = i[0]
            else:
                good_index.append(list([end, i[0], 0]))
            end = i[0]
        if abs(x[i[1]] - x[i[0]]) > min_length \
                or abs(y[i[1]] - y[i[0]]) > min_height:
            good_index.append(i)
            end = i[1]

    # Write small plane zone which should have been eliminate beacause of its
    # small length
    if end < indices[len(indices)-1][1]:
        if len(good_index) > 0 and good_index[len(good_index)-1][2] == 0:
            good_index[len(good_index)-1][1] = indices[len(indices)-1][1]
        else:
            good_index.append(list([end, indices[len(indices)-1][0], 0]))
        end = indices[len(indices)-1][1]

    # Write last plane zone if exists
    if indices[len(indices)-1][1] < len(diff)-1:
        if len(good_index) > 0 and good_index[len(good_index)-1][2] == 0:
            good_index[len(good_index)-1][1] = len(diff)-1
        else:
            good_index.append(list([indices[len(indices)-1][1],
                                    len(diff)-1, 0]))
    return good_index


def differential_separate(x, y, indices):
    ''' Go deeper in derivates datas analyse to find different fast of
    increase and decrease in order to detect increases and decreases even
    on inclined stem '''
    new_index = list()
    total_means = list()
    for ind in indices:
        direction = ind[2]
        if not direction == 0:
            tab = list([list(ind)])
            while tab[0][1] - tab[0][0] > 10:
                temp = list(tab)
                tab = list()
                for elem in temp:
                    longueur = int(round((elem[1] - elem[0])/2))
                    tab.append(list([elem[0], elem[0]+longueur, direction]))
                    tab.append(list([elem[0]+longueur, elem[1], direction]))
            means = list()
            for elem in tab:
                if x[elem[1]] - x[elem[0]] > 0:
                    means.append(float(y[elem[1]] - y[elem[0]]) /
                                 float(x[elem[1]] - x[elem[0]]))
                else:
                    means.append(np.sign(float(y[elem[1]] - y[elem[0]]))*np.Inf)
            loop_again = 1
            while loop_again:
                loop_again = 0
                i = 0
                while 1:
                    if i+1 < len(means):
                        if abs(means[i]) == np.inf \
                                or abs(means[i] - means[i+1]) < 0.2:
                            tab[i][1] = tab[i+1][1]
                            if x[tab[i][1]] - x[tab[i][0]] > 0:
                                means[i] = float(y[tab[i][1]] - y[tab[i][0]]) /\
                                           float(x[tab[i][1]] - x[tab[i][0]])
                            else:
                                means[i] = np.sign(float(y[tab[i][1]] -
                                                         y[tab[i][0]]))*np.Inf
                            tab.pop(i+1)
                            means.pop(i+1)
                            loop_again = 1
                        elif abs(means[i+1]) == np.inf:
                            if i+2 < len(means):
                                if abs(means[i+2]) > abs(means[i]):
                                    i += 1
                                else:
                                    tab[i][1] = tab[i+1][1]
                                    if x[tab[i][1]] - x[tab[i][0]] > 0:
                                        means[i] = float(y[tab[i][1]] - y[tab[i][0]])/float(x[tab[i][1]] - x[tab[i][0]])
                                    else:
                                        means[i] = np.sign(float(y[tab[i][1]] - y[tab[i][0]]))*np.Inf
                                    tab.pop(i+1)
                                    means.pop(i+1)
                                    loop_again = 1
                            else:
                                tab[i][1] = tab[i+1][1]
                                if x[tab[i][1]] - x[tab[i][0]] > 0:
                                    means[i] = float(y[tab[i][1]] - y[tab[i][0]])/float(x[tab[i][1]] - x[tab[i][0]])
                                else:
                                    means[i] = np.sign(float(y[tab[i][1]] - y[tab[i][0]]))*np.Inf
                                tab.pop(i+1)
                                means.pop(i+1)
                                loop_again = 1
                        else:
                            i += 1
                    else:
                        break
            for i in range(len(tab)):
                new_index.append(tab[i])
                total_means.append(means[i])
        else:
            if ind[1] - ind[0] < 4 and len(new_index):
                new_index[len(new_index) - 1][1] = ind[1]
            else:
                new_index.append(ind)
                total_means.append(0)

    return new_index, total_means


def majors_axes_regression_ww(pixels):
    """ Performs a major axis regression on 2D distributed dots

    :param pixels: (np array of 2 np array of int) distributed dots to perform
    regression
    :return: a: (float) slope of regression line
             b: (float) intercept of regression line
             mean_error: (float) mean error of dots to regression line
    """
    values = np.transpose(np.array([pixels[0], pixels[1]]))
    mean_values = np.mean(values, 0)

    s_xy = ((values[:, 0]-mean_values[0]) *
            (values[:, 1]-mean_values[1])).sum()
    s_xx = np.power(values[:, 0]-mean_values[0], 2).sum()
    s_yy = np.power(values[:, 1]-mean_values[1], 2).sum()

    if s_xy > 0:
        a = m.sqrt(s_yy/s_xx)
    else:
        a = -m.sqrt(s_yy/s_xx)

    b = mean_values[1] - a*mean_values[0]

    errors = np.array(abs(values[:, 1] - a * values[:, 0] - b))

    mean_error = np.mean(errors)

    return a, b, mean_error


def majors_axes_regression_line(binary_img):
    """ Performs a major axis regression on binary image as distributed dots

    :param binary_img: (numpy 2D binary uint8 array) binary image to perform
    regression
    :return: result: (numpy 3D uint8 array) color image with regression line
    draws on it
             a: (float) slope of regression line
             b: (float) intercept of regression line
             mean_error: (float) mean error of pixels to regression line
             alpha: angle of regression line (in degrees)
    """
    result = np.zeros([binary_img.shape[0], binary_img.shape[1], 3], 'uint8')
    result[:, :, 0] = np.array(binary_img)
    result[:, :, 1] = np.array(binary_img)
    result[:, :, 2] = np.array(binary_img)

    pixels = np.where(binary_img > 0)
    n = len(pixels[0])
    if n:
        a, b, errors_means = majors_axes_regression_ww(pixels)
        alpha = (m.atan2(a/m.sqrt(m.pow(a, 2)+1),
                         1/m.sqrt(m.pow(a, 2)+1)))*180/m.pi
        cv2.line(result, (int(b + a*pixels[0][0]), pixels[0][0]),
                 (int(b + a*pixels[0][n-1]), pixels[0][n-1]), (0, 0, 255), 2)
        return result, a, b, errors_means, alpha


def robust_majors_axes_regression_ww(pixels):
    """ Performs a robust (hinich et al.) major axis regression on 2D
    distributed dots

    :param pixels: (np array of 2 np array of int) distributed dots to perform
    regression
    :return: a: (float) slope of robust regression line
             b: (float) intercept of robust regression line
             useful_pixels: (np array of 2 np array of int) dots kept by robust
      regression
             useless_pixels: (np array of 2 np array of int) dots ousted by
      robust regression
    """
    n = len(pixels[0])
    a = b = 0
    useless_pixels = np.empty([0, 2], 'int')

    useful_pixels = np.transpose(np.array([pixels[0], pixels[1]]))

    values = useful_pixels[np.random.randint(n, size=int(n/2)), :]

    loop_again = 1
    while loop_again:
        mean_values = np.mean(values, 0)

        s_xy = ((values[:, 0]-mean_values[0]) *
                (values[:, 1]-mean_values[1])).sum()
        s_xx = np.power(values[:, 0]-mean_values[0], 2).sum()
        s_yy = np.power(values[:, 1]-mean_values[1], 2).sum()

        if s_xy > 0:
            a = m.sqrt(s_yy/s_xx)
        else:
            a = - m.sqrt(s_yy/s_xx)

        b = mean_values[1] - a*mean_values[0]

        errors = np.array(abs(useful_pixels[:, 1] - a*useful_pixels[:, 0] - b))
        sorted_errors = np.sort(errors)

        u28 = sorted_errors[int(round(28*n/100))]
        u72 = sorted_errors[int(round(72*n/100))]

        s = (u72 - u28)/1.654

        loop_again = 0
        pixels_to_delete = np.where(errors > 4*s)[0]
        if pixels_to_delete.shape[0]:
            useless_pixels = np.append(useless_pixels,
                                       useful_pixels[pixels_to_delete, :],
                                       axis=0)
            useful_pixels = np.delete(useful_pixels, pixels_to_delete, axis=0)
            loop_again = 1
        values = np.array(useful_pixels)
        n = values.shape[0]

    return a, b, useful_pixels, useless_pixels


def get_view_angles(binary_img, mask):
    """ This function analyse top view binary image to get
    :param binary_img: (numpy array of uint8) representing binary image
    :param mask: (numpy array of uint8) mask representing  the center of
    image to know if a leave can be considered as obstructing
    :return:
        (list of int) informative angles of view to analyse
        (numpy array of uint8) result image for log
        (string) log to write
    """
    result = np.zeros([binary_img.shape[0], binary_img.shape[1], 3],
                      'uint8')
    pixels = np.where(binary_img > 0)
    n = len(pixels[0])
    exclusions = list()
    if n > 1000:
        a, b, useful_pixels, useless_pixels = \
            robust_majors_axes_regression_ww(pixels)
        alpha = (m.atan2(a/m.sqrt(m.pow(a, 2) + 1),
                         1/m.sqrt(m.pow(a, 2) + 1)))*180/m.pi
        a90 = -1/a
        alpha90 = ((m.atan2(a90/m.sqrt(m.pow(a90, 2) + 1),
                            1/m.sqrt(m.pow(a90, 2) + 1)))*180/m.pi) % 360
        alpha270 = (alpha90 + 180) % 360

        result[useful_pixels[:, 0], useful_pixels[:, 1], :] = (255, 255, 255)
        cv2.line(result, (int(b+a*pixels[0][0]), pixels[0][0]),
                 (int(b+a*pixels[0][n-1]), pixels[0][n-1]), (0, 0, 255), 3)
        cv2.line(result, (int(b+a*pixels[0][0]), pixels[0][0]+2),
                 (int(b+a*pixels[0][n-1]), pixels[0][n-1]+1), (0, 0, 255), 3)
        cv2.line(result, (int(b+a*pixels[0][0]), pixels[0][0]-2),
                 (int(b+a*pixels[0][n-1]), pixels[0][n-1]-1), (0, 0, 255), 3)

        loop_again = 1
        while loop_again:
            loop_again = 0
            temp_img = np.zeros(binary_img.shape, 'uint8')
            temp_img[useless_pixels[:, 0], useless_pixels[:, 1]] = 255
            useless_pixels = np.empty([0, 2], 'int')
            labelled_img = measure.label(temp_img, neighbors=8)
            for region in measure.regionprops(labelled_img):
                pixels2 = np.where(labelled_img == region['label'])
                temp_useful_pixels = \
                    np.transpose(np.array([pixels2[0], pixels2[1]]))
                n2 = region.area
                if n2 > n/20:
                    a2, b2, useful_pixels2, useless_pixels2 = \
                        robust_majors_axes_regression_ww(pixels2)

                    alpha2 = (m.atan2(a2/m.sqrt(m.pow(a2, 2) + 1),
                                      1/m.sqrt(m.pow(a2, 2) + 1)))*180/m.pi
                    errors = np.array(abs(useful_pixels2[:, 1] -
                                           a * useful_pixels2[:, 0] - b))
                    x_intersection_line = int((b - b2)/(a2 - a))
                    y_intersection_line = int(a*x_intersection_line + b)
                    useless_pixels = np.append(useless_pixels, useless_pixels2, axis=0)

                    if 0 <= x_intersection_line < mask.shape[0] and \
                                    0 <= y_intersection_line < mask.shape[1]:
                        if abs(alpha-alpha2) > 20 and mask[x_intersection_line, y_intersection_line] and \
                                        errors.max() > 300:
                            max_error_pos = np.where(errors == errors.max())[0][0]
                            max_signed_error = useful_pixels2[max_error_pos,1] - a * useful_pixels2[max_error_pos,0] - b
                            diff = alpha - alpha2
                            if diff*max_signed_error < 0:
                                alpha2 = (alpha2 + 180) % 360
                            else:
                                alpha2 %= 360
                            exclusions.append(alpha2)
                            result[useful_pixels2[:, 0],
                                   useful_pixels2[:, 1], :] = (0, 255, 0)
                            cv2.line(result,
                                     (int(b2+a2*pixels2[0][0]), pixels2[0][0]),
                                     (int(b2+a2*pixels2[0][n2-1]), pixels2[0][n2-1]),
                                     (255, 0, 255), 2)
                            cv2.line(result,
                                     (int(b2+a2*pixels2[0][0]), pixels2[0][0]+1),
                                     (int(b2+a2*pixels2[0][n2-1]), pixels2[0][n2-1]+1),
                                     (255, 0, 255), 2)
                            cv2.line(result,
                                     (int(b2+a2*pixels2[0][0]), pixels2[0][0]-1),
                                     (int(b2+a2*pixels2[0][n2-1]), pixels2[0][n2-1]-1),
                                     (255, 0, 255), 2)
                        else:
                            result[temp_useful_pixels[:, 0],
                                   temp_useful_pixels[:, 1], :] = (0, 0, 255)
                    else:
                        result[temp_useful_pixels[:, 0],
                               temp_useful_pixels[:, 1], :] = (0, 0, 255)
                    loop_again = 1
                else:
                    result[temp_useful_pixels[:, 0],
                           temp_useful_pixels[:, 1], :] = (0, 0, 255)

        return result[::-1, ::-1], alpha90, alpha270, exclusions
    else:
        return result, -1, -1, exclusions


def robust_mean(values, images, std_error=20):
    """ This function perform a 'vote' between few values to extract the most
    representative(s) and the corresponding images

    :param values: (2 dimensional numpy float array) the vote will be perform on
    first value of each 2 values array
    :param images: (numpy array of string) id of image corresponding to each
    value
    :param std_error: (int) maximum standard error to reckon that 2 values are
    in the same group
    :return: means: (2 values numpy array) mean value of kept 2 values array
      ((-1, -1) if standard error remains more than std_error param)
             values: (2 dimensional numpy float array) kept values as most
      representatives
             images: (numpy array of string) id of image corresponding to each
      kept value
    """
    means = 0
    loop_again = 1
    while loop_again:
        loop_again = 0
        means = np.mean(values, 0)
        std_deviation = m.sqrt(np.power(values[:, 0]-means[0], 2).sum()/values.shape[0])
        if std_deviation > std_error:
            loop_again = 1
            errors = abs(values[:, 0] - means[0])
            if len(np.unique(errors)) == 1:
                means = np.array([-1, -1])
                images = images[np.unique(values[:, 0], return_index=True)[1]]
                values = values[np.unique(values[:, 0],
                                          return_index=True)[1], :]
                loop_again = 0
            else:
                values_to_delete = np.where(errors == errors.max())[0]
                values = np.delete(values, values_to_delete, 0)
                images = np.delete(images, values_to_delete, 0)
                if values.shape[0] <= 1:
                    means = np.array([-1, -1])
                    loop_again = 0
        else:
            images = images[np.unique(values[:, 0], return_index=True)[1]]
            values = values[np.unique(values[:, 0], return_index=True)[1], :]
    return means, values, images


def ear_detection(distances):
    """

    :param distances: (list of int) representing distance transform values all
    along the stem
    :return: (list of list of 2 int) first value of each 2 int list is a
             probable solution, second value is its weight
            (list of (list of (2 int and one list))) representing parts of
            distances interpreted as stem (begin, end, [values])
            (list of (list of (2 int and one list))) representing parts of
            distances interpreted as leaves (begin, end, [values])
            (list of 2 int), width of stem under ear and upper ear
    """
    distances_length = float(len(distances))
    part_1 = int(round(len(distances)/2.5))
    td = distances[:part_1]
    td.sort()
    mini = td[int(round(len(td)*15/100))]
    pos_min = np.where(td == mini)[0][0]

    # Look for peaks
    dist_array = np.array(distances)
    sorted_distances = list(distances)
    sorted_distances.sort()

    median = sorted_distances[int(round(part_1))]
    peak_begin = 0
    peaks = np.empty([0, 3], 'int')
    i = 1
    while i < len(distances)-1:
        if distances[i] > median:
            if not peak_begin:
                peak_begin = i
            if distances[i] > distances[i+1] and distances[i] > distances[i-1]:
                peaks = np.append(peaks, [[i, peak_begin, i]], axis=0)
            elif distances[i] > distances[i-1] and \
                            distances[i] == distances[i+1]:
                while distances[i] == distances[i+1]:
                    i += 1
                    if i >= len(distances)-1:
                        break
                if (i >= len(distances)-1) or (distances[i] > distances[i+1]):
                    peaks = np.append(peaks, [[i, peak_begin, i]], axis=0)
        elif peak_begin:
            if peaks.shape[0]:
                peaks[peaks.shape[0]-1, 2] = i
            peak_begin = 0

        i += 1
    if (i < len(distances)) and (distances[i] > distances[i-1] and
                                         distances[i] > median):
        peaks = np.append(peaks, [[i, peak_begin, i]], axis=0)

    i = 0
    while i < peaks.shape[0]-1:
        if (dist_array[peaks[i, 0]:peaks[i+1 ,0]] > median).all() or \
                        (dist_array[peaks[i, 0]:peaks[i+1,0]] <= median).sum() < 10:
            peaks[i+1,1] = peaks[i,1]
            peaks = np.delete(peaks, i, axis=0)
        else:
            i += 1
    peaks = peaks[np.where(peaks[:,0] >= part_1)[0],:]

    # Look for hollows representative of stem
    stems = list()

    route_distances = list()
    for i in range(len(distances)):
        route_distances.append([i, distances[i]])
    dist_diff, dist_x, dist_y = derivate(route_distances)
    dist_array = np.array(dist_y)
    dist_indexes = differential_cleaning(dist_diff, dist_x, dist_y, 10, 5, 2)
    dist_new_indexes, dist_total_means = differential_separate(dist_x, dist_y,
                                                               dist_indexes)
    for ind in dist_new_indexes:
        if ind[0] > part_1 and ind[1] < len(distances) and ind[1] - ind[0] > 20:
            if dist_array[ind[0]:ind[1]].min() <= mini and \
                            dist_array[ind[0]:ind[1]].max() <= median and \
                            ind[2] == 0:
                stems.append([ind[0], ind[1],
                              np.mean(dist_array[ind[0]:ind[1]])])

    # Group hollows
    i = 0
    while i < len(stems) - 1:
        j = 0
        while j < len(peaks) and stems[i][0] > peaks[j, 0]:
            j += 1
        j -= 1
        if (j == peaks.shape[0] - 1) or (stems[i][0] > peaks[j, 0] and
                                            stems[i + 1][1] < peaks[j + 1, 0]):
            if abs(stems[i][2] - stems[i + 1][2]) < 3:
                stems[i][1] = stems[i + 1][1]
                stems.pop(i + 1)
            else:
                if stems[i][2] > stems[i + 1][2]:
                    stems.pop(i)
                else:
                    stems.pop(i + 1)
        else:
            i += 1

    # Save previous peak of each hollow and weighting them from criteria
    # detailed in method
    td = distances[part_1:]
    td.sort()
    # keep only percentile at 15% on stem width to eliminate possible noise
    superior_min = td[int(round(len(td)*15/100))]

    solutions = np.empty([0, 2], 'int')
    stem_pos_after_ear = 0
    best_solution = 0
    first_found = False
    iteration = 0
    while not first_found and iteration < 2:
        for stem in stems:
            comparison = np.where(peaks[:, 0] < stem[0])[0]
            if len(comparison):
                pic = peaks[comparison[len(comparison)-1]]
                solutions = np.append(solutions, [[pic[0], 0]], axis=0)
                if not first_found and \
                                dist_array[stem[0]:stem[1]].mean() < mini:
                        for dist in dist_array[stem[0]:stem[1]]:
                            if superior_min <= dist < mini:
                                first_found = True
                                solutions[len(solutions)-1, 1] += 2
                                break
                if 8 < float(pic[2] - pic[1])*100./distances_length < 30:
                    solutions[len(solutions)-1, 1] += 1
                if solutions[len(solutions)-1, 1] > best_solution:
                    stem_pos_after_ear = stem[0]

        # If no solution found with percentile at 15%, redo calculation with
        # percentile at 5% to force a result
        if not first_found:
            solutions = np.empty([0, 2], 'int')
            superior_min = td[int(round(len(td)*5/100))]
        iteration += 1

    return solutions, stems, peaks, [pos_min, stem_pos_after_ear]
