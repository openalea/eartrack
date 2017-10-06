# coding: utf-8
""" Specific module for notebook detailed example of eartrack

"""

import math as m

import numpy as np
import cv2
from skimage import measure

import openalea.eartrack.eartrack as et_lib


def main_top_robust_regression(binary_img):
    """ Perform robust regression on binary image

    Part of openalea.eartrack.eartrack.get_view_angles, which return all
    parameters needed for further step

    :param binary_img: (numpy array 2-D of uint8) representing binary image
    :return:
        robust_img: (numpy array 3-D of uint8) color image representing results
            of robust regression
        (a, b, alpha): (list of floats) results of robust regression, a = slope,
            b = intercept, alpha = angle
        (alpha90, alpha270): (list of floats) alpha's perpendicular angles
        useless_pixels: (list of coordinates) pixels exclude by robustness
    """
    robust_img = np.zeros([binary_img.shape[0], binary_img.shape[1], 3],
                          'uint8')
    pixels = np.where(binary_img > 0)
    n = len(pixels[0])
    a, b, useful_pixels, useless_pixels = et_lib.robust_majors_axes_regression_ww(
        pixels)
    alpha = (m.atan2(a / m.sqrt(m.pow(a, 2) + 1),
                     1 / m.sqrt(m.pow(a, 2) + 1))) * 180 / m.pi

    a90 = -1 / a
    alpha90 = ((m.atan2(a90 / m.sqrt(m.pow(a90, 2) + 1),
                        1 / m.sqrt(m.pow(a90, 2) + 1))) * 180 / m.pi) % 360
    alpha270 = (alpha90 + 180) % 360

    robust_img[useful_pixels[:, 0], useful_pixels[:, 1], :] = (255, 255, 255)
    robust_img[useless_pixels[:, 0], useless_pixels[:, 1], :] = (0, 0, 255)
    cv2.line(robust_img, (int(b + a * pixels[0][0]), pixels[0][0]),
             (int(b + a * pixels[0][n - 1]), pixels[0][n - 1]), (0, 0, 255), 3)
    cv2.line(robust_img, (int(b + a * pixels[0][0]), pixels[0][0] + 2),
             (int(b + a * pixels[0][n - 1]), pixels[0][n - 1] + 1), (0, 0, 255),
             3)
    cv2.line(robust_img, (int(b + a * pixels[0][0]), pixels[0][0] - 2),
             (int(b + a * pixels[0][n - 1]), pixels[0][n - 1] - 1), (0, 0, 255),
             3)
    return robust_img, (a, b, alpha), (alpha90, alpha270), useless_pixels


def hamper_top_robust_regression(binary_img, mask, useless_pixels, robust_img,
                                 a, b, alpha):
    """ Perform robust regression on unused pixels of the result of a first
    regression

    Part of openalea.eartrack.eartrack.get_view_angles, which return all
    parameters needed for further steps

    :param binary_img: (numpy array 2-D of uint8) representing binary image
    :param mask: (numpy array 2-D of uint8) representing mask
    :param useless_pixels: (list of coordinates) pixels exclude by first
        robustness
    :param robust_img: (numpy array 3-D of uint8) color image representing
        results of first robust regression
    :param a: (float) slope of first robust regression
    :param b: (float) intercept of first robust regression
    :param alpha: (float) angle of line from first robust regression
    :return:
        exclusions: (list of float) angles of leaves that could hamper stem
            detection
        plot_img: (list of numpy array 3-D of uint8) intermediary images of
            hampering leaves detection
    """
    plot_img = dict()

    exclusions = list()
    unused_pixels = useless_pixels.copy()
    hampering_img = robust_img.copy()
    pixels = np.where(binary_img > 0)
    n = len(pixels[0])
    loop_again = True
    found_leaves = 0
    while loop_again:
        loop_again = False
        temp_img = np.zeros(binary_img.shape, 'uint8')
        temp_img[useless_pixels[:, 0], useless_pixels[:, 1]] = 255
        useless_pixels = np.empty([0, 2], 'int')
        labelled_img = measure.label(temp_img, neighbors=8)
        for region in measure.regionprops(labelled_img):
            pixels_tmp = np.where(labelled_img == region['label'])
            used_pixels = np.transpose(np.array([pixels_tmp[0], pixels_tmp[1]]))
            n_tmp = region.area
            if n_tmp > n / 20:
                a_tmp, b_tmp, useful_pixels_tmp, useless_pixels_tmp = \
                    et_lib.robust_majors_axes_regression_ww(pixels_tmp)
                alpha_tmp = (m.atan2(a_tmp / m.sqrt(m.pow(a_tmp, 2) + 1),
                                     1 / m.sqrt(
                                         m.pow(a_tmp, 2) + 1))) * 180 / m.pi
                errors = np.array(abs(
                    useful_pixels_tmp[:, 1] - a * useful_pixels_tmp[:, 0] - b))
                x_intersection_line = int((b - b_tmp) / (a_tmp - a))
                y_intersection_line = int(a * x_intersection_line + b)
                useless_pixels = np.append(useless_pixels, useless_pixels_tmp,
                                           axis=0)

                if 0 <= x_intersection_line < mask.shape[
                    0] and 0 <= y_intersection_line < mask.shape[1]:
                    if abs(alpha - alpha_tmp) > 20 and mask[
                        x_intersection_line, y_intersection_line] and errors.max() > 300:
                        max_error_pos = np.where(errors == errors.max())[0][0]
                        max_signed_error = \
                            useful_pixels_tmp[max_error_pos, 1] - \
                            a * useful_pixels_tmp[max_error_pos, 0] - b
                        diff = alpha - alpha_tmp
                        if diff * max_signed_error < 0:
                            alpha_tmp %= 360
                        else:
                            alpha_tmp = (alpha_tmp + 180) % 360
                        exclusions.append(alpha_tmp)
                        hampering_img[useful_pixels_tmp[:, 0],
                        useful_pixels_tmp[:, 1], :] = (0, 255, 0)
                        cv2.line(hampering_img,
                                 (int(b_tmp + a_tmp * pixels_tmp[0][0]),
                                  pixels_tmp[0][0]),
                                 (int(b_tmp + a_tmp * pixels_tmp[0][n_tmp - 1]),
                                  pixels_tmp[0][n_tmp - 1]), (255, 0, 255), 2)
                        cv2.line(hampering_img,
                                 (int(b_tmp + a_tmp * pixels_tmp[0][0]),
                                  pixels_tmp[0][0] + 1),
                                 (int(b_tmp + a_tmp * pixels_tmp[0][n_tmp - 1]),
                                  pixels_tmp[0][n_tmp - 1] + 1), (255, 0, 255),
                                 2)
                        cv2.line(hampering_img,
                                 (int(b_tmp + a_tmp * pixels_tmp[0][0]),
                                  pixels_tmp[0][0] - 1),
                                 (int(b_tmp + a_tmp * pixels_tmp[0][n_tmp - 1]),
                                  pixels_tmp[0][n_tmp - 1] - 1), (255, 0, 255),
                                 2)
                        found_leaves += 1
                        plot_img["Hampering leaf " + str(found_leaves)] = \
                            hampering_img.copy()
                    else:
                        hampering_img[used_pixels[:, 0], used_pixels[:, 1],
                        :] = (0, 0, 255)
                else:
                    hampering_img[used_pixels[:, 0], used_pixels[:, 1], :] = (
                    0, 0, 255)
                loop_again = True
            else:
                hampering_img[used_pixels[:, 0], used_pixels[:, 1], :] = (
                0, 0, 255)

    return exclusions, plot_img


def select_angles_to_keep(existing_angles, perp_alphas):
    """ Select angles to keep from existing ones

    :param existing_angles: (list of int) of all existing side view images
    :param perp_alphas: (list of 2 float) the 2 angles perpendicular to main
        leaves plane
    :return:
        angles_to_keep: (list of int) angles to keep from existing ones
    """
    angles_to_keep = list()
    for angle in perp_alphas:
        if angle > 345:
            angle -= 360
        for i in range(len(existing_angles)):
            if abs(existing_angles[i] - angle) <= 10:
                if i > 0:
                    angles_to_keep.append(existing_angles[i - 1])
                else:
                    angles_to_keep.append(
                        existing_angles[len(existing_angles) - 1])

                angles_to_keep.append(existing_angles[i])

                if i < len(existing_angles) - 1:
                    angles_to_keep.append(existing_angles[i + 1])
                else:
                    angles_to_keep.append(existing_angles[0])
                break
            elif abs(existing_angles[i] - angle) <= 15:
                angles_to_keep.append(existing_angles[i])
                if existing_angles[i] < angle:
                    if i < len(existing_angles) - 1:
                        angles_to_keep.append(existing_angles[i + 1])
                    else:
                        angles_to_keep.append(existing_angles[0])
                else:
                    if i > 0:
                        angles_to_keep.append(existing_angles[i - 1])
                    else:
                        angles_to_keep.append(
                            existing_angles[len(existing_angles) - 1])
    angles_to_keep.sort()
    return angles_to_keep


def exclude_angles(angles_to_keep, exclusions):
    """ Angles to exclude of side view analyse because of possible hampering
    leaf

    :param angles_to_keep: (list of float) list of angles to keep
    :param exclusions: (list of float) list of angles where leaf could hamper
        stem detection
    :return:
        excluded_angles: (list of int) side view images' angles to not analyse
            because of possible hampering leaf
    """
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
    return excluded_angles