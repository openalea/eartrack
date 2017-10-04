# coding: utf-8

import numpy
import cv2


def dilate(binary_image, kshape='MORPH_CROSS', ksize=3, iterations=1):
    """
    Dilate an image

    Dilate an image using opencv dilate method
    :param binary_image: numpy.ndarray
        2-D array
    :param kshape: str, opt
        See opencv documentation
    :param ksize: int, opt
        See opencv documentation
    :param iterations: int, opt
        Number of iteration of dilatation
    :return: dilated : numpy.ndarray 2-D image
    """

    kshape = getattr(cv2, kshape)
    element = cv2.getStructuringElement(kshape, (ksize, ksize))
    dilated = cv2.dilate(binary_image, element, iterations=iterations)
    return dilated


def open(binary_image, kshape='MORPH_CROSS', ksize=3, iterations=1):
    """
    Open an image

    Perform morphology opening algorithm on image using opencv method
    :param binary_image: numpy.ndarray
        2-D array
    :param kshape: str, opt
        See opencv documentation
    :param ksize: int, opt
        See opencv documentation
    :param iterations: int, opt
        Number of iteration
    :return: opened : numpy.ndarray 2-D image
    """
    kshape = getattr(cv2, kshape)
    element = cv2.getStructuringElement(kshape, (ksize, ksize))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, element,
                              iterations=iterations)
    return opened


def close(binary_image, kshape='MORPH_CROSS', ksize=3, iterations=1):
    """
    Close an image

    Perform morphology closing algorithm on image using opencv method
    :param binary_image: numpy.ndarray
        2-D array
    :param kshape: str, opt
        See opencv documentation
    :param ksize: int, opt
        See opencv documentation
    :param iterations: int, opt
        Number of iteration
    :return: closed : numpy.ndarray 2-D image
    """
    kshape = getattr(cv2, kshape)
    element = cv2.getStructuringElement(kshape, (ksize, ksize))
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, element,
                              iterations=iterations)
    return closed


def erode_dilate(binary_image,
                 kernel_shape=(3, 3),
                 iterations=1,
                 mask=None):
    """
    Applied a morphology (erode & dilate) on binary_image on mask ROI.

    Parameters
    ----------
    binary_image : numpy.ndarray
        2-D array

    kernel_shape: (N, M) of integers, optional
        kernel shape of (erode & dilate) applied to binary_image

    iterations: int, optional
        number of successive iteration of (erode & dilate)

    mask : numpy.ndarray, optional
        Array of same shape as `image`. Only points at which mask == True
        will be processed.

    Returns
    -------
    out : numpy.ndarray
        Binary Image
    """
    # ==========================================================================
    # Check Parameters
    if not isinstance(binary_image, numpy.ndarray):
        raise TypeError('binary_image must be a numpy.ndarray')

    if binary_image.ndim != 2:
        raise ValueError('binary_image must be 2D array')

    if mask is not None:
        if not isinstance(mask, numpy.ndarray):
            raise TypeError('mask must be a numpy.ndarray')
        if mask.ndim != 2:
            raise ValueError('mask must be 2D array')
    # ==========================================================================

    if mask is not None:
        out = cv2.bitwise_and(binary_image, mask)
    else:
        out = binary_image.copy()

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_shape)
    out = cv2.erode(out, element, iterations=iterations)
    out = cv2.dilate(out, element, iterations=iterations)

    if mask is not None:
        res = cv2.subtract(binary_image, mask)
        out = cv2.add(res, out)

    return out


def threshold_hsv(image, hsv_min, hsv_max, mask=None):
    """
    Binarize HSV image with hsv_min and hsv_max parameters.
    => cv2.inRange(hsv_image, hsv_min, hsv_max)

    If mask is not None :
    => cv2.bitwise_and(binary_hsv_image, mask)

    Parameters
    ----------
    image : numpy.ndarray of integers
        3-D array of image RGB

    hsv_min : tuple of integers
        HSV value of minimum range

    hsv_max : tuple of integers
        HSV value of maximum range

    mask : numpy.ndarray, optional
        Array of same shape as `image`. Only points at which mask == True
        will be thresholded.

    Returns
    -------
    out : numpy.ndarray
        Thresholded binary image

    See Also
    --------
    threshold_meanshift
    """
    # ==========================================================================
    # Check Parameters
    if not isinstance(image, numpy.ndarray):
        raise TypeError('image should be a numpy.ndarray')
    if image.ndim != 3:
        raise ValueError('image should be 3D array')

    if not isinstance(hsv_min, tuple):
        raise TypeError('hsv_min should be a Tuple')
    if len(hsv_min) != 3:
        raise ValueError('hsv_min should be of size 3')
    for value in hsv_min:
        if not isinstance(value, int):
            raise ValueError('hsv_min value should be a integer')

    if not isinstance(hsv_max, tuple):
        raise TypeError('hsv_max should be a Tuple')
    if len(hsv_max) != 3:
        raise ValueError('hsv_max should be of size 3')
    for value in hsv_max:
        if not isinstance(value, int):
            raise ValueError('hsv_max value should be a integer')

    if mask is not None:
        if not isinstance(mask, numpy.ndarray):
            raise TypeError('mask should be a numpy.ndarray')
        if mask.ndim != 2:
            raise ValueError('mask should be 2D array')
        if image.shape[0:2] != mask.shape:
            raise ValueError('image and mask should have the same shape')
    # ==========================================================================

    out = cv2.inRange(image, hsv_min, hsv_max)

    if mask is not None:
        out = cv2.bitwise_and(out, mask)

    return out


def threshold_meanshift(image, mean_image,
                        threshold=0.3,
                        mask=None):
    """
    Threshold pixels in numpy array such as::

        image / mean <= (1.0 - threshold)

    If reverse is True (Inequality is reversed)::

        image / mean <= (1.0 + threshold

    Parameters
    ----------
    image : numpy.ndarray of integers
        3-D array

    mean_image : numpy.ndarray of the same shape as 'image'
        3-D array 'mean_image'

    threshold : float, optional
        Threshold value. Must between 0.0 and 1.0

    reverse : bool, optional
       If True reverse inequality

    mask : numpy.ndarray, optional
        Array of same shape as `image`. Only points at which mask == True
        will be thresholded.

    Returns
    -------
    out : numpy.ndarray
        Thresholded binary image

    See Also
    --------
    get_mean_image, threshold_hsv

    """
    # ==========================================================================
    # Check Parameters
    if not isinstance(image, numpy.ndarray):
        raise TypeError('image should be a numpy.ndarray')
    if not isinstance(mean_image, numpy.ndarray):
        raise TypeError('mean should be a numpy.ndarray')

    if image.ndim != 3:
        raise ValueError('image should be 3D array')
    if mean_image.ndim != 3:
        raise ValueError('mean should be 3D array')
    if image.shape != mean_image.shape:
        raise ValueError('image and mean must have equal sizes')
    if not (0.0 <= threshold <= 1.0):
        raise ValueError('threshold must be between 0.0 and 1.0')

    if mask is not None:
        if not isinstance(mask, numpy.ndarray):
            raise TypeError('mask should be a numpy.ndarray')
        if mask.ndim != 2:
            raise ValueError('mask should be 2D array')
        if image.shape[0:2] != mask.shape:
            raise ValueError('mask and image must have equal sizes')
    # ==========================================================================

    with numpy.errstate(divide='ignore', invalid='ignore'):
        img = numpy.divide(numpy.float32(image), numpy.float32(mean_image))
        img[~ numpy.isfinite(img)] = 0

    # Take min value of RGB tuple
    img = img.min(2)
    out = img <= (1. - threshold)

    out = numpy.uint8(out)

    if mask is not None:
        out = cv2.bitwise_and(out, mask)

    del img

    return out * 255


def mean_shift_hsv(image, mean_img,
                   threshold=0.3,
                   hsv_min=(30, 11, 0),
                   hsv_max=(129, 254, 141),
                   iterations_clean_noise=3,
                   iterations=1,
                   mask_mean_shift=None,
                   mask_hsv=None,
                   mask_clean_noise=None):
    """
    Segmentation using mean shift method

    Compute segmentation of an object in image using a combination of
    meanshift method and hsv threshold

    :param image: numpy.ndarray of integers
        3-D array
    :param mean_img: numpy.ndarray of integers (same shape as 'image')
        3-D array
    :param threshold: float, optional
        Threshold value. Must between 0.0 and 1.0
    :param hsv_min: tuple of 3 int, optional
        Minimum values to threshold hsv image. Values must be between 0 and 255
    :param hsv_max: tuple of 3 int, optional
        Maximum values to threshold hsv image. Values must be between 0 and 255
    :param iterations_clean_noise: int, optional
        Number of iterations to clean noise on binary result image under mask
    :param iterations: int, optional
        Number of iterations to clean noise on binary result image
    :param mask_mean_shift: numpy.ndarray, optional
        Array 2-D of same shape as `image`. Only points at which mask == True
        will be calculated in meanshift method.
    :param mask_hsv: numpy.ndarray, optional
        Array 2-D of same shape as `image`. Only points at which mask == True
        will be calculated with hsv method.
    :param mask_clean_noise: numpy.ndarray, optional
        Array 2-D of same shape as `image`. Only points at which mask == True
        will be cleaned
    :return:
        result: numpy.ndarray 2-D of same shape as `image`
                Binary image representing plant segmentation of 'image'
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary_hsv_image = threshold_hsv(hsv_image, hsv_min, hsv_max, mask_hsv)

    binary_mean_shift_image = threshold_meanshift(
        image, mean_img, threshold, mask_mean_shift)

    result = cv2.add(binary_hsv_image, binary_mean_shift_image)

    if mask_clean_noise is not None and iterations_clean_noise > 0:
        result = erode_dilate(result, iterations=iterations_clean_noise,
                              mask=mask_clean_noise)

    if iterations > 0:
        result = erode_dilate(result, iterations=iterations)

    return result


def mean_image(images):
    """
    Compute the mean of a image list.

    Parameters
    ----------
    images : [ numpy.ndarray of integers ]
        list of 3-D array

    Returns
    -------
    out : numpy.ndarray
         Mean of the list image

    See Also
    --------
    threshold_meanshift
    """
    # ==========================================================================
    # Check Parameters
    if not isinstance(images, list):
        raise TypeError('images is not a list')
    if not images:
        raise ValueError('images is empty')

    shape_image_ref = None
    for image in images:
        if not isinstance(image, numpy.ndarray):
            raise TypeError('image in list images is not a ndarray')

        if shape_image_ref is None:
            shape_image_ref = numpy.shape(image)
        elif numpy.shape(image) != shape_image_ref:
            raise ValueError('Shape of ndarray image in list is different')
    # ==========================================================================

    length = len(images)
    weight = 1. / length

    start = cv2.addWeighted(images[0], weight, images[1], weight, 0)

    return reduce(lambda x, y: cv2.addWeighted(x, 1, y, weight, 0),
                  images[2:],
                  start)


def color_tree(bgr, cabin=None, mask_pot=None, mask_rails=None, empty_img=None):
    """
    Segmentation using decision tree and mask

    Platform specific method, masks and decision trees depend on imagery cabin
    :param bgr: numpy.ndarray of integers
        3-D array
    :param cabin: string, 2 possible values : cabin-1 or cabin-2
    :param mask_pot: mask_mean_shift: numpy.ndarray, optional
        Array 2-D of same shape as `bgr` representing pot position on image
    :param mask_rails: mask_mean_shift: numpy.ndarray, optional
        Array 2-D of same shape as `bgr` representing rails position
    :param empty_img: numpy.ndarray of integers
        3-D array of empty cabin (without plant)
    :return:
        result : numpy.ndarray 2-D of same shape as `bgr`
                 Binary image representing plant segmentation of 'bgr'
    """
    if cabin == "cabin-1":
        image_bin = decision_tree_threshold_phenoarch_1(bgr)
    elif cabin == "cabin-2":
        image_bin = decision_tree_threshold_phenoarch_2(bgr)
    else:
        image_bin = numpy.zeros(bgr.shape[0:2], 'uint8')

    if mask_pot is None:
        mask_pot = numpy.zeros(bgr.shape[0:2], 'uint8')

    if mask_rails is None:
        mask_rails = numpy.zeros(bgr.shape[0:2], 'uint8')

    # Using reference image corrects error out of the mask (pot and rails)
    if empty_img is not None:
        # Getting mask pot and rails "hand-made"
        mask = numpy.bitwise_or(mask_pot, mask_rails)

        # Making an extended mask to correct possible human error
        mask_extend = dilate(mask, iterations=3)

        # Calculating threshold on the extended mask only
        # We keep the extended mask because human error delete pixel in diff
        image_bin_threshold_pot = numpy.bitwise_and(image_bin, mask_extend)

        # Calculating diff between reference image and image with plant
        image_bin_diff = mean_shift_hsv(bgr, empty_img,
                                        mask_hsv=numpy.zeros(bgr.shape[0: 2], 'uint8'))

        # Out of the mask, keeping only pixels in both diff and threshold
        image_bin_diff = numpy.bitwise_and(
            numpy.bitwise_and(image_bin_diff, image_bin),
            numpy.bitwise_not(mask))

        result = numpy.add(image_bin_threshold_pot, image_bin_diff)
    else:
        result = image_bin

    return open(result, iterations=3)


# TODO auto-generate these 2 functions from decisions trees description
def decision_tree_threshold_phenoarch_1(bgr):
    """
    Implementation of a decision tree

    Platform specific method, for top image in cabin 1 of Phenoarch
    :param bgr: numpy.ndarray of integers
        3-D array
    :return:
        result : numpy.ndarray 2-D of same shape as `bgr`
                 Binary image representing True or False value of each pixel
                 threw decision tree
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    luv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LUV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    xyz = cv2.cvtColor(bgr, cv2.COLOR_BGR2XYZ)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

    image_bin_seuil = numpy.uint8(
        numpy.bitwise_or(
            numpy.bitwise_and(lab[:, :, 1] >= 120.5,
                              numpy.bitwise_or( \
                                  numpy.bitwise_and(lab[:, :, 2] < 139.5, \
                                                    numpy.bitwise_or( \
                                                        numpy.bitwise_and(
                                                            lab[:, :,
                                                            1] >= 122.5, \
                                                            numpy.bitwise_and(
                                                                lab[:, :,
                                                                1] < 123.5, \
                                                                numpy.bitwise_and(
                                                                    bgr[:,
                                                                    :,
                                                                    0] < 91.5, \
                                                                    numpy.bitwise_and(
                                                                        hsv[
                                                                        :,
                                                                        :,
                                                                        1] >= 28.5, \
                                                                        numpy.bitwise_and(
                                                                            yuv[
                                                                            :,
                                                                            :,
                                                                            0] >= 52.5, \
                                                                            numpy.bitwise_or(
                                                                                luv[
                                                                                :,
                                                                                :,
                                                                                1] < 94.5, \
                                                                                numpy.bitwise_and(
                                                                                    luv[
                                                                                    :,
                                                                                    :,
                                                                                    1] >= 94.5,
                                                                                    bgr[
                                                                                    :,
                                                                                    :,
                                                                                    2] >= 82.5) \
                                                                                ) \
                                                                            ) \
                                                                        ) \
                                                                    ) \
                                                                ) \
                                                            ),
                                                        numpy.bitwise_and(
                                                            lab[:, :,
                                                            1] < 122.5, \
                                                            numpy.bitwise_or(
                                                                xyz[:, :,
                                                                2] < 103.5, \
                                                                numpy.bitwise_and(
                                                                    xyz[:,
                                                                    :,
                                                                    2] >= 103.5, \
                                                                    numpy.bitwise_and(
                                                                        xyz[
                                                                        :,
                                                                        :,
                                                                        2] < 114.5,
                                                                        lab[
                                                                        :,
                                                                        :,
                                                                        1] < 121.5) \
                                                                    ) \
                                                                ) \
                                                            ) \
                                                        ) \
                                                    ), \
                                  numpy.bitwise_and(lab[:, :, 2] >= 139.5, \
                                                    numpy.bitwise_or( \
                                                        numpy.bitwise_and(
                                                            hsv[:, :,
                                                            1] < 55.5, \
                                                            numpy.bitwise_and(
                                                                bgr[:, :,
                                                                0] < 143.5,
                                                                bgr[:, :,
                                                                2] < 110.5) \
                                                            ), \
                                                        numpy.bitwise_and(
                                                            hsv[:, :,
                                                            1] >= 55.5, \
                                                            numpy.bitwise_and(
                                                                xyz[:, :,
                                                                2] >= 56.5, \
                                                                numpy.bitwise_or( \
                                                                    numpy.bitwise_and(
                                                                        hsv[
                                                                        :,
                                                                        :,
                                                                        1] < 69.5, \
                                                                        numpy.bitwise_or(
                                                                            hsv[
                                                                            :,
                                                                            :,
                                                                            0] >= 32.5, \
                                                                            numpy.bitwise_and(
                                                                                hsv[
                                                                                :,
                                                                                :,
                                                                                0] < 32.5, \
                                                                                numpy.bitwise_and(
                                                                                    hsv[
                                                                                    :,
                                                                                    :,
                                                                                    1] >= 61.5, \
                                                                                    numpy.bitwise_or(
                                                                                        numpy.bitwise_and(
                                                                                            hsv[
                                                                                            :,
                                                                                            :,
                                                                                            0] <= 20.5,
                                                                                            xyz[
                                                                                            :,
                                                                                            :,
                                                                                            2] < 138.5), \
                                                                                        numpy.bitwise_and(
                                                                                            hsv[
                                                                                            :,
                                                                                            :,
                                                                                            0] < 20.5, \
                                                                                            numpy.bitwise_or(
                                                                                                lab[
                                                                                                :,
                                                                                                :,
                                                                                                1] < 121.5, \
                                                                                                numpy.bitwise_and(
                                                                                                    lab[
                                                                                                    :,
                                                                                                    :,
                                                                                                    1] >= 121.5, \
                                                                                                    numpy.bitwise_or(
                                                                                                        luv[
                                                                                                        :,
                                                                                                        :,
                                                                                                        1] < 97.5, \
                                                                                                        numpy.bitwise_and(
                                                                                                            luv[
                                                                                                            :,
                                                                                                            :,
                                                                                                            1] >= 97.5, \
                                                                                                            numpy.bitwise_and(
                                                                                                                yuv[
                                                                                                                :,
                                                                                                                :,
                                                                                                                1] >= 134.5,
                                                                                                                yuv[
                                                                                                                :,
                                                                                                                :,
                                                                                                                1] < 137.5) \
                                                                                                            ) \
                                                                                                        ) \
                                                                                                    ) \
                                                                                                ) \
                                                                                            ) \
                                                                                        ) \
                                                                                    ) \
                                                                                ) \
                                                                            ) \
                                                                        ), \
                                                                    numpy.bitwise_and(
                                                                        hsv[
                                                                        :,
                                                                        :,
                                                                        1] >= 69.5, \
                                                                        numpy.bitwise_or( \
                                                                            numpy.bitwise_and(
                                                                                bgr[
                                                                                :,
                                                                                :,
                                                                                1] < 84.5, \
                                                                                numpy.bitwise_or(
                                                                                    yuv[
                                                                                    :,
                                                                                    :,
                                                                                    1] < 129.5,
                                                                                    yuv[
                                                                                    :,
                                                                                    :,
                                                                                    1] >= 135.5) \
                                                                                ), \
                                                                            numpy.bitwise_and(
                                                                                bgr[
                                                                                :,
                                                                                :,
                                                                                1] >= 84.5, \
                                                                                numpy.bitwise_or( \
                                                                                    numpy.bitwise_and(
                                                                                        hsv[
                                                                                        :,
                                                                                        :,
                                                                                        1] < 85.5,
                                                                                        yuv[
                                                                                        :,
                                                                                        :,
                                                                                        1] < 143.5), \
                                                                                    numpy.bitwise_and(
                                                                                        hsv[
                                                                                        :,
                                                                                        :,
                                                                                        1] >= 85.5,
                                                                                        lab[
                                                                                        :,
                                                                                        :,
                                                                                        1] < 151.5) \
                                                                                    ) \
                                                                                ) \
                                                                            ) \
                                                                        ) \
                                                                    ) \
                                                                ) \
                                                            ) \
                                                        ) \
                                                    ) \
                                  ) \
                              ), \
            numpy.bitwise_and(lab[:, :, 1] < 120.5, \
                              numpy.bitwise_or(bgr[:, :, 0] < 127.5, \
                                               numpy.bitwise_and(
                                                   bgr[:, :, 0] >= 127.5, \
                                                   numpy.bitwise_and(
                                                       hsv[:, :, 1] >= 49.5,
                                                       yuv[:, :, 1] < 205.5) \
                                                   ) \
                                               ) \
                              ) \
            ) * 255)
    return image_bin_seuil


def decision_tree_threshold_phenoarch_2(bgr):
    """
    Implementation of a decision tree

    Platform specific method, for top image in cabin 1 of Phenoarch
    :param bgr: numpy.ndarray of integers
        3-D array
    :return:
        result : numpy.ndarray 2-D of same shape as `bgr`
                 Binary image representing True or False value of each pixel
                 threw decision tree
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    luv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LUV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    xyz = cv2.cvtColor(bgr, cv2.COLOR_BGR2XYZ)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

    image_bin_seuil = numpy.uint8( \
        numpy.bitwise_or( \
            numpy.bitwise_and(lab[:, :, 1] >= 121.5, \
                              numpy.bitwise_or( \
                                  numpy.bitwise_and(lab[:, :, 2] < 146.5, \
                                                    numpy.bitwise_or( \
                                                        numpy.bitwise_and(
                                                            lab[:, :,
                                                            1] >= 122.5, \
                                                            numpy.bitwise_or( \
                                                                numpy.bitwise_and(
                                                                    luv[:,
                                                                    :,
                                                                    1] >= 94.5, \
                                                                    numpy.bitwise_and(
                                                                        hsv[
                                                                        :,
                                                                        :,
                                                                        1] >= 38.5, \
                                                                        numpy.bitwise_or( \
                                                                            numpy.bitwise_and(
                                                                                lab[
                                                                                :,
                                                                                :,
                                                                                1] >= 124.5, \
                                                                                numpy.bitwise_and(
                                                                                    luv[
                                                                                    :,
                                                                                    :,
                                                                                    2] >= 143.5, \
                                                                                    numpy.bitwise_and(
                                                                                        hsv[
                                                                                        :,
                                                                                        :,
                                                                                        1] >= 57.5, \
                                                                                        numpy.bitwise_and(
                                                                                            hsv[
                                                                                            :,
                                                                                            :,
                                                                                            0] >= 22.5,
                                                                                            yuv[
                                                                                            :,
                                                                                            :,
                                                                                            2] < 116.5) \
                                                                                        ) \
                                                                                    ) \
                                                                                ), \
                                                                            numpy.bitwise_and(
                                                                                lab[
                                                                                :,
                                                                                :,
                                                                                1] < 124.5, \
                                                                                numpy.bitwise_and(
                                                                                    bgr[
                                                                                    :,
                                                                                    :,
                                                                                    0] < 119.5, \
                                                                                    numpy.bitwise_or( \
                                                                                        numpy.bitwise_and(
                                                                                            hsv[
                                                                                            :,
                                                                                            :,
                                                                                            1] < 47.5, \
                                                                                            numpy.bitwise_and(
                                                                                                lab[
                                                                                                :,
                                                                                                :,
                                                                                                1] < 123.5,
                                                                                                bgr[
                                                                                                :,
                                                                                                :,
                                                                                                0] < 103.5) \
                                                                                            ), \
                                                                                        numpy.bitwise_and(
                                                                                            hsv[
                                                                                            :,
                                                                                            :,
                                                                                            1] >= 47.5,
                                                                                            bgr[
                                                                                            :,
                                                                                            :,
                                                                                            1] >= 56.5) \
                                                                                        ) \
                                                                                    ) \
                                                                                ) \
                                                                            ) \
                                                                        ) \
                                                                    ), \
                                                                numpy.bitwise_and(
                                                                    luv[:,
                                                                    :,
                                                                    1] < 94.5, \
                                                                    numpy.bitwise_and(
                                                                        bgr[
                                                                        :,
                                                                        :,
                                                                        0] < 110.5, \
                                                                        numpy.bitwise_and(
                                                                            lab[
                                                                            :,
                                                                            :,
                                                                            1] < 123.5, \
                                                                            numpy.bitwise_or(
                                                                                numpy.bitwise_and(
                                                                                    bgr[
                                                                                    :,
                                                                                    :,
                                                                                    0] < 63.5,
                                                                                    bgr[
                                                                                    :,
                                                                                    :,
                                                                                    1] >= 56.5), \
                                                                                numpy.bitwise_and(
                                                                                    bgr[
                                                                                    :,
                                                                                    :,
                                                                                    0] >= 63.5, \
                                                                                    numpy.bitwise_and(
                                                                                        xyz[
                                                                                        :,
                                                                                        :,
                                                                                        0] < 96.5, \
                                                                                        numpy.bitwise_or(
                                                                                            luv[
                                                                                            :,
                                                                                            :,
                                                                                            2] >= 143.5, \
                                                                                            numpy.bitwise_and(
                                                                                                luv[
                                                                                                :,
                                                                                                :,
                                                                                                2] < 143.5, \
                                                                                                numpy.bitwise_and(
                                                                                                    luv[
                                                                                                    :,
                                                                                                    :,
                                                                                                    1] < 93.5,
                                                                                                    bgr[
                                                                                                    :,
                                                                                                    :,
                                                                                                    0] < 93.5) \
                                                                                                ) \
                                                                                            ) \
                                                                                        ) \
                                                                                    ) \
                                                                                ) \
                                                                            ) \
                                                                        ) \
                                                                    ) \
                                                                ) \
                                                            ), \
                                                        numpy.bitwise_and(
                                                            lab[:, :,
                                                            1] < 122.5, \
                                                            numpy.bitwise_or( \
                                                                numpy.bitwise_and(
                                                                    bgr[:,
                                                                    :,
                                                                    0] >= 112.5, \
                                                                    numpy.bitwise_and(
                                                                        hsv[
                                                                        :,
                                                                        :,
                                                                        1] >= 48.5,
                                                                        bgr[
                                                                        :,
                                                                        :,
                                                                        0] < 130.5) \
                                                                    ), \
                                                                numpy.bitwise_or(
                                                                    bgr[:,
                                                                    :,
                                                                    0] < 98.5, \
                                                                    numpy.bitwise_and(
                                                                        bgr[
                                                                        :,
                                                                        :,
                                                                        0] < 112.5, \
                                                                        numpy.bitwise_or(
                                                                            hsv[
                                                                            :,
                                                                            :,
                                                                            1] >= 38.5, \
                                                                            numpy.bitwise_and(
                                                                                hsv[
                                                                                :,
                                                                                :,
                                                                                1] < 38.5, \
                                                                                numpy.bitwise_and(
                                                                                    bgr[
                                                                                    :,
                                                                                    :,
                                                                                    0] < 105.5,
                                                                                    hsv[
                                                                                    :,
                                                                                    :,
                                                                                    0] < 73.5) \
                                                                                ) \
                                                                            ) \
                                                                        ) \
                                                                    ) \
                                                                ) \
                                                            ) \
                                                        ) \
                                                    ), \
                                  numpy.bitwise_and(lab[:, :, 2] >= 146.5,
                                                    bgr[:, :, 0] < 161.5) \
                                  ) \
                              ), \
            numpy.bitwise_and(lab[:, :, 1] < 121.5, \
                              numpy.bitwise_or( \
                                  numpy.bitwise_and(bgr[:, :, 0] >= 126.5, \
                                                    numpy.bitwise_and(
                                                        hsv[:, :,
                                                        1] >= 49.5, \
                                                        numpy.bitwise_or( \
                                                            numpy.bitwise_and(
                                                                hsv[:, :,
                                                                1] < 56.5, \
                                                                numpy.bitwise_and(
                                                                    bgr[:,
                                                                    :,
                                                                    0] < 165.5, \
                                                                    numpy.bitwise_or(
                                                                        luv[
                                                                        :,
                                                                        :,
                                                                        2] >= 163.5, \
                                                                        numpy.bitwise_and(
                                                                            luv[
                                                                            :,
                                                                            :,
                                                                            2] < 163.5,
                                                                            xyz[
                                                                            :,
                                                                            :,
                                                                            0] >= 157.5) \
                                                                        ) \
                                                                    ) \
                                                                ), \
                                                            numpy.bitwise_and(
                                                                hsv[:, :,
                                                                1] >= 56.5,
                                                                bgr[:, :,
                                                                0] < 169.5) \
                                                            ) \
                                                        ) \
                                                    ), \
                                  numpy.bitwise_or(bgr[:, :, 0] < 108.5, \
                                                   numpy.bitwise_and(
                                                       bgr[:, :, 0] < 126.5, \
                                                       numpy.bitwise_and(
                                                           bgr[:, :,
                                                           0] >= 108.5, \
                                                           numpy.bitwise_or(
                                                               hsv[:, :,
                                                               1] >= 53.5, \
                                                               numpy.bitwise_and(
                                                                   hsv[:, :,
                                                                   1] < 53.5, \
                                                                   numpy.bitwise_and(
                                                                       luv[
                                                                       :, :,
                                                                       2] >= 146.5, \
                                                                       numpy.bitwise_or(
                                                                           bgr[
                                                                           :,
                                                                           :,
                                                                           0] < 116.5, \
                                                                           numpy.bitwise_and(
                                                                               bgr[
                                                                               :,
                                                                               :,
                                                                               0] >= 116.5,
                                                                               hsv[
                                                                               :,
                                                                               :,
                                                                               1] >= 35.5) \
                                                                           ) \
                                                                       ) \
                                                                   ) \
                                                               ) \
                                                           ) \
                                                       ) \
                                                   ) \
                                  ) \
                              ) \
            ) * 255)
    return image_bin_seuil
