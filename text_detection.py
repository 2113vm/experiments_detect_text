import cv2
import numpy as np

import utils
from filters import Filter
from geometry import Contour, Rectangle

__author__ = 'Dmitry Nikonov <nikonovds@gmail.com>'

TEXT_BOXES = 0
TEXT_CONTOURS = 1
TEXT_CONTOURS_BOXES = 2


# def rotate_image(original, angle):
#     center = (original.shape[0] / 2, original.shape[1] / 2)


def find_text_lines(original, thresh=0.02, min_size=4, max_size=12,
                    spacing=1.8, min_length=1, otsu=False, x_only=False):
    def _threshold_normal(_image, _thresh):
        return np.uint8(cv2.threshold(_image,
                                      _thresh * utils.dtype_limits(_image)[1],
                                      255,
                                      cv2.THRESH_BINARY)[1])

    def _threshold_otsu(_image, *args):  # Careful with that axe, Eugene
        return cv2.threshold(np.uint8(_image / utils.dtype_limits(_image)[1] * 255.),
                             0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if otsu:
        get_thresh = _threshold_otsu
    else:
        get_thresh = _threshold_normal

    def _find_text_preprocess(_image):
        # _image = cv2.bilateralFilter(_image, -1, 64, 3)
        # if gauss:
        #     _image = cv2.medianBlur(_image, 3)
            # cv2.imshow('prep', _image)
        if len(_image.shape) > 2:
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        # image = cv2.equalizeHist(image)
        return _image

    def _get_diff(_image):
        # _image = cv2.medianBlur(_image, 5)
        # _image = cv2.GaussianBlur(_image, (3, 3), 5)
        _image = utils.differentiate(_image, metric=utils.METRIC_MAX)
        return utils.differentiate(_image, metric=utils.METRIC_MIN)


    if spacing != 0:
        ksize = int(min_size * spacing)
    else:
        ksize = 1

    filters = Filter()
    # filters.add_filter("outer")
    filters.add_filter("min_area", min_size * min_size * min_length * 12)
    # filters.add_filter("min_fill", 0.3)
    filters.add_filter("min_aspect_ratio", 0.1)
    # filters.add_filter("smoothness", 5)

    # if diffs is None:
    #     diffs = utils.differentiate(np.float32(_find_text_preprocess(original)) / 255.,
    #                                 xkernel=5, ykernel=5, metric=utils.METRIC_SPLIT)
    # if x_only:
    #     diff = utils.differentiate(np.float32(_find_text_preprocess(original)) / 255.,
    #  xkernel=5, ykernel=5, metric=utils.METRIC_SPLIT)[0]
    # else:
    #     diff = utils.differentiate(np.float32(_find_text_preprocess(original)) / 255.,
    # xkernel=5, ykernel=5)
    # diff = utils.differentiate(np.float32(_find_text_preprocess(original) / 255.),
    # metric=utils.METRIC_MIN)

    diff = _get_diff(_find_text_preprocess(original))
    ndiff = diff / diff.max()
    binary = get_thresh(ndiff, thresh)
    # cv2.imshow('diff', binary * 255)
    # Connecting
    connected = cv2.morphologyEx(binary,
                                 cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT,
                                                           utils.fix_kernel_size(ksize, ksize)),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # cv2.imshow('connected', connected)
    # Filtering
    binary = cv2.morphologyEx(connected,
                              cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT,
                                                        utils.fix_kernel_size(int(min_size) - 1,
                                                                              int(min_size) - 1)))
    # cv2.imshow('bin', binary)
    return filters.filter([Contour(c,
                                   thresh) for c in cv2.findContours(binary,
                                                                     cv2.RETR_EXTERNAL,
                                                                     cv2.CHAIN_APPROX_SIMPLE)[0]])


def find_text(original=None, diffs=None, first_thresh=0.02, second_thresh=0.04, min_size=4,
              max_size=12, hspacing=1.8, vspacing=0, min_length=1, offset=(0, 0), otsu=False,
              split=True, max_box_rad=(32, 32), default_box_rad=(2, 2), ret=TEXT_CONTOURS_BOXES):
    """
    Detects text areas on image regardless to its contents

    :param hspacing: Maximal horizontal spacing for symbols connection (relative to min_size)
    :param min_length: Minimal text length (relative to min_size)

    """
    _debug = False

    def _threshold_normal(_image, _thresh):
        return np.uint8(cv2.threshold(_image,
                                      _thresh * utils.dtype_limits(_image)[1],
                                      255,
                                      cv2.THRESH_BINARY)[1])

    def _threshold_otsu(_image, *args):  # Careful with that axe, Eugene
        return cv2.threshold(np.uint8(_image / utils.dtype_limits(_image)[1] * 255.),
                             0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if otsu:
        get_thresh = _threshold_otsu
    else:
        get_thresh = _threshold_normal

    def _find_text_preprocess(_image):
        # image = cv2.bilateralFilter(image, -1, 64, 3)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        # image = cv2.equalizeHist(image)
        return _image

    assert(original is not None or diffs is not None)

    if hspacing != 0:
        ksize_h = int(min_size * hspacing)
    else:
        ksize_h = 1

    if vspacing != 0:
        ksize_v = int(min_size * vspacing)
    else:
        ksize_v = 1

    filters = Filter()
    filters.add_filter("outer")
    filters.add_filter("min_area", min_size * min_size * min_length)
    # filters.add_filter("min_fill", 0.5)
    # filters.add_filter("min_aspect_ratio", min_length)

    # if diffs is None:
    #     diffs = utils.differentiate(np.float32(_find_text_preprocess(original)) / 255.,
    #                                 xkernel=5, ykernel=5, metric=utils.METRIC_SPLIT)
    if diffs is None:
        diffs = utils.differentiate(np.float32(_find_text_preprocess(original)) / 255.,
                                    xkernel=5, ykernel=5)
        diffs = (diffs, diffs)

    ndiffs = (diffs[0] / diffs[0].max(), diffs[1] / diffs[1].max())

    if _debug:
        cv2.imshow("raw bin", ndiffs[0] / ndiffs[0].max())

    binary = get_thresh(ndiffs[0], first_thresh)

    if _debug:
        cv2.imshow("bin", binary)

    # Remove too long vertical lines
    sub = cv2.morphologyEx(binary,
                           cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     utils.fix_kernel_size(1, max_size)))
    sub = cv2.morphologyEx(sub,
                           cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (1, 3)))
    if _debug:
        cv2.imshow("sub", sub)
    binary = cv2.bitwise_xor(binary, sub)

    # Connecting
    connected = cv2.morphologyEx(binary,
                                 cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT,
                                                           utils.fix_kernel_size(ksize_h, ksize_v)),
                                 borderType=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    if _debug:
        cv2.imshow("connected", connected)

    # Filtering
    binary = cv2.morphologyEx(connected,
                              cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT,
                                                        utils.fix_kernel_size(min_length * int(min_size) - 1,
                                                                                         int(min_size) - 1)))
    if _debug:
        cv2.imshow("connected and filtered", binary)

    # 1st detect_angle, not accurate, with max_thresh, not using offset yet
    conts = [Contour(c, first_thresh) for c in cv2.findContours(binary,
                                                                cv2.RETR_LIST,
                                                                cv2.CHAIN_APPROX_SIMPLE)[0]]
    filtered = filters.filter(conts)

    # Preparing data for floodFill
    if ret == TEXT_BOXES or ret == TEXT_CONTOURS_BOXES:
        second_bin = (get_thresh(ndiffs[0], second_thresh), get_thresh(ndiffs[1], second_thresh))
    else:
        second_bin = (get_thresh(ndiffs[0], second_thresh), None)

    bin_for_ff = cv2.bitwise_or(second_bin[0] * 1., (connected / 255) * 2)
    mask = cv2.bitwise_not(cv2.bitwise_or(binary,  connected))
    mask = np.pad(mask, 1, 'constant', constant_values=255)  # add 1px. border

    for c in filtered:
        cv2.floodFill(bin_for_ff, mask, (c.raw[0][0][0], c.raw[0][0][1]), 255, flags=4)
    bin_for_ff = np.uint8(cv2.threshold(bin_for_ff, 1, 255, cv2.THRESH_BINARY)[1])
    # bin_for_ff = cv2.morphologyEx(bin_for_ff, cv2.MORPH_OPEN,
    # cv2.getStructuringElement(cv2.MORPH_RECT,
    # utils.fix_kernel_size(min_length * int(min_size) - 1,
    # int(min_size) - 1)))

    if split:
        bin_for_split = bin_for_ff.copy()
    if _debug:
        cv2.imshow("bin_for_ff", bin_for_ff)
    # (offset[0] - 1) is a workaround for some kind of bug
    # 2nd detect_angle, more accurate, with min_thresh and offset
    conts = [Contour(c, first_thresh) for c in cv2.findContours(bin_for_ff,
                                                                cv2.RETR_LIST,
                                                                cv2.CHAIN_APPROX_SIMPLE,
                                                                offset=(offset[0] - 1,
                                                                        offset[1]))[0]]
    filtered = filters.filter(conts)

    if not split:
        splitted = filtered
    else:
        splitted = []
        for c in filtered:
            if c.rect.height >= 2 * min_size:
                splitted.extend(split_lines(bin_for_split,
                                            c,
                                            min_size,
                                            offset=(-offset[0], -offset[1])))
            else:
                splitted.append(c)

    if ret == TEXT_CONTOURS:
        return splitted
    else:
        boxes = [find_text_box(contour=c,
                               bdiff=second_bin,
                               max_rad=max_box_rad,
                               default_rad=default_box_rad,
                               offset=offset) for c in splitted]
        if ret == TEXT_BOXES:
            return boxes
        elif ret == TEXT_CONTOURS_BOXES:
            return zip(splitted, boxes)
        # else:
            # for c, b in zip(splitted, boxes):
            #     roi = b.get_roi(original, offset)
            #     return c.get_contour_roi(roi, offset)


def find_text_box(contour, image=None, bdiff=None, max_rad=(32, 32), default_rad=(2, 2),
                  raw=False, offset=(0, 0), thresh=0.015):
    """
    :param bdiff: tuple of differences along each axis
    :param max_rad: tuple of max values (relative to the bounding rect of `contour` of box radius
    by each axis
    :param default_rad: tuple of default radius values that will be used if algorithm will be
    unable to detect_angle a box
    :return: Rectangle object
    """
    assert(image is not None or bdiff is not None)
    if bdiff is None:
        diffs = utils.differentiate((cv2.cvtColor(image,
                                                  cv2.COLOR_BGR2GRAY).astype(np.float32)) / 255.,
                                    xkernel=5,
                                    ykernel=5,
                                    metric=utils.METRIC_SPLIT)

        bdiff = (np.uint8(cv2.threshold(diffs[0],
                                        thresh * utils.dtype_limits(diffs[0])[1],
                                        255,
                                        cv2.THRESH_BINARY)[1]),
                 np.uint8(cv2.threshold(diffs[1],
                                        thresh * utils.dtype_limits(diffs[0])[1],
                                        255,
                                        cv2.THRESH_BINARY)[1]))
    for_draw = [utils.shrink_contour(contour).raw]
    # for_draw = [contour.raw]
    cv2.drawContours(bdiff[0], for_draw, -1, 0, -1, offset=offset)
    cv2.drawContours(bdiff[1], for_draw, -1, 0, -1, offset=offset)
    # cv2.imshow(str(time.time()), bdiff[0])
    # cv2.imshow(str(time.time()), bdiff[1])

    max_left = max(contour.rect.left + 2 - offset[0], 0)
    min_left = max(contour.rect.left - max_rad[0] - offset[0], 0)

    min_right = contour.rect.right - offset[0] - 1
    max_right = contour.rect.right + max_rad[0] - offset[0] + 1

    max_top = max(contour.rect.top + 2 - offset[1], 0)
    min_top = max(contour.rect.top - max_rad[1] - offset[1], 0)

    min_bottom = contour.rect.bottom - 1 - offset[1]
    max_bottom = contour.rect.bottom + max_rad[1] + 1 - offset[1]

    left = max_left
    right = min_right
    top = max_top
    bottom = min_bottom

    # Move left border
    left_arr = (
        bdiff[0][top, min_left:max_left] | bdiff[0][bottom, min_left:max_left]
    ).nonzero()[0]

    if left_arr.shape[0] == 0:
        left = max(contour.rect.left + 1 - default_rad[0] - offset[0], 0)
    else:
        left = min_left + left_arr[left_arr.shape[0] - 1]

    # Move right border
    right_arr = (
        bdiff[0][top, min_right:max_right] | bdiff[0][bottom, min_right:max_right]
    ).nonzero()[0]

    if right_arr.shape[0] == 0:
        right = min(contour.rect.right - 1 + default_rad[0] - offset[0], bdiff[0].shape[1] - 1)
    else:
        right = min_right + right_arr[0]

    # Move top border
    top_arr = (bdiff[1][min_top:max_top, left] | bdiff[1][min_top:max_top, right]).nonzero()[0]
    if top_arr.shape[0] == 0:
        top = max(contour.rect.top + 1 - default_rad[1] - offset[1], 0)
    else:
        top = min_top + top_arr[top_arr.shape[0] - 1]

    # Move bottom border
    bottom_arr = (
        bdiff[1][min_bottom:max_bottom, left] | bdiff[1][min_bottom:max_bottom, right]
    ).nonzero()[0]

    if bottom_arr.shape[0] == 0:
        bottom = min(contour.rect.bottom - 1 + default_rad[1] - offset[1], bdiff[1].shape[0] - 1)
    else:
        bottom = min_bottom + bottom_arr[0]

    left += offset[0]
    right += offset[0]
    top += offset[1]
    bottom += offset[1]

    if raw:
        return left, top, right - left, bottom - top
    return Rectangle(left, top, right - left, bottom - top)


def split_lines(binary, contour, min_size, thresh=0.5, roi=False, offset=(0, 0)):
    if not roi:
        roi_slice = utils.get_roi_slice(contour, offset)
        binary = binary[roi_slice]
    # cv2.imshow("bin", binary)
    # ACHTUNG: may not work correct if `roi` is not correct contour's roi
    binary = binary & np.array(utils.contour_mask(contour), dtype=bool)

    means = binary.mean(axis=1)

    # THRESH_INV
    means[means < thresh] = -1
    means[means >= thresh] = 0
    means[means == -1] = 1

    # Find where gaps starts and ends
    means = np.pad(means, 1, 'constant')
    means = means - np.roll(means, 1)
    begins = (means == 1).nonzero()[0] - 1  # -1 caused by pad
    ends = (means == -1).nonzero()[0] - 1

    # Getting ars to cut
    args = (begins + ends) / 2
    args = args[args >= min_size]
    args = args[args <= means.shape[0] - min_size]
    if args.shape[0] == 0:
        return [contour]

    lines = []
    prev = 0
    args = np.append(args, means.shape[0] - 2)
    for arg in args:  # TODO: make this code reusable for fragmentation
        line_roi = binary[prev:arg, :]
        curr_off = (contour.rect.left, contour.rect.top + prev)
        conts = [Contour(c) for c in cv2.findContours(line_roi,
                                                      cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_SIMPLE,
                                                      offset=curr_off)[0]]
        lines.extend(conts)
        prev = arg

    flt = Filter()
    flt.add_filter("min_area", min_size*min_size)
    lines = flt.filter(lines)
    if len(lines) <= 1:
        return [contour]

    return lines
