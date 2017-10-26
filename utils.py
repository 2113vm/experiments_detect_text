from math import fabs
from itertools import permutations

import cv2
import numpy as np

from geometry import Contour


class ContoursContainer(object):

    def __init__(self, shape):
        self._table = np.empty(shape, dtype=object)

    def put(self, contour):
        cx, cy = contour.rect.center
        if self._table[cx][cy] is None:
            self._table[cx][cy] = [contour]
        else:
            self._table[cx][cy].append(contour)

    def get(self, (cx, cy)):
        ret = self._table[cx][cy]
        if ret is None:
            return []
        return ret

    def get_nearby_contours(self, contour, epsilon):
        cx, cy = contour.rect.center
        s = self._table[(cx - epsilon):(cx + epsilon), (cy - epsilon):(cy + epsilon)]
        return s[s.astype(np.bool)]

    def get_all_contours(self):
        ret = self._table[self._table.astype(np.bool)].sum()  # Returns False if empty and list of elements otherwise
        if not ret:
            return []
        else:
            return ret

    @property
    def size(self):
        return np.count_nonzero(self._table)


dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


def dtype_limits(image, clip_negative=True):
    """This is copied from skimage

    Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
    """
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


def unique_rows(a):
    """Returns array with removed duplicate rows
    """
    if not len(a):
        return a
    # a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def corner_kernel(size, radius, offset=0):
    """
    >>> corner_kernel(3, 0)
    array([[ 0.2,  0.2,  0.2],
           [ 0.2,  0. ,  0. ],
           [ 0.2,  0. ,  0. ]], dtype=float32)
    >>> corner_kernel(3, 1)
    array([[ 0.  ,  0.25,  0.25],
           [ 0.25,  0.  ,  0.  ],
           [ 0.25,  0.  ,  0.  ]], dtype=float32)
    >>> corner_kernel(3, 0, 1)
    array([[ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.33333334,  0.33333334],
           [ 0.        ,  0.33333334,  0.        ]], dtype=float32)
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[radius+offset:, offset] = 1
    kernel[offset, radius+offset:] = 1

    if radius:
        corner = np.fliplr(np.triu(np.ones((radius, radius), dtype=np.float32)))
        kernel[offset:offset+radius, offset:offset+radius] = corner / radius

    kernel /= np.sum(kernel)
    return kernel


def rotate_kernel(kernel, anchor, k):
    """
    Rotate array and anchor point (x, y) 90 degrees counter-clockwise

    >>> kernel = np.array([[1, 2, 3], [4, 5, 6]])
    >>> anchor = (2, 0)  # coordinates of 3
    >>> rotate_kernel(kernel, anchor, k=1)
    (array([[3, 6],
           [2, 5],
           [1, 4]]), (0, 0))
    >>> rotate_kernel(kernel, anchor, k=3)
    (array([[4, 1],
           [5, 2],
           [6, 3]]), (1, 2))
    """
    k %= 4
    if k == 0:
        return kernel, anchor
    sy, sx = kernel.shape[:2]
    ax, ay = anchor
    if k == 1:
        anchor_rotated = (ay, sx - 1 - ax)
    elif k == 2:
        anchor_rotated = (sx - 1 - ax, sy - 1 - ay)
    else:  # k == 3
        anchor_rotated = (sy - 1 - ay, ax)
    return np.rot90(kernel, k=k), anchor_rotated


def random_color_hsv(seed):
    """Returns random HSV color with large saturation and maximum brightness.
        Used for drawing of detected objects.
    """
    np.random.seed(seed)
    hue = np.random.randint(180)
    sat = np.random.randint(128, 256)
    return hue, sat, 255


def to_hsv_image(image):
    """Converts BGR or grayscale image to HSV colorspace.
    """
    if len(image.shape) == 2:
        h, v = image.shape
        dst = np.zeros((h, v, 3), dtype=np.uint8)
        dst[:, :, 2] = image
        return dst
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def rectangle(shape, dtype=None):
    ret = np.zeros(shape, dtype)
    ret[(0, -1), :] = 1
    ret[1:-1, (0, -1)] = 1
    return ret


def scharr(image, dx, dy):
    dst = cv2.Scharr(image, cv2.CV_32F, dx, dy, scale=1.0/16, borderType=cv2.BORDER_REPLICATE)
    dst = abs(dst)
    if len(image.shape) > 2:
        dst = np.max(dst, axis=2)
    return dst


def scharr_max(image):
    dx = scharr(image, 1, 0)
    dy = scharr(image, 0, 1)
    return cv2.max(dx, dy)


def get_max(dx, dy):
    return cv2.max(dx, dy)


def get_mean(dx, dy):
    return np.mean(np.array([dx, dy]), axis=0)


def get_rms(dx, dy):
    return np.sqrt(np.mean(np.array([cv2.pow(dx, 2), cv2.pow(dy, 2)]), axis=0))

DIFF_CUSTOM = -1
DIFF_SCHARR = 0
DIFF_SOBEL = 1
DIFF_ROBERTS = 2
DIFF_PREWITT = 3
DIFF_SOBEL_MOD = 4
DIFF_LAPLACE = 5

METRIC_MIN = -1
METRIC_MAX = 0
METRIC_MEAN = 1
METRIC_RMS = 2
METRIC_SPLIT = 3


def differentiate(image, method=DIFF_SOBEL, metric=METRIC_MAX, xkernel=None, ykernel=None, coeff=1):
    if method == DIFF_SCHARR:
        dx = cv2.Scharr(image, cv2.CV_32F, 1, 0, scale=1.0/16, borderType=cv2.BORDER_REPLICATE)
        dy = cv2.Scharr(image, cv2.CV_32F, 0, 1, scale=1.0/16, borderType=cv2.BORDER_REPLICATE)
    elif method == DIFF_SOBEL:
        # kernels with sizes other than 3 are not effective at all
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, scale=1.0/16, borderType=cv2.BORDER_REPLICATE)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, scale=1.0/16, borderType=cv2.BORDER_REPLICATE)
    elif method == DIFF_LAPLACE:
        return cv2.Laplacian(image, cv2.CV_32F)
    else:
        if method == DIFF_ROBERTS:
            xkernel = np.array(((1./16,     0),
                                (0, -1./16)), dtype=np.float32)
            ykernel = np.array(((0, 1./16),
                                (-1./16, 0)), dtype=np.float32)
        elif method == DIFF_PREWITT:
            xkernel = np.array(((-1./16, -1./16, -1./16),
                                (0, 0, 0),
                                (1./16, 1./16, 1./16)), dtype=np.float32)
            ykernel = np.array(((-1./16, 0, 1./16),
                                (-1./16, 0, 1./16),
                                (-1./16, 0, 1./16)), dtype=np.float32)
        elif method == DIFF_SOBEL_MOD:
            # coeff == 1 makes Prewitt kernel, coeff == 2 makes Sobel kernel, coeff == 10/3 makes Scharr kernel
            xkernel = np.array(((-1./16., -coeff/16., -1./16.),
                                (0, 0, 0),
                                (1./16., coeff/16., 1./16.)), dtype=np.float32)
            ykernel = np.array(((-1./16., 0, 1./16.),
                                (-coeff/16., 0, coeff/16.),
                                (-1./16., 0, 1./16.)), dtype=np.float32)
        else:
            assert(xkernel is not None)
            if ykernel is None:
                ykernel = np.transpose(xkernel)
        dx = cv2.filter2D(image, cv2.CV_32F, xkernel, borderType=cv2.BORDER_REPLICATE)
        dy = cv2.filter2D(image, cv2.CV_32F, ykernel, borderType=cv2.BORDER_REPLICATE)

    dx = abs(dx)
    dy = abs(dy)
    if len(image.shape) > 2:
        dx = np.max(dx, axis=2)
        dy = np.max(dy, axis=2)

    if metric == METRIC_MAX:
        return get_max(dx, dy)
    elif metric == METRIC_MEAN:
        return get_mean(dx, dy)
    elif metric == METRIC_RMS:
        return get_rms(dx, dy)
    elif metric == METRIC_SPLIT:
        return dx, dy
    else:
        return cv2.min(dx, dy)


def compare_moments(sample, moments):
    """Calculates distance between two collections of moments. Considers only keys contained in `sample`
    :type sample: dict
    :type moments: dict
    :rtype: float
    """
    # TODO: better metric
    return sum(abs(sample[k] - moments[k]) for k in sample)


def contour_mask(contour, thickness=-1):
    """
    Returns contour mask and origin
    :type contour: Contour
    :param thickness: set to -1 for filled mask
    :return: mask image and origin (x, y coordinates of top left corner)
    :rtype: ndarray
    """
    x, y, w, h = contour.rect
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour.raw], 0, 1, thickness=thickness, offset=(-x, -y))
    return mask


def get_roi_slice(contour, offset=None):
    rect = contour.rect
    if offset is not None:
        rect = rect.shift(*offset)
    return rect.slice()


def contour_roi(image, contour, offset=None, masking=True):
    """Returns region of `image` contained inside `contour`
    :type contour: Contour
    :param offset: offset by which every contour point is shifted
    """

    roi = image[get_roi_slice(contour, offset)]

    if not masking:
        return roi

    mask = contour_mask(contour)

    h, w = mask.shape
    assert roi.shape[:2] == (h, w)

    if len(roi.shape) == 3:
        mask = mask.reshape((h, w, 1))

    return roi * mask


def shrink_contour(contour, ksize=3):
    """
    :type contour: pvision.geometry.Contour
    """
    mask = contour.get_mask()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    cv2.erode(mask, kernel, dst=mask, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=contour.rect[:2])
    if not contours:
        return contour
    if len(contours) == 1:
        return Contour(contours[0])

    return Contour(max(contours, key=lambda c: cv2.arcLength(c, True)))


def to_drawable_contours(contours):
    """Returns an array of contours in correct format for drawContours method"""
    return [np.array(contour, dtype=np.int32) for contour in contours]


def cmp_rects(first, second):
    """Returns a metric between two rects represented as tulpe of x, y, width and height."""
    # Comparing centers of rectangles should be more accurate than comparing corners
    first_ = [first[0] + first[2]/2,
              first[1] + first[3]/2,
              first[2],
              first[3]]

    second_ = [second[0] + second[2]/2,
               second[1] + second[3]/2,
               second[2],
               second[3]]

    res = 0
    for idx in xrange(len(first)):
        res += fabs(first_[idx] - second_[idx])  # Let's try Manhattan as a metric for a while
    return res


def xor_compare(shape, contour1, contour2):
    """Returns a number of different pixels between contour1 and contour2 areas"""

    img1 = np.zeros(shape, np.uint8)
    cv2.drawContours(img1, [contour1.contour], -1, 255, cv2.cv.CV_FILLED)

    img2 = np.zeros(shape, np.uint8)
    cv2.drawContours(img2, [contour2.contour], -1, 255, cv2.cv.CV_FILLED)

    x = min(contour1.rect[0], contour2.rect[0])
    y = min(contour1.rect[1], contour2.rect[1])
    w = max(contour1.rect[2], contour2.rect[2])
    h = max(contour1.rect[3], contour2.rect[3])
    sym_diff = cv2.bitwise_xor(img1[y:y+h, x:x+w], img2[y:y+h, x:x+w])
    return cv2.countNonZero(sym_diff)


def compare_centers(centers1, centers2):
    """Return normalized distance between center1 and center2.
    Note: O(n!) may not be something you're looking for."""
    distances = []
    assert(len(centers1) == len(centers2))
    channels = 1
    for perm in permutations(range(len(centers1))):
        idx1 = 0
        for idx2 in perm:
            channels = min(len(centers1[idx1]), len(centers2[idx2]))
            res = 0
            for clr in xrange(channels):
                res += fabs(centers1[idx1][clr] - centers2[idx2][clr])
            idx1 += 1
            distances.append(res)
    return min(distances) / (channels * 255.)


def compare_colors(px1, px2):
    """Return normalized distance in colorspace between px1 and px2. Works faster than compare_centers"""
    channels = min(len(px1), len(px2))
    res = 0
    for idx in xrange(channels):
        res += fabs(px1[idx] - px2[idx])
    return res / (channels * 255.)


def detect_border_kmeans(image, contour, max_thickness=3, max_cent_diff=0.05):
    """Achtung! Deprecated due to its randomness, works with plain contour (not Contour object)"""
    # kmeans method parameters are encapsulated here, there's no need to keep them in method signature
    # utils.compare_centers has O(k!) complexity, so using k more than 2 is undesirable. Anyway, k = 2 seems to be
    # enough.
    k = 2
    maxiter = 10
    eps = 1.
    attempts = 4
    kernel = np.ones((3, 3), np.uint8)
    x, y, w, h = cv2.boundingRect(contour)
    #print x, y, w, h

    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, to_drawable_contours([contour]), 0, 255, cv2.cv.CV_FILLED, offset=(-x, -y))
    filled = cv2.erode(filled, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.bitwise_xor(filled, cv2.erode(filled, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    roi = image[y:y+h, x:x+w]

    old_centers = None

    for thickness in xrange(max_thickness + 1):
        bool_mask = mask.astype(np.bool)
        masked = roi[bool_mask]
        # cv2.imshow("mask", mask)
        # cv2.imshow("filled", filled)
        # cv2.imshow("masked", masked)
        if len(masked) < k:  # number of key points should be >= k
            break
        new_centers = cv2.kmeans(np.float32(masked), k,
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxiter, eps),
                                 attempts, cv2.KMEANS_RANDOM_CENTERS)[2]
        if old_centers is not None:
            diff = compare_centers(old_centers, new_centers)
            if diff > max_cent_diff:
                new_conts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE, offset=(x, y))
                length = len(new_conts)
                idx = 0
                if length == 0 or length > 2:  # It means that erosion fragmented contour
                    return contour
                elif length == 2:
                    #print hier
                    if hier[0][1][3] >= 0:  # Looking for outer contour
                        idx = 1
                    elif hier[0][0][3] < 0:  # If these two contours are independent
                        return contour
                return new_conts[idx]
        if thickness != max_thickness:
            old_centers = new_centers
            filled = cv2.erode(filled, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.bitwise_xor(filled, cv2.erode(filled, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    return contour


def draw_text(image, text, origin, color, scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):
    """Draws text in area with top left corner at given `origin`
    Returns width and height of text area.
    """
    x, y = origin
    width, height = cv2.getTextSize(text, font, scale, thickness)[0]
    cv2.putText(image, text, (x, y + height), font, scale, color, thickness)
    return width, height


def get_convex_hull(contour, orient=None):
    """Returns convex hull of given contour preserving its orientation"""
    if orient is None:
        orient = (cv2.contourArea(contour, oriented=True) < 0)
    return cv2.convexHull(contour, clockwise=orient)


#deprecated
def normalize_image(image, mask=None):
    if image.min() < 0:
        image = cv2.convertScaleAbs(image)
    if mask is not None:
        outer_mask = (1 - mask).astype(np.bool)
        image[outer_mask] = 0
    if image.max() > 0:
        image *= 255.0/image.max()
    # assert image.dtype == np.uint8
    return image


def rescale_if_necessary(image, mask, area_threshold):
    area = image.shape[0] * image.shape[1]
    if area <= area_threshold:
        return image, mask.astype(np.bool)
    else:
        scale = float(area_threshold)/area
        res_img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        res_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
        return res_img, res_mask.astype(np.bool)


def scale(image, factor):
    return cv2.resize(image, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_NEAREST)


def get_convex_image_mask(image):
    """Returns convex hull of image's nonzero pixels as binary image
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    ones_mask = np.ones(image.shape[:2], dtype=np.uint8)
    w_orig, h_orig = image.shape[:2]

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))[0]

    if contours:
        contour = contours[0]
        contour = cv2.convexHull(contour)

        cv2.drawContours(mask, [contour], 0, 1, -1)
        (_, _, w, h) = cv2.boundingRect(contour)

        if float(w*h)/float(w_orig*h_orig) < 0.8:  # TODO: hardcoded
            return ones_mask
    return mask


def draw_hist(im, mask=None, ch=0):
    hist_height = 64
    hist_width = 512
    nbins = 128
    bin_width = hist_width/nbins

    #Create an empty image for the histogram
    h = np.zeros((hist_height, hist_width))

    #Create array for the bins
    bins = np.arange(nbins, dtype=np.int32).reshape(nbins, 1)

    #Calculate and normalise the histogram
    hist_item = cv2.calcHist([im], [ch], mask, [nbins], [0, 256])
    cv2.normalize(hist_item, hist_item, hist_height, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    pts = np.column_stack((bins,hist))

    #Loop through each bin and plot the rectangle in white
    for x, y in enumerate(hist):
        cv2.rectangle(h, (x*bin_width, y), (x*bin_width + bin_width-1, hist_height), (255), -1)

    #Flip upside down
    h=np.flipud(h)

    #Show the histogram
    cv2.imshow('colorhist %i' %ch , h)


def fix_kernel_size(*kernel):
    x, y = kernel[:2]
    if x % 2 == 0:
        x += 1
    if y % 2 == 0:
        y += 1
    return x, y


# All methods below are deprecated. Use the new Filter class in vision.filters
def get_maxarea_filter(maxarea):
    return lambda c: c.area <= maxarea


def get_minarea_filter(minarea):
    return lambda c: c.area >= minarea


def get_minfill_filter(minfill):
    return lambda c: (c.area / c.rect.area) >= minfill


def get_smoothness_filter(thresh):
    return lambda c: (c.length / c.vertexes) >= thresh


def get_max_height_filter(max_height):
    return lambda c: c.rect.height <= max_height


def get_min_height_filter(min_height):
    return lambda c: c.rect.height >= min_height


def get_min_aspect_ratio_filter(min_ratio):
    return lambda c: (float(c.rect.width) / c.rect.height) >= min_ratio


def filter_contours(contours, filters):
    for cnt in contours:
        for flt in filters:
            if not flt(cnt):
                break
        else:
            yield cnt
