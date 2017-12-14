from abc import ABCMeta, abstractmethod, abstractproperty
from math import fabs
import re
import cv2
import numpy as np


class Point(tuple):
    def __new__(cls, x, y):
        return super(Point, cls).__new__(cls, (x, y))

    def __init__(self, x, y):
        super(Point, self).__init__()

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    def __neg__(self):
        """
        :rtype: Point
        """
        return Point(-self[0], -self[1])

    def __add__(self, other):
        """
        :rtype: Point
        """
        assert len(other) == 2
        return Point(self[0] + other[0], self[1] + other[1])

    def __sub__(self, other):
        """
        :rtype: Point
        """
        assert len(other) == 2
        return -(-self + other)

    def __mul__(self, other):
        return Point(self[0] * other, self[1] * other)

    def __rsub__(self, other):
        return -self + other

    def __radd__(self, other):
        """
        :rtype: Point
        """
        return self + other


class Shape(object):
    """Abstract base class for shapes like Rectangle and Contour
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_bounding_rect(self):
        """
        :rtype: Rectangle
        """
        raise NotImplementedError

    @abstractmethod
    def as_contour(self):
        """
        :rtype: Contour
        """
        raise NotImplementedError

    @abstractproperty
    def area(self):
        raise NotImplementedError

    def get_roi(self, image, offset=None, scale_factor=None):
        """Returns shape ROI of given image

        :type image: ndarray
        :param Point offset: offset by which `image` is shifted
        :param float scale_factor: factor by which `image` is scaled
        :rtype: ndarray
        """
        rect = self.get_bounding_rect()

        if offset is not None:
            rect = rect.shift(*-offset)

        if scale_factor is not None:
            rect = rect.scale(scale_factor)

        return image[rect.slice()]

    @abstractmethod
    def get_mask(self, scale_factor=1):
        """Return shape mask

        :return: mask image with dimensions of `bounding_rect`
        :rtype: ndarray
        """
        raise NotImplementedError

    @staticmethod
    def _expand_and_offset(mask, dimensions, offset=None):
        """Creates new zero image with given dimensions and
        places mask on it, adding offset.

        :type mask: ndarray
        :type dimensions: tuple
        :type offset: tuple
        :rtype: ndarray
        """
        mh, mw = mask.shape
        w, h = dimensions

        if offset is None:
            ox, oy = 0, 0
        else:
            ox, oy = offset

        if (ox, oy, mw, mh) == (0, 0, w, h):
            return mask

        assert ox + mw <= w and oy + mh <= h

        ret = np.zeros((h, w), dtype=mask.dtype)
        ret[oy:oy + mh, ox:ox + mw] = mask
        return ret

    def _get_adjusted_masks(self, other):
        """
        :type other: Shape
        :return: (ndarray, ndarray)
        """
        rect = self.get_bounding_rect()
        other_rect = other.get_bounding_rect()

        container = rect.extend(other_rect)
        dims = container.width, container.height

        mask = self._expand_and_offset(self.get_mask(), dims,
                                       rect.origin - container.origin)
        other_mask = self._expand_and_offset(other.get_mask(), dims,
                                             other_rect.origin - container.origin)

        assert mask.shape == other_mask.shape

        return mask, other_mask

    def intersects(self, other, return_area=False):
        """
        :type other: Shape
        :rtype: bool or int
        """
        rect = self.get_bounding_rect()
        other_rect = other.get_bounding_rect()

        # make fast check first
        if not rect.intersects_rect(other_rect):
            return 0 if return_area else False

        intersection = np.logical_and(*self._get_adjusted_masks(other))
        if return_area:
            return np.count_nonzero(intersection)
        else:
            return np.any(intersection)

    def contains(self, other):
        """
        :type other: Shape
        :rtype: bool
        """
        rect = self.get_bounding_rect()
        other_rect = other.get_bounding_rect()

        # make fast check first
        if not rect.contains_rect(other_rect):
            return False

        return np.all(np.greater_equal(*self._get_adjusted_masks(other)))


class MaskedShape(Shape):
    """
    Shape specified by binary mask
    """
    def __init__(self, mask, origin):
        """
        :type mask: ndarray
        :type origin: tuple
        """
        assert mask.dtype == np.uint8 and len(mask.shape) == 2

        super(MaskedShape, self).__init__()

        self._mask = mask
        self._origin = Point(*origin)
        self._area = None
        self._rect = None
        self._contour = None

    @property
    def area(self):
        if self._area is None:
            self._area = np.count_nonzero(self._mask)
        return self._area

    def as_contour(self):
        if self._contour is None:
            contours = cv2.findContours(self._mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                        offset=self._origin)[1]
            hull = cv2.convexHull(np.concatenate(contours))
            try:
                self._contour = Contour(hull)
            except IndexError:
                raise Exception('Unable to get contour from mask')

        return self._contour

    def get_bounding_rect(self):
        if self._rect is None:
            x = self._mask.sum(axis=0).nonzero()[0]
            y = self._mask.sum(axis=1).nonzero()[0]
            try:
                rect = Rectangle.from_coordinates(x[0], y[0], x[-1], y[-1])
            except IndexError:
                rect = Rectangle(0, 0, 0, 0)

            self._rect = rect.shift(*self._origin)

        return self._rect

    def get_mask(self, scale_factor=1):
        rect = self.get_bounding_rect()
        if scale_factor != 1:
            rect = rect.scale(scale_factor)
        return rect.get_roi(self._mask, offset=self._origin)

        # def extend(self, other):
        #     """
        #     :type other: Shape
        #     :rtype: MaskedShape
        #     """
        #     return MaskedShape(np.maximum(self.get_mask(), other.get_mask))
        #
        # def get_scaled_shape(self, scale_factor):
        #     return MaskedShape(cv2.resize(self._mask, (0, 0), fx=scale_factor, fy=scale_factor))

        # def get_expanded_shape(self, pos, new_shape):
        #     x, y = pos
        #     w, h = self._mask.shape
        #     rect = Rectangle(x, y, h, w)
        #     new_mask = np.zeros(new_shape, dtype=np.uint8)
        #     new_mask[rect.slice()] = self._mask
        #     return MaskedShape(new_mask)


class Rectangle(tuple, Shape):
    ATTRIBUTES = ('left', 'right', 'top', 'bottom', 'width', 'height')

    def __new__(cls, left, top, width, height):
        return super(Rectangle, cls).__new__(cls, (left, top, width, height))

    def __init__(self, left, top, width, height):
        super(Rectangle, self).__init__()
        self._mask = None

    def __eq__(self, other):
        return (self.left == other.left and
                self.top == other.top and
                self.width == other.width and
                self.height == other.height)

    @classmethod
    def from_coordinates(cls, left, top, right, bottom):
        return cls(left, top, max(0, right - left), max(0, bottom - top))

    @property
    def left(self):
        return self[0]

    @property
    def top(self):
        return self[1]

    @property
    def width(self):
        return self[2]

    @property
    def height(self):
        return self[3]

    @property
    def right(self):
        return self[0] + self[2]

    @property
    def bottom(self):
        return self[1] + self[3]

    @property
    def area(self):
        return self[2] * self[3]

    @property
    def center(self):
        return self[0] + self[2] / 2, self[1] + self[3] / 2

    @property
    def shape(self):
        return self[3], self[2]

    @property
    def origin(self):
        """Top-left point of rectangle.
        """
        return Point(self[0], self[1])

    def contains_rect(self, other):
        """Returns whether given rectangle is contained inside this.
        """
        left, top, width, height = other
        return (left >= self.left and
                top >= self.top and
                left + width <= self.right and
                top + height <= self.bottom)

    def intersects_rect(self, other):
        """Returns whether given rectangle intersects with this.
        """
        left, top, width, height = other
        return (left < self.left + self.width and
                self.left < left + width and
                top < self.top + self.height and
                self.top < top + height)

    def extend(self, other):
        """Returns minimal rectangle containing both this and given rectangle.
        """
        left, top, width, height = other
        right = left + width
        bottom = top + height
        return Rectangle.from_coordinates(min(left, self.left),
                                          min(top, self.top),
                                          max(right, self.right),
                                          max(bottom, self.bottom))

    def intersect(self, other):
        left, top, width, height = other
        right = left + width
        bottom = top + height
        return Rectangle.from_coordinates(max(left, self.left),
                                          max(top, self.top),
                                          min(right, self.right),
                                          min(bottom, self.bottom))

    def shift(self, x, y):
        """Returns rectangle shifted by given values
        :rtype: Rectangle
        """
        return Rectangle(self.left + x, self.top + y, self.width, self.height)

    def scale(self, factor):
        return Rectangle(self.left - int(self.width * (factor - 1)) / 2,
                         self.top - int(self.height * (factor - 1)) / 2,
                         int(self.width * factor),
                         int(self.height * factor))
        # return Rectangle(self.left * factor, self.top * factor,
        #                  self.width * factor, self.height * factor)

    def to_square(self):
        if self.width == self.height:
            return self

        expansion = abs(self.width - self.height) / 2
        width_expansion = 0 if self.width > self.height else expansion
        height_expansion = 0 if self.height > self.width else expansion
        square_side = max(self.width, self.height)
        square = Rectangle(self.left - width_expansion,
                           self.top - height_expansion,
                           square_side, square_side)
        return square

    def __str__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, *self)

    def _convert_arg(self, param_name, value):
        if isinstance(value, (str, np.unicode)):
            ret = 0
            q = 1
            for component in re.split('([+-])', value):
                if not component or component == '+':
                    continue

                if component == '-':
                    q = -1
                    continue

                if component.endswith('%'):
                    if param_name in ('left', 'right', 'width'):
                        related_parameter = self.width
                    elif param_name in ('top', 'bottom', 'height'):
                        related_parameter = self.height
                    else:
                        raise TypeError(
                            "Unexpected keyword argument '{}'".format(
                                param_name))

                    percents = float(component[:-1]) / 100
                    # if param_name in ('height', 'width') and percents < 0:
                    #     percents = -percents + 1
                    num = related_parameter * percents
                else:
                    num = int(component)

                ret += q * num
                q = 1

            return int(ret)
        else:
            return value

    def get_relative_rectangle(self, **kwargs):
        kwargs = dict(
            (k, v) for k, v in kwargs.iteritems() if v)  # Remove all Nones
        for name in kwargs:
            kwargs[name] = self._convert_arg(name, kwargs[name])

        left = self.left + kwargs['left'] if 'left' in kwargs else None
        right = self.right - kwargs['right'] if 'right' in kwargs else None
        top = self.top + kwargs['top'] if 'top' in kwargs else None
        bottom = self.bottom - kwargs['bottom'] if 'bottom' in kwargs else None

        if left is None:
            left = right - kwargs['width']
        elif right is None:
            right = left + kwargs['width']
        if top is None:
            top = bottom - kwargs['height']
        elif bottom is None:
            bottom = top + kwargs['height']
        return Rectangle.from_coordinates(left, top, right, bottom)

    def expand(self, x, y):
        xr = str(x)
        xr = xr[1:] if xr.startswith('-') else '-' + xr
        yr = str(y)
        yr = yr[1:] if yr.startswith('-') else '-' + yr
        return self.get_relative_rectangle(left=xr, right=xr, top=yr,
                                           bottom=yr)

    def slice(self):
        """Used for convenient extraction of image region:
            image[rect.slice()]
         is the same as
            image[rect.top:rect.bottom, rect.left:rect.right]

        """
        return (slice(int(max(0, self.top)), int(max(0, self.bottom))),
                slice(int(max(0, self.left)), int(max(0, self.right))))

    def as_contour(self):
        left, top, width, height = self
        raw = np.array([[[left, top]],
                        [[left + width - 1, top]],
                        [[left + width - 1, top + height - 1]],
                        [[left, top + height - 1]]], dtype=np.int32)
        return Contour(raw)

    def get_bounding_rect(self):
        return self

    def get_mask(self, scale_factor=1):
        if scale_factor != 1:
            return self.scale(scale_factor).get_mask()

        if self._mask is None:
            self._mask = np.ones((max(0, self.bottom) - max(0, self.top),
                                  max(0, self.right) - max(0, self.left)),
                                 dtype=np.uint8)
        return self._mask

    def contains(self, other):
        return self.contains_rect(other.get_bounding_rect())


class Contour(Shape):
    """Contains contour, its boundingRect and area (last two are lazy evaluated)"""

    def __init__(self, contour, thresh=None):
        self._contour = contour
        self._rect = None
        self._mask = None
        self._area = None
        self._outer = None
        self._moments = None
        self._convex_hull = None
        self._vertexes = None
        self._length = None
        self.threshold = thresh

    def _init_area(self):
        _area = cv2.contourArea(np.float32(self._contour), True)
        self._area = fabs(_area)
        self._outer = _area < 0

    @property
    def contour(self):
        return self._contour

    @property
    def raw(self):
        return self._contour

    @property
    def rect(self):
        """
        :rtype: Rectangle
        """
        if self._rect is None:
            self._rect = Rectangle(*cv2.boundingRect(self._contour))
        return self._rect

    @property
    def area(self):
        """returns absolute value"""
        if self._area is None:
            self._init_area()
        return self._area

    @property
    def outer(self):
        if self._outer is None:
            self._init_area()
        return self._outer

    @property
    def inner(self):
        return not self.outer

    @property
    def convex_hull(self):
        if self._convex_hull is None:
            hull = cv2.convexHull(self.contour, clockwise=self.outer)
            self._convex_hull = Contour(hull, self.threshold)
            self._convex_hull._rect = self._rect
        return self._convex_hull

    @property
    def moments(self):
        if self._moments is None:
            self._moments = cv2.moments(self._contour)
        return self._moments

    def as_contour(self):
        return self

    def get_bounding_rect(self):
        return self.rect

    def _create_mask(self, scale_factor=None):
        rect = self.rect
        contour = self.raw

        if scale_factor is not None:
            rect = rect.scale(scale_factor)
            contour = contour * scale_factor

        x, y, w, h = rect
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 1, thickness=-1, offset=(-x, -y))
        return mask

    def get_mask(self, scale_factor=1):
        """Return contour mask

        :return: mask image
        :rtype: ndarray
        """
        if scale_factor != 1:
            return self._create_mask(scale_factor)

        if self._mask is None:
            self._mask = self._create_mask()

        return self._mask
