class Filter(list):
    def __init__(self, *args):
        super(Filter, self).__init__(args)

    def add_filter(self, filter_type, *args):
        if filter_type == "max_area":
            Filter.check_args(filter_type, 1, len(args))
            self.append(lambda c: c.area <= args[0])
        elif filter_type == "min_area":
            Filter.check_args(filter_type, 1, len(args))
            self.append(lambda c: c.area >= args[0])
        elif filter_type == "min_fill":
            Filter.check_args(filter_type, 1, len(args))
            self.append(lambda c: c.rect.area != 0 and (c.area / c.rect.area) >= args[0])
        elif filter_type == "smoothness":
            Filter.check_args(filter_type, 1, len(args))
            self.append(lambda c: c.vertexes != 0 and (c.length / c.vertexes) >= args[0])
        elif filter_type == "max_height":
            Filter.check_args(filter_type, 1, len(args))
            self.append(lambda c: c.rect.height <= args[0])
        elif filter_type == "min_height":
            Filter.check_args(filter_type, 1, len(args))
            self.append(lambda c: c.rect.height >= args[0])
        elif filter_type == "min_aspect_ratio":
            Filter.check_args(filter_type, 1, len(args))
            self.append(lambda c: c.rect.height != 0 and (float(c.rect.width) / c.rect.height) >= args[0])
        elif filter_type == "outer":
            self.append(lambda c: c.outer)
        elif filter_type == "inner":
            self.append(lambda c: c.inner)
        else:
            raise AttributeError("Unsupported filter type: `" + filter_type + "`")

    def filter(self, contours):
        return filter(lambda c: all(f(c) for f in self), contours)

    @staticmethod
    def check_args(filter_type, exp_arg, got_arg):
        if exp_arg != got_arg:
            raise AttributeError("Expected " + str(exp_arg) + " arguments for `" +
                                 filter_type + "` filter (got " + str(got_arg) + ")")
