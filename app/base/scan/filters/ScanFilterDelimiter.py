from app.base.scan.filters.ScanFilterABC import ScanFilterABC


class ScanFilterDelimiter(ScanFilterABC):

    def __init__(self, delimiter):
        self.delimiter = delimiter
        self.counter = 0

    def filter(self, scan):
        point_lst = []
        for point in scan:
            if self.counter % self.delimiter == 0:
                point_lst.append(point)
            self.counter += 1
        return point_lst
