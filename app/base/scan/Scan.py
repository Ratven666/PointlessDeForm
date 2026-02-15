from app.base.Point import Point
from app.base.scan.ScanPoint import ScanPoint
from app.base.scan.exporters.ScanExportersToTxt import ScanExportersToTxt
from app.base.scan.parsers.ScanParserFactory import ScanParserFactory
from app.base.scan.plotters.ScanPlotterMPL import ScanPlotterMPL


class Scan:

    def __init__(self, scan_name: str):
        self.name = scan_name
        self._points = []
        self.borders = {"x_min": None,
                        "x_max": None,
                        "y_min": None,
                        "y_max": None,
                        "z_min": None,
                        "z_max": None,
                        }

    def __len__(self):
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __str__(self):
        return f"{self.__class__.__name__} (scan_name={self.name}, num_of_point={len(self)}, borders={self.borders})"

    def add_point(self, point, color=(0, 0, 0)):
        if isinstance(point, ScanPoint):
            self._points.append(point)
            self.borders = self._check_border(borders_dict=self.borders, point=point)
        elif isinstance(point, Point):
            s_point = ScanPoint(x=point.x,
                                y=point.y,
                                z=point.z,
                                color=color)
            self.add_point(s_point)
        return self

    def import_points_from_file(self, file_path, parser=ScanParserFactory):
        parser = parser(file_path)
        parser.parse(scan=self)
        return self

    def export_points_from_file(self, file_path, parser=ScanExportersToTxt):
        parser = parser(file_path)
        parser.export(scan=self)

    def filter_scan(self, filter_cls, *args, replace_points_in_scan=True, **kwargs):
        filter = filter_cls(*args, **kwargs)
        filtered_points = filter.filter(scan=self)
        if replace_points_in_scan:
            self._points = filtered_points
            f_scan = self
        else:
            f_scan_name = f"{self.name}_filtered"
            f_scan = Scan(scan_name=f_scan_name)
            f_scan._points = filtered_points
        f_scan.borders = self._get_borders_dict(f_scan._points)
        return f_scan

    def plot(self, *args, plotter=ScanPlotterMPL, **kwargs):
        plotter = plotter(*args, **kwargs)
        fig_ax = plotter.plot(scan=self)
        return fig_ax

    @staticmethod
    def _check_border(borders_dict, point):
        if borders_dict["x_min"] is None:
            borders_dict = {"x_min": point.x,
                            "x_max": point.x,
                            "y_min": point.y,
                            "y_max": point.y,
                            "z_min": point.z,
                            "z_max": point.z,
                            }
        if point.x < borders_dict["x_min"]:
            borders_dict["x_min"] = point.x
        if point.y < borders_dict["y_min"]:
            borders_dict["y_min"] = point.y
        if point.z < borders_dict["z_min"]:
            borders_dict["z_min"] = point.z
        if point.x > borders_dict["x_max"]:
            borders_dict["x_max"] = point.x
        if point.y > borders_dict["y_max"]:
            borders_dict["y_max"] = point.y
        if point.z > borders_dict["z_max"]:
            borders_dict["z_max"] = point.z
        return borders_dict

    def _get_borders_dict(self, points_lst):
        borders = {"x_min": None, "x_max": None,
                   "y_min": None, "y_max": None,
                   "z_min": None, "z_max": None,
                   }
        for point in points_lst:
            borders = self._check_border(borders_dict=borders,
                                         point=point)
        return borders


if __name__ == "__main__":
    scan = Scan("Scan1")
    print(scan)
    # scan.import_points_from_file(file_path=r"../../src/PCLD_1.txt")
    scan.import_points_from_file(file_path=r"../../src/PCLD_1.las")
    print(scan)

    scan.plot()
    for point in scan:
         print(point)
