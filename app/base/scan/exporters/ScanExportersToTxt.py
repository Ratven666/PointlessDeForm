

class ScanExportersToTxt:

    def __init__(self, file_path):
        self.file_path = file_path

    def export(self, scan):
        with open(self.file_path, "w", encoding="UTF-8") as file:
            for point in scan:
                if point.color is None:
                    point_str = f"{point.x:.3f} {point.y:.3f} {point.z:.3f}\n"
                elif point.id_ is None:
                    point_str = f"{point.x:.3f} {point.y:.3f} {point.z:.3f} {point.color[0]} {point.color[1]} {point.color[2]}\n"
                else:
                    point_str = (f"{point.x:.3f} {point.y:.3f} {point.z:.3f} "
                                 f"{point.color[0]} {point.color[1]} {point.color[2]} {point.id_}\n")
                file.write(point_str)
