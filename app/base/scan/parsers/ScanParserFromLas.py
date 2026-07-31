import laspy
from CONFIG import POINTS_CHUNK_COUNT
from app.base.scan.ScanPoint import ScanPoint
from app.base.scan.parsers.ScanParserABC import ScanParserABC


class ScanParserFromLas(ScanParserABC):

    def __init__(self, file_path, chunk_count=POINTS_CHUNK_COUNT):
        super().__init__(file_path)
        self.chunk_count = chunk_count

    @staticmethod
    def __get_xyz(not_scaled_xyz, scales, offsets):
        xyz = []
        for idx, coord in enumerate(not_scaled_xyz):
            xyz.append(float(coord * scales[idx] + offsets[idx]))
        return xyz

    @staticmethod
    def __get_rgb(not_scaled_rgb):
        # Проверяем, есть ли цветовые данные
        if not_scaled_rgb is None:
            return [0, 0, 0]  # Возвращаем чёрный по умолчанию
        COLOR_SCALE = 0.003906309605554284  # 1/256
        return [int(r_g_b * COLOR_SCALE) for r_g_b in not_scaled_rgb]

    def parse(self, scan):
        with laspy.open(self.file_path) as input_las:
            # Проверяем, есть ли цветовые данные в файле
            print("Available dimensions:", input_las.header.point_format.dimension_names)

            for points in input_las.chunk_iterator(self.chunk_count):
                offsets = points.offsets
                scales = points.scales
                points_array = points.array

                # Проверяем наличие цветовых полей
                has_red = 'red' in points_array.dtype.names
                has_green = 'green' in points_array.dtype.names
                has_blue = 'blue' in points_array.dtype.names

                for point in points_array:
                    xyz = self.__get_xyz((point['X'], point['Y'], point['Z']), offsets=offsets, scales=scales)

                    # Получаем цвет, если он есть
                    if has_red and has_green and has_blue:
                        rgb = self.__get_rgb((point['red'], point['green'], point['blue']))
                    else:
                        rgb = [0, 0, 0]  # Чёрный по умолчанию

                    point = ScanPoint(x=xyz[0], y=xyz[1], z=xyz[2], color=rgb)
                    scan.add_point(point)
