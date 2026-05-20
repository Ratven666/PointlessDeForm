import os
import sys
from typing import List
from pathlib import Path

from app.base.NamedPoint import NamedPoint
from app.base.scan.Scan import Scan


def get_named_point_list(file_path: str) -> List[NamedPoint]:
    points_list = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            point = NamedPoint(line[0], *map(float, line[1:]))
            points_list.append(point)
    return points_list

def split_scan_by_points(scan, points, cube_size: float = 0.5):
    """
    Разделяет scan на подсканы вокруг заданных точек.

    Для каждой точки из points создается пустой Scan с именем точки.
    Затем все точки исходного scan одним проходом распределяются
    по соответствующим подсканам, если попадают в куб вокруг опорной точки.

    Parameters
    ----------
    scan : Scan
        Исходный скан.
    points : iterable
        Список точек-центров. У точки должны быть:
        - x, y, z
        - name (или id)
    cube_size : float
        Размер ребра куба.

    Returns
    -------
    dict[str, Scan]
        Словарь подсканов по именам опорных точек.
    """
    if cube_size <= 0:
        raise ValueError("cube_size must be > 0")

    half = cube_size / 2

    sub_scans = {}
    point_items = []

    for i, center_point in enumerate(points):
        point_name = getattr(center_point, "name", None) or getattr(center_point, "id", None) or f"point_{i}"
        sub_scans[point_name] = Scan(scan_name=point_name)
        point_items.append((point_name, center_point))

    for scan_point in scan:
        for point_name, center_point in point_items:
            if (
                abs(scan_point.x - center_point.x) <= half
                and abs(scan_point.y - center_point.y) <= half
                and abs(scan_point.z - center_point.z) <= half
            ):
                sub_scans[point_name].add_point(scan_point)

    return sub_scans

def save_sub_scans(sub_scans: dict, parent_scan, base_dir: str = "."):
    """
    Сохраняет каждый sub_scan в отдельный файл.

    Parameters
    ----------
    sub_scans : dict[str, Scan]
        Словарь субсканов, где ключ — имя точки/субскана.
    parent_scan : Scan
        Родительский scan, имя которого используется как имя папки.
    base_dir : str
        Базовая директория, в которой будет создана папка родительского скана.

    Returns
    -------
    Path
        Путь к папке, куда были сохранены файлы.
    """
    parent_dir = Path(base_dir) / parent_scan.name
    parent_dir.mkdir(parents=True, exist_ok=True)

    for point_name, sub_scan in sub_scans.items():
        file_path = parent_dir / f"{point_name}.txt"
        sub_scan.export_points_from_file(str(file_path))

    return parent_dir

if __name__ == '__main__':
    file_path = "../../data/8_floors_wall/2334_tochki.txt"

    point_list = get_named_point_list(file_path)
    # print(*point_list, sep='\n')

    scan_names = "scan_2334_filt", "scan_2335_filt", "scan_2336_filt"
    for scan_name in scan_names:
        scan = Scan(scan_name=scan_name)
        scan.import_points_from_file(os.path.join("..", "..", "data", "8_floors_wall", f"{scan_name}.txt"))
        print(scan)

        sub_scans = split_scan_by_points(scan, point_list, cube_size=0.6)

        save_sub_scans(sub_scans, scan)
