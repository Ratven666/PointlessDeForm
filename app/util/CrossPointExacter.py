import os

import numpy as np

from app.base.Point import Point
from app.base.scan.Scan import Scan
from app.base.scan.ScanPlane import ScanPlane
from app.base.scan.filters.ScanFilterByDBSCAN import ScanFilterByDBSCAN
from app.base.scan.plane_fitters.IterativePlaneFitter import IterativePlaneFitter
from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.base.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter
from app.base.scan.plotters.ScanPlotterWithLabelsMPL import ScanPlotterWithLabelsMPL
from app.base.scan.utils.ScanNormalsDirectionClassifier import ScanNormalsDirectionClassifier
from app.base.scan.utils.ScanSplitterByLabels import ScanSplitterByLabels
from app.util.CrossPoint import CrossPoint


class CrossPointExacter:

    def __init__(self, file_path: str,
                 choose_scan_directly_from_dbscan: bool = True,
                 show_scans=True,
                 labels=None,
                 eps=0.01):
        self.base_scan = self.__init_scan(file_path)
        self.choose_scan_directly_from_dbscan = choose_scan_directly_from_dbscan
        self.plane_scans = self.__separate_plane_scans(show_scans=show_scans,
                                                       labels=labels,
                                                       eps=eps)
        self.planes = None
        self.cross_point = None

    @staticmethod
    def __init_scan(file_path: str):
        scan_name = os.path.basename(file_path).split(".")[0]
        scan = Scan(scan_name)
        scan.import_points_from_file(file_path)
        scan.compute_normals(k=8)
        s_normals_c = ScanNormalsDirectionClassifier(scan)
        s_normals_c.classify_normals(n_classes=3, unify_hemisphere=True)
        return scan

    def __separate_plane_scans(self, show_scans, eps, labels=None):
        def choose_plane_scan(scan, label=None, eps=eps, min_samples=5, min_cluster_size=100):
            scan.filter_scan(filter_cls=ScanFilterByDBSCAN,
                             eps=eps,
                             min_samples=min_samples,
                             min_cluster_size=min_cluster_size,
                             )
            if show_scans:
                scan.plot(plotter=ScanPlotterWithLabelsMPL)
            scans = ScanSplitterByLabels(scan).split()
            if label is None:
                label = float(input("Number_of_claster? (int): "))
            return scans[label]

        scans = ScanSplitterByLabels(self.base_scan).split()
        plane_scan = []
        for idx, scan in enumerate(scans.values()):
            label = labels[idx] if labels is not None else None
            if self.choose_scan_directly_from_dbscan:
                scan = choose_plane_scan(scan, label=label)
            plane_scan.append(scan)
            if show_scans:
                scan.plot()
        return plane_scan

    def calculate_planes(self,
                         base_fitter=PlaneL1Fitter,
                         mse_threshold=0.0001,
                         max_iteration=20,
                         k_sigma=2,
                         ):
        scan_planes = []
        for scan in self.plane_scans:
            scan_plane = ScanPlane.fit_plane_to_scan(scan=scan, fitter=IterativePlaneFitter,
                                                     base_fitter=base_fitter,
                                                     mse_threshold=mse_threshold,
                                                     max_iteration=max_iteration,
                                                     k_sigma=k_sigma,
                                                     )
            scan_planes.append(scan_plane)
        self.planes = scan_planes
        return scan_planes

    def has_parallel_subset(self, angle_tol_rad=np.deg2rad(10.0), min_group_size=2):
        """
        planes          – iterable[Plane]
        angle_tol_rad   – допуск по углу между нормалями (радианы)
        min_group_size  – минимальный размер подмножества параллельных плоскостей
        Возвращает True, если существует подмножество из min_group_size
        (или больше) плоскостей, которые попарно параллельны в пределах допуска.
        """
        planes = list(self.planes)
        n = len(planes)
        if n < min_group_size:
            return False
        normals = np.array([p.normal for p in self.planes], dtype=float)
        # нормируем нормали на всякий случай
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        # косинус допуска
        cos_tol = np.cos(angle_tol_rad)
        # пытаемся сгруппировать нормали по направлению (с учётом ±)
        groups = []
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            if used[i]:
                continue
            # создаём новую группу с базовой нормалью
            base = normals[i]
            group = [i]
            used[i] = True
            # ищем все нормали, параллельные base в пределах допуска
            dots = normals @ base  # скалярные произведения
            # параллельность с учётом направления: |cos(theta)| ≈ 1
            mask = np.abs(dots) >= cos_tol
            idxs = np.where(mask & (~used))[0]
            for j in idxs:
                group.append(j)
                used[j] = True
            if len(group) >= min_group_size:
                return True
        return False

    def calculate_intersect_point(self):
        """
        Находит точку пересечения трёх плоскостей Ax + By + Cz + D = 0.

        planes: список из трёх объектов Plane.
        Возвращает np.ndarray shape (3,) – координаты точки пересечения.

        Если детерминант матрицы нормалей близок к нулю, выбрасывает ValueError.
        """
        # Система:
        # A1 x + B1 y + C1 z = -D1
        # A2 x + B2 y + C2 z = -D2
        # A3 x + B3 y + C3 z = -D3
        A = []
        b = []
        for pl in self.planes:
            Ai, Bi, Ci, Di = pl.equation
            A.append([Ai, Bi, Ci])
            b.append(-Di)
        A = np.array(A, dtype=float)  # shape (3,3)
        b = np.array(b, dtype=float)  # shape (3,)
        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            raise ValueError("Плоскости не имеют единственной точки пересечения (детерминант ~ 0)")
        # Решаем линейную систему
        x = np.linalg.solve(A, b)  # shape (3,)
        self.cross_point = CrossPoint(name=self.base_scan.name,
                                      x=x[0], y=x[1], z=x[2])
        plane_mses = [plane.mse for plane in self.planes]
        self.cross_point.load_mses(plane_mses=plane_mses)
        self.cross_point.status = "BAD" if self.has_parallel_subset() else "GOOD"
        return self.cross_point

    def get_result_str(self):
        return f"{self.base_scan.name}, {self.cross_point.x}, {self.cross_point.y}, {self.cross_point.z}"


if __name__ == "__main__":
    base_path = "../../data/200226/сканер/lazpredobr"
    target_path = "../../src"
    file_path = os.path.join(base_path, "Lt1predobr.las")

    cpe = CrossPointExacter(file_path, labels=[14, 5, 2], show_scans=False)

    # for scan in cpe.plane_scans:
    #     print(scan)
    #     scan.export_points_from_file(file_path=f"{os.path.join(target_path, scan.name)}.txt")

    cpe.calculate_planes()

    for plane in cpe.planes:
        print(plane)

    x = cpe.calculate_intersect_point()
    print(x)
    print(cpe.get_result_str())


