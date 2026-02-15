import matplotlib.pyplot as plt
import numpy as np

from app.base.Plane import Plane


class ScanLSMPlaneFitter:
    """
    Вписывает одну плоскость Ax + By + Cz + D = 0 во весь скан методом НСК (PCA).
    """

    def __init__(self, scan):
        self.scan = scan

    def _scan_to_numpy(self):
        """Все точки скана в np.ndarray (N,3)."""
        pts = np.array([[p.x, p.y, p.z] for p in self.scan], dtype=float)
        return pts

    def fit_least_squares(self):
        """
        Строит плоскость по всем точкам скана.
        Возвращает Plane.
        """
        pts = self._scan_to_numpy()
        if pts.shape[0] < 3:
            raise ValueError("Для оценки плоскости нужно минимум 3 точки")

        # центр масс
        centroid = pts.mean(axis=0)

        # PCA через SVD
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)

        # нормаль – последний сингулярный вектор
        normal = vh[-1, :]
        normal = normal / np.linalg.norm(normal)

        # Ax + By + Cz + D = 0, D = -n·p0
        d = -np.dot(normal, centroid)

        plane = Plane(normal=normal, point_on_plane=centroid, d=d)
        return plane

if __name__ == "__main__":
    from app.base.scan.Scan import Scan
    scan = Scan("L1_0")
    scan.import_points_from_file(file_path="../../../../src/l1_label_1.txt")
    # scan.plot()

    sf = ScanLSMPlaneFitter(scan)
    plane = sf.fit_least_squares()
    print(plane)

    fig, ax = scan.plot(is_show=False)

    plt.show()

