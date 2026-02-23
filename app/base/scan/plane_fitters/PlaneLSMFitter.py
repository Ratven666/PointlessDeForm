import numpy as np

from app.base.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC


class PlaneLSMFitter(PlaneFitterABC):

    def fit_plane(self, *args, **kwargs):
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
        normal = normal
        point_on_plane = centroid
        d = d
        scan = self.scan
        return scan, normal, point_on_plane, d