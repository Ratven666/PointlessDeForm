import numpy as np
from app.base.Plane import Plane
from app.base.scan.Scan import Scan


class ScanL1PlaneFitter:
    """
    Вписывает одну плоскость Ax + By + Cz + D = 0 во весь скан,
    минимизируя сумму модулей расстояний до плоскости (L1).
    Реализовано через итеративно взвешенные НСК (IRLS).
    """

    def __init__(self, scan: Scan,
                 eps=1e-6,
                 max_iter=50,
                 tol=1e-6):
        """
        eps      – малый параметр для стабилизации весов (деление на |d|+eps).
        max_iter – максимум итераций IRLS.
        tol      – критерий остановки по изменению нормали и D.
        """
        self.scan = scan
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol

    def _scan_to_numpy(self):
        """Все точки скана в np.ndarray (N,3)."""
        pts = np.array([[p.x, p.y, p.z] for p in self.scan], dtype=float)
        return pts

    def _initial_l2_plane(self, pts):
        """Стартовое решение – обычный НСК (PCA)."""
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        return normal, d

    def _weighted_l2_plane(self, pts, weights):
        """
        Взвешенный НСК: минимизируем sum w_i * dist_i^2.
        Реализация через центрирование с учётом весов.
        """
        w = weights.reshape(-1, 1)
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("Сумма весов нулевая или отрицательная")

        # взвешенный центр
        centroid = (w * pts).sum(axis=0) / w_sum

        centered = pts - centroid
        # применяем веса к точкам
        W_centered = centered * np.sqrt(w)

        _, _, vh = np.linalg.svd(W_centered, full_matrices=False)
        normal = vh[-1, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        return normal, d

    def fit_l1(self):
        """
        Строит плоскость по всем точкам скана (L1).
        Возвращает Plane.
        """
        pts = self._scan_to_numpy()
        if pts.shape[0] < 3:
            raise ValueError("Для оценки плоскости нужно минимум 3 точки")

        # старт – обычный L2-подгон
        normal, d = self._initial_l2_plane(pts)

        for _ in range(self.max_iter):
            # расстояния до текущей плоскости (с учётом единичной нормали)
            dist = np.dot(pts, normal) + d
            # веса ~ 1 / |dist|
            weights = 1.0 / (np.abs(dist) + self.eps)

            # новое L2-решение с весами
            normal_new, d_new = self._weighted_l2_plane(pts, weights)

            # проверка сходимости
            delta_n = np.linalg.norm(normal_new - normal)
            delta_d = abs(d_new - d)
            normal, d = normal_new, d_new

            if max(delta_n, delta_d) < self.tol:
                break

        plane = Plane(normal=normal, point_on_plane=self._compute_point_on_plane(normal, d), d=d)
        return plane

    @staticmethod
    def _compute_point_on_plane(normal, d):
        """
        Находит какую-нибудь точку на плоскости.
        Например, пересечение с осью Z (если возможно).
        """
        A, B, C = normal
        if abs(C) > 1e-8:
            # x=y=0 → Cz + D = 0
            z = -d / C
            return np.array([0.0, 0.0, z], dtype=float)
        elif abs(B) > 1e-8:
            y = -d / B
            return np.array([0.0, y, 0.0], dtype=float)
        else:
            x = -d / A
            return np.array([x, 0.0, 0.0], dtype=float)

if __name__ == "__main__":
    scan = Scan("L1_0")
    scan.import_points_from_file(file_path="../../../../src/l1_label_0.txt")
    # scan.plot()

    sf = ScanL1PlaneFitter(scan)
    plane = sf.fit_l1()
    print(plane)

    # fig, ax = scan.plot()


