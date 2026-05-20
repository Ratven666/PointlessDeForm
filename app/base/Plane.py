import numpy as np


class Plane:
    """
    Плоскость Ax + By + Cz + D = 0.
    Хранит коэффициенты, нормаль, точку на плоскости и inliers.
    """

    def __init__(self, normal, point_on_plane, d):
        """
        normal          – np.ndarray shape (3,), единичная нормаль (A,B,C)
        point_on_plane  – np.ndarray shape (3,), любая точка на плоскости
        d               – скаляр D в уравнении Ax+By+Cz + D = 0
        """
        self.normal = normal.astype(float)
        self.point = point_on_plane.astype(float)
        self.d = float(d)

    @property
    def A(self):
        return self.normal[0]

    @property
    def B(self):
        return self.normal[1]

    @property
    def C(self):
        return self.normal[2]

    @property
    def D(self):
        return self.d

    @property
    def equation(self):
        """Коэффициенты (A,B,C,D)."""
        return self.A, self.B, self.C, self.d

    def distance_to_point(self, xyz):
        """
        xyz: np.ndarray shape (3,) или (N,3).
        Возвращает расстояние(я) от точки(точек) до плоскости.
        """
        xyz = np.asarray(xyz, dtype=float)
        num = np.dot(xyz, self.normal) + self.d
        return np.abs(num)  # нормаль единичная → делить на ||n|| не нужно

    def project_point(self, xyz):
        """
        Проекция точки(точек) на плоскость.
        xyz: (3,) или (N,3).
        """
        xyz = np.asarray(xyz, dtype=float)
        dist = np.dot(xyz - self.point, self.normal)
        return xyz - np.outer(dist, self.normal)

    def __repr__(self):
        A, B, C, D = self.equation
        return f"{self.__class__.__name__} (A={A:.6f}, B={B:.6f}, C={C:.6f}, D={D:.6f})"
