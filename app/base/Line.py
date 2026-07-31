import numpy as np
from typing import List
from app.base.Point import Point
from app.base.scan.Scan import Scan


class Line:
    """
    Прямая в 3D в виде:
      point + t * direction
    """

    def __init__(self, point: np.ndarray, direction: np.ndarray):
        point = np.asarray(point, dtype=float).reshape(3)
        direction = np.asarray(direction, dtype=float).reshape(3)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Направляющий вектор не должен быть нулевым")
        self.point = point
        self.direction = direction / norm

    @classmethod
    def _fit_from_points(cls, coords: np.ndarray) -> "Line":
        """
        Подгонка прямой по массиву координат shape (N, 3).
        """
        pts = np.asarray(coords, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("coords должен иметь форму (N, 3)")
        if pts.shape[0] < 2:
            raise ValueError("Для оценки прямой нужно минимум 2 точки")

        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        direction = vh[0, :]  # главный компонент

        return cls(point=centroid, direction=direction)

    @classmethod
    def fit_from_points_list(cls, points: List[Point] | Scan) -> "Line":
        """
        Подгонка прямой по списку Point.
        """
        if len(points) < 2:
            raise ValueError("Для оценки прямой нужно минимум 2 точки")

        coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
        return cls._fit_from_points(coords)

    def _distance_to_points(self, coords: np.ndarray) -> np.ndarray:
        """
        Ортогональное расстояние от набора координат (N,3) до прямой.
        """
        pts = np.asarray(coords, dtype=float)
        r = pts - self.point
        t = np.dot(r, self.direction)
        proj = self.point + np.outer(t, self.direction)
        diff = pts - proj
        return np.linalg.norm(diff, axis=1)

    def distance_to_point_objects(self, points: List[Point] | Scan) -> np.ndarray:
        """
        Ортогональные расстояния до прямой от списка Point.
        """
        coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
        return self._distance_to_points(coords)

    def __repr__(self) -> str:
        return f"Line(point={self.point}, direction={self.direction})"
