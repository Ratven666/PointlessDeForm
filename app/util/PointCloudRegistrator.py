from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from app.util.SpatialTransformation import SpatialTransformation

logger = logging.getLogger(__name__)


class PointCloudRegistrator:
    """
    Вычисляет матрицу жёсткой пространственной трансформации (R, t)
    для совмещения трансформируемого списка точек CrossPoint с базовым.

    Схема работы:
    1. Находим общие точки по именам.
    2. Оцениваем R и t одним из двух методов:
       - 'LSM': метод Кабша через SVD (минимизирует сумму квадратов расстояний).
       - 'L1':  итерационный IRLS (минимизирует сумму модулей расстояний).
    3. Возвращает объект SpatialTransformation с полной оценкой качества.

    Параметры:
    ----------
    base_points       – список CrossPoint (целевая система)
    transform_points  – список CrossPoint (трансформируемая система)
    method            – 'LSM' или 'L1'
    k_sigma           – порог отбраковки выбросов (None = без отбраковки)
    max_iter          – максимум итераций IRLS (только для L1)
    eps               – стабилизатор весов в IRLS
    tol               – критерий остановки IRLS по изменению R

    Пример:
    -------
    reg = PointCloudRegistrator(base_pts, moving_pts, method='LSM')
    transform = reg.compute()
    print(transform)
    """

    def __init__(self,
                 base_points: list,
                 transform_points: list,
                 method: str = "LSM",
                 k_sigma: float | None = 3.0,
                 max_iter: int = 100,
                 eps: float = 1e-6,
                 tol: float = 1e-9):
        self.base_points = base_points
        self.transform_points = transform_points
        self.method = method.upper()
        self.k_sigma = k_sigma
        self.max_iter = max_iter
        self.eps = eps
        self.tol = tol

        if self.method not in {"LSM", "L1"}:
            raise ValueError(f"method должен быть 'LSM' или 'L1', получено: {self.method!r}")

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------
    def compute(self) -> SpatialTransformation:
        base_xyz, mov_xyz, n_common = self._match_common_points()

        if n_common < 3:
            raise ValueError(f"Найдено только {n_common} общих точек — нужно минимум 3")

        logger.info("Найдено общих точек: %d | метод: %s", n_common, self.method)

        if self.method == "LSM":
            R, t = self._fit_kabsch(base_xyz, mov_xyz)
        else:
            R, t = self._fit_l1_irls(base_xyz, mov_xyz)

        if self.k_sigma is not None:
            base_xyz, mov_xyz, residuals_all = self._filter_outliers(base_xyz, mov_xyz, R, t)
            if self.method == "LSM":
                R, t = self._fit_kabsch(base_xyz, mov_xyz)
            else:
                R, t = self._fit_l1_irls(base_xyz, mov_xyz)

        residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)
        n_used = len(base_xyz)

        logger.info(
            "Трансформация вычислена | n_used=%d | RMSE=%.6f | MAE=%.6f",
            n_used,
            float(np.sqrt(np.mean(residuals ** 2))),
            float(np.mean(np.abs(residuals))),
        )

        return SpatialTransformation(
            R=R,
            t=t,
            method=self.method,
            n_common=n_common,
            n_used=n_used,
            residuals=residuals,
        )

    # ------------------------------------------------------------------
    # Сопоставление точек по именам
    # ------------------------------------------------------------------
    def _match_common_points(self):
        base_dict = {p.name: p for p in self.base_points}
        mov_dict = {p.name: p for p in self.transform_points}

        common_names = sorted(set(base_dict.keys()) & set(mov_dict.keys()))
        n_common = len(common_names)

        if n_common == 0:
            raise ValueError("Нет ни одной точки с совпадающими именами")

        base_xyz = np.array([[base_dict[n].x, base_dict[n].y, base_dict[n].z]
                              for n in common_names], dtype=float)
        mov_xyz = np.array([[mov_dict[n].x, mov_dict[n].y, mov_dict[n].z]
                             for n in common_names], dtype=float)

        return base_xyz, mov_xyz, n_common

    # ------------------------------------------------------------------
    # Метод Кабша (МНК, SVD)
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_kabsch(base_xyz: np.ndarray,
                    mov_xyz: np.ndarray,
                    weights: np.ndarray | None = None):
        """
        Алгоритм Кабша:
        Минимизирует sum_i w_i * ||base_i - (R @ mov_i + t)||^2.

        1. Центрируем облака с учётом весов.
        2. Строим ковариационную матрицу H = mov_c^T W base_c.
        3. SVD: H = U S V^T, R = V U^T.
        4. Обрабатываем особый случай неправильного отражения.
        """
        n = len(base_xyz)
        if weights is None:
            weights = np.ones(n, dtype=float)

        weights = weights / weights.sum()

        centroid_base = (weights[:, None] * base_xyz).sum(axis=0)
        centroid_mov = (weights[:, None] * mov_xyz).sum(axis=0)

        base_c = base_xyz - centroid_base
        mov_c = mov_xyz - centroid_mov

        H = (mov_c * weights[:, None]).T @ base_c

        U, S, Vt = np.linalg.svd(H)
        V = Vt.T

        d = np.linalg.det(V @ U.T)
        D = np.diag([1.0, 1.0, d])

        R = V @ D @ U.T
        t = centroid_base - R @ centroid_mov

        return R, t

    # ------------------------------------------------------------------
    # Итерационный IRLS (L1)
    # ------------------------------------------------------------------
    def _fit_l1_irls(self, base_xyz: np.ndarray, mov_xyz: np.ndarray):
        """
        IRLS для L1:
        На каждой итерации w_i = 1 / (||r_i|| + eps), затем взвешенный Кабш.
        """
        weights = np.ones(len(base_xyz), dtype=float)
        R, t = self._fit_kabsch(base_xyz, mov_xyz, weights)

        for _ in range(self.max_iter):
            residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)
            weights = 1.0 / (np.abs(residuals) + self.eps)
            weights = weights / weights.sum()

            R_new, t_new = self._fit_kabsch(base_xyz, mov_xyz, weights)

            delta = np.linalg.norm(R_new - R) + np.linalg.norm(t_new - t)
            R, t = R_new, t_new

            if delta < self.tol:
                break

        return R, t

    # ------------------------------------------------------------------
    # Отбраковка выбросов
    # ------------------------------------------------------------------
    def _filter_outliers(self,
                         base_xyz: np.ndarray,
                         mov_xyz: np.ndarray,
                         R: np.ndarray,
                         t: np.ndarray):
        residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)
        mean = np.mean(residuals)
        std = np.std(residuals)
        threshold = mean + self.k_sigma * std

        mask = residuals <= threshold
        n_removed = int((~mask).sum())

        if n_removed > 0:
            logger.info("Отбраковано выбросов: %d / %d", n_removed, len(base_xyz))

        return base_xyz[mask], mov_xyz[mask], residuals

    # ------------------------------------------------------------------
    # Вычисление остатков
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_residuals(base_xyz: np.ndarray,
                           mov_xyz: np.ndarray,
                           R: np.ndarray,
                           t: np.ndarray) -> np.ndarray:
        transformed = (R @ mov_xyz.T).T + t
        diffs = base_xyz - transformed
        return np.linalg.norm(diffs, axis=1)
