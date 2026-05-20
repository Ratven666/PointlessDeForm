import numpy as np

from app.base.scan.Scan import Scan
from app.base.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC


class PlaneL1Fitter(PlaneFitterABC):
    """
    Робастная L1-подгонка плоскости методом IRLS.

    ВАЖНО ПО ТОЧНОСТИ:
    ------------------
    Эта реализация оценивает положение плоскости через IRLS + SVD взвешенного
    облака точек. Такая оценка параметров (A, B, C, D) не является обычным
    линейным МНК в явной параметризации X @ p = 0.

    Поэтому "строгая" ковариация параметров плоскости в духе

        Sigma = (X^T W X)^(-1) (X^T W^2 X) (X^T W X)^(-1)

    здесь статистически НЕСОГЛАСОВАНА с реальным способом оценки параметров,
    и на практике может давать неадекватно большие или неинтерпретируемые
    значения.

    В этой версии класс ведёт себя честнее и стабильнее:
    - положение плоскости оценивается робастно;
    - sigma0 оценивается по медиане абсолютных расстояний до плоскости;
    - cov_params НЕ выдаётся как строгая матрица ковариации, а заменяется на
      консервативную диагональную surrogate-оценку масштаба параметров;
    - при необходимости внешний код может трактовать такую ковариацию как
      fallback, а не как строгую статистическую оценку.

    Это делает поведение устойчивым и предотвращает взрыв ошибок точки
    пересечения из-за некорректной ковариационной модели.
    """

    def __init__(self, scan: Scan):
        super().__init__(scan)
        self.eps = None
        self.max_iter = None
        self.tol = None

        self.cov_params: np.ndarray | None = None
        self.sigma0: float | None = None
        self.covariance_mode: str = "surrogate_diagonal"

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------
    def fit_plane(self, *args, eps=1e-6, max_iter=50, tol=1e-6, **kwargs):
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol

        normal, point_on_plane, d, weights_final = self._fit_l1()
        pts = self._scan_to_numpy()
        self._compute_conservative_accuracy(pts, normal, d, weights_final)

        return self.scan, normal, point_on_plane, d

    # ------------------------------------------------------------------
    # Начальное L2-решение
    # ------------------------------------------------------------------
    def _initial_l2_plane(self, pts: np.ndarray):
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        return normal, d

    # ------------------------------------------------------------------
    # Один шаг взвешенного PCA/LS-приближения
    # ------------------------------------------------------------------
    def _weighted_l2_plane(self, pts: np.ndarray, weights: np.ndarray):
        w = weights.reshape(-1, 1)
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("Сумма весов нулевая или отрицательная")

        centroid = (w * pts).sum(axis=0) / w_sum
        centered = pts - centroid
        weighted_centered = centered * np.sqrt(w)

        _, _, vh = np.linalg.svd(weighted_centered, full_matrices=False)
        normal = vh[-1, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        return normal, d

    # ------------------------------------------------------------------
    # Основной IRLS-цикл
    # ------------------------------------------------------------------
    def _fit_l1(self):
        pts = self._scan_to_numpy()
        if pts.shape[0] < 3:
            raise ValueError("Для оценки плоскости нужно минимум 3 точки")

        normal, d = self._initial_l2_plane(pts)
        weights = np.ones(pts.shape[0], dtype=float)

        for _ in range(self.max_iter):
            dist = pts @ normal + d
            weights = 1.0 / (np.abs(dist) + self.eps)

            normal_new, d_new = self._weighted_l2_plane(pts, weights)

            if np.dot(normal_new, normal) < 0:
                normal_new = -normal_new
                d_new = -d_new

            delta_n = np.linalg.norm(normal_new - normal)
            delta_d = abs(d_new - d)

            normal = normal_new
            d = d_new

            if max(delta_n, delta_d) < self.tol:
                break

        point_on_plane = self._compute_point_on_plane(normal, d)
        return normal, point_on_plane, d, weights

    # ------------------------------------------------------------------
    # Честная и устойчивая оценка масштаба точности
    # ------------------------------------------------------------------
    def _compute_conservative_accuracy(self,
                                       pts: np.ndarray,
                                       normal: np.ndarray,
                                       d: float,
                                       weights: np.ndarray):
        """
        Оценивает только устойчивый масштаб ошибки, без претензии на строгую
        ковариацию параметров IRLS-PCA модели.

        Что делаем:
        1. Считаем ортогональные остатки до плоскости.
        2. Оцениваем sigma0 как медиану абсолютных остатков.
        3. Строим консервативную surrogate-ковариацию параметров в виде
           диагональной матрицы.

        Идея: для L1-фитера лучше вернуть устойчивую, ограниченную по масштабу
        матрицу, чем статистически несогласованную и потенциально взрывающуюся
        "точную" ковариацию.
        """
        residuals = np.abs(pts @ normal + d)

        median_abs = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median_abs)))

        robust_scale = max(median_abs, 1.4826 * mad, self.eps)
        self.sigma0 = robust_scale

        n = pts.shape[0]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        spreads = np.std(centered, axis=0, ddof=1) if n > 1 else np.ones(3, dtype=float)
        spreads = np.where(spreads > self.eps, spreads, 1.0)

        sigma_a = robust_scale / spreads[0]
        sigma_b = robust_scale / spreads[1]
        sigma_c = robust_scale / spreads[2]
        sigma_d = robust_scale

        if n > 4:
            shrink = np.sqrt(n)
            sigma_a /= shrink
            sigma_b /= shrink
            sigma_c /= shrink
            sigma_d /= shrink

        self.cov_params = np.diag([
            sigma_a ** 2,
            sigma_b ** 2,
            sigma_c ** 2,
            sigma_d ** 2,
        ])

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_point_on_plane(normal: np.ndarray, d: float) -> np.ndarray:
        A, B, C = normal
        if abs(C) > 1e-8:
            return np.array([0.0, 0.0, -d / C], dtype=float)
        if abs(B) > 1e-8:
            return np.array([0.0, -d / B, 0.0], dtype=float)
        return np.array([-d / A, 0.0, 0.0], dtype=float)
