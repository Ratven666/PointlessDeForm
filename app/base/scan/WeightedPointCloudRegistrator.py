"""
Взвешенная регистрация облаков точек с учётом ковариаций CrossPoint.

Отличие от PointCloudRegistrator
---------------------------------
- Каждой точке назначается вес, обратно пропорциональный суммарной
  дисперсии координат: w_i = 1 / (tr(Σ_base_i) + tr(Σ_mov_i)).
- Если ковариация у точки отсутствует или помечена как ненадёжная
  (reliable_accuracy=False), ей назначается «медианный» вес по
  стратегии MISSING_COV_STRATEGY:
    'median'  – медиана весов надёжных точек
    'mean'    – среднее весов надёжных точек
    'min'     – минимальный вес (самая пессимистичная оценка)
    'unit'    – w=1 (как в оригинальном классе)
- Отбраковка выбросов также учитывает точности: порог рассчитывается
  как t_i = k_sigma * sqrt(tr(Σ_i)), а остаток нормируется на него.
- После вычисления трансформации ковариации точек переносятся:
  Σ' = R Σ R^T  (унаследовано из SpatialTransformation.transform_points).
"""

from __future__ import annotations

import logging

import numpy as np

from app.base.scan.SpatialTransformation import SpatialTransformation

logger = logging.getLogger(__name__)

# MISSING_COV_STRATEGY = "median"   # 'median' | 'mean' | 'min' | 'unit'
MISSING_COV_STRATEGY = "min"   # 'median' | 'mean' | 'min' | 'unit'


class WeightedPointCloudRegistrator:
    """
    Вычисляет жёсткую пространственную трансформацию (R, t) между двумя
    наборами CrossPoint, используя полную ковариационную информацию точек.

    Схема работы
    ------------
    1. Сопоставление по именам → base_xyz, mov_xyz, weights.
    2. w_i = 1 / (tr(Σ_base_i) + tr(Σ_mov_i)) для надёжных точек;
       fallback-вес для точек без ковариации.
    3. Взвешенный алгоритм Кабша (LSM) или взвешенный IRLS (L1),
       где начальные веса IRLS умножаются на весовую матрицу точности.
    4. Опциональная отбраковка выбросов по нормированным остаткам
       r̃_i = ||r_i|| / sqrt(tr(Σ_i)).
    5. Возврат SpatialTransformation с полной статистикой.

    Параметры
    ---------
    base_points       : list[CrossPoint]  – целевая система (эпоха 1)
    transform_points  : list[CrossPoint]  – трансформируемая система (эпоха 2)
    method            : 'LSM' или 'L1'
    k_sigma           : порог отбраковки по нормированным остаткам (None – без отбраковки)
    max_iter          : макс. итераций IRLS
    eps               : стабилизатор весов IRLS
    tol               : критерий остановки IRLS по норме ΔR + Δt
    missing_cov_strategy : стратегия для точек без ковариации
    """

    def __init__(
        self,
        base_points: list,
        transform_points: list,
        method: str = "LSM",
        k_sigma: float | None = 3.0,
        max_iter: int = 100,
        eps: float = 1e-6,
        tol: float = 1e-9,
        missing_cov_strategy: str = MISSING_COV_STRATEGY,
    ):
        self.base_points = base_points
        self.transform_points = transform_points
        self.method = method.upper()
        self.k_sigma = k_sigma
        self.max_iter = max_iter
        self.eps = eps
        self.tol = tol
        self.missing_cov_strategy = missing_cov_strategy

        if self.method not in {"LSM", "L1"}:
            raise ValueError(
                f"method должен быть 'LSM' или 'L1', получено: {self.method!r}"
            )

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def compute(self) -> SpatialTransformation:
        base_xyz, mov_xyz, weights, sigma_total, n_common = (
            self._match_common_points()
        )

        if n_common < 3:
            raise ValueError(
                f"Найдено только {n_common} общих точек — нужно минимум 3"
            )

        logger.info(
            "Общих точек: %d | метод: %s | "
            "w_min=%.4g w_max=%.4g w_mean=%.4g",
            n_common, self.method,
            float(weights.min()), float(weights.max()), float(weights.mean()),
        )

        # ── первичная оценка ─────────────────────────────────────────
        R, t = self._fit(base_xyz, mov_xyz, weights)

        # ── отбраковка выбросов по нормированным остаткам ────────────
        if self.k_sigma is not None:
            base_xyz, mov_xyz, weights, sigma_total = self._filter_outliers(
                base_xyz, mov_xyz, weights, sigma_total, R, t
            )
            R, t = self._fit(base_xyz, mov_xyz, weights)

        residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)
        n_used = len(base_xyz)

        logger.info(
            "Трансформация вычислена | n_used=%d | RMSE=%.6f m | "
            "MAE=%.6f m | max=%.6f m",
            n_used,
            float(np.sqrt(np.mean(residuals**2))),
            float(np.mean(np.abs(residuals))),
            float(np.max(np.abs(residuals))),
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
    # Сопоставление точек и построение весовой матрицы
    # ------------------------------------------------------------------

    def _match_common_points(self):
        base_dict = {p.name: p for p in self.base_points}
        mov_dict  = {p.name: p for p in self.transform_points}

        common_names = sorted(set(base_dict.keys()) & set(mov_dict.keys()))
        n_common = len(common_names)

        if n_common == 0:
            raise ValueError("Нет ни одной точки с совпадающими именами")

        base_xyz = np.array(
            [[base_dict[n].x, base_dict[n].y, base_dict[n].z] for n in common_names],
            dtype=float,
        )
        mov_xyz = np.array(
            [[mov_dict[n].x, mov_dict[n].y, mov_dict[n].z] for n in common_names],
            dtype=float,
        )

        # ── суммарная дисперсия и вес для каждой пары точек ─────────
        total_variances = []
        for n in common_names:
            bp = base_dict[n]
            mp = mov_dict[n]

            tr_b = self._trace_cov(bp)
            tr_m = self._trace_cov(mp)

            if tr_b is not None and tr_m is not None:
                total_variances.append(tr_b + tr_m)
            elif tr_b is not None:
                total_variances.append(tr_b)
            elif tr_m is not None:
                total_variances.append(tr_m)
            else:
                total_variances.append(None)   # нет ковариации у обеих точек

        weights, sigma_total = self._build_weights(np.array(total_variances, dtype=object))

        self._log_weight_summary(common_names, weights, total_variances)

        return base_xyz, mov_xyz, weights, sigma_total, n_common

    # ------------------------------------------------------------------

    @staticmethod
    def _trace_cov(point) -> float | None:
        """Возвращает tr(Σ) если ковариация надёжна, иначе None."""
        if not getattr(point, "reliable_accuracy", True):
            return None
        cov = getattr(point, "cov_xyz", None)
        if cov is None:
            return None
        tr = float(np.trace(cov))
        return tr if tr > 1e-20 else None

    def _build_weights(self, total_variances: np.ndarray):
        """
        Строит нормированный вектор весов и вектор sigma_total.

        sigma_total_i = sqrt(variance_i) используется при нормированной
        отбраковке выбросов.
        """
        n = len(total_variances)

        # Разделяем надёжные и ненадёжные
        known_vars = np.array(
            [v for v in total_variances if v is not None], dtype=float
        )

        # Fallback-вес для точек без ковариации
        if len(known_vars) > 0:
            strat = self.missing_cov_strategy
            if strat == "median":
                fallback_var = float(np.median(known_vars))
            elif strat == "mean":
                fallback_var = float(np.mean(known_vars))
            elif strat == "min":
                fallback_var = float(np.max(known_vars))   # макс. дисперсия → мин. вес
            else:   # 'unit' или любой другой
                fallback_var = None
        else:
            fallback_var = None   # нет ни одной надёжной точки → все w=1

        raw_weights = np.zeros(n, dtype=float)
        sigma_total = np.zeros(n, dtype=float)

        for i, var in enumerate(total_variances):
            if var is not None:
                raw_weights[i] = 1.0 / var
                sigma_total[i] = np.sqrt(var)
            elif fallback_var is not None:
                raw_weights[i] = 1.0 / fallback_var
                sigma_total[i] = np.sqrt(fallback_var)
                logger.debug(
                    "Точка %d: нет ковариации → fallback w=%.4g (стратегия: %s)",
                    i, raw_weights[i], self.missing_cov_strategy,
                )
            else:
                raw_weights[i] = 1.0
                sigma_total[i] = 1.0

        # Нормировка: sum(w) = 1 для численной стабильности Кабша
        raw_weights /= raw_weights.sum()

        return raw_weights, sigma_total

    # ------------------------------------------------------------------
    # Диспетчер методов
    # ------------------------------------------------------------------

    def _fit(self, base_xyz, mov_xyz, weights):
        if self.method == "LSM":
            return self._fit_kabsch(base_xyz, mov_xyz, weights)
        return self._fit_l1_irls(base_xyz, mov_xyz, weights)

    # ------------------------------------------------------------------
    # Взвешенный алгоритм Кабша (МНК, SVD)
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_kabsch(
        base_xyz: np.ndarray,
        mov_xyz: np.ndarray,
        weights: np.ndarray,
    ):
        """
        Взвешенный алгоритм Кабша.

        Минимизирует  Σ_i w_i · ‖base_i − (R @ mov_i + t)‖²

        1. Взвешенные центроиды:
               c_b = Σ w_i · base_i,   c_m = Σ w_i · mov_i
        2. Центрирование:
               base_c = base − c_b,    mov_c = mov − c_m
        3. Взвешенная ковариационная матрица:
               H = (W · mov_c)^T @ base_c,   W = diag(w)
        4. SVD:  H = U S V^T
               R = V · diag(1, 1, det(VU^T)) · U^T
        5. t = c_b − R @ c_m
        """
        w = weights / weights.sum()  # гарантируем нормировку

        c_b = (w[:, None] * base_xyz).sum(axis=0)
        c_m = (w[:, None] * mov_xyz).sum(axis=0)

        base_c = base_xyz - c_b
        mov_c  = mov_xyz  - c_m

        H = (mov_c * w[:, None]).T @ base_c   # (3×N) @ (N×3) = (3×3)

        U, _, Vt = np.linalg.svd(H)
        V = Vt.T

        # Исправление отражения (reflection fix)
        d = np.linalg.det(V @ U.T)
        D = np.diag([1.0, 1.0, d])

        R = V @ D @ U.T
        t = c_b - R @ c_m

        return R, t

    # ------------------------------------------------------------------
    # Взвешенный IRLS (L1) с учётом точностных весов
    # ------------------------------------------------------------------

    def _fit_l1_irls(
        self,
        base_xyz: np.ndarray,
        mov_xyz: np.ndarray,
        accuracy_weights: np.ndarray,
    ):
        """
        Взвешенный IRLS для L1-нормы с учётом точности точек.

        На каждой итерации:
            w_irls_i = 1 / (‖r_i‖ + eps)
            w_i = accuracy_weights_i · w_irls_i   (комбинация двух весов)
            затем нормировка и взвешенный Кабш.
        """
        # Инициализация: только точностные веса
        combined = accuracy_weights.copy()
        R, t = self._fit_kabsch(base_xyz, mov_xyz, combined)

        for iteration in range(self.max_iter):
            residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)

            # Веса IRLS × точностные веса
            w_irls = 1.0 / (np.abs(residuals) + self.eps)
            combined = accuracy_weights * w_irls
            combined /= combined.sum()

            R_new, t_new = self._fit_kabsch(base_xyz, mov_xyz, combined)

            delta = np.linalg.norm(R_new - R) + np.linalg.norm(t_new - t)
            R, t = R_new, t_new

            if delta < self.tol:
                logger.debug("IRLS сошёлся на итерации %d", iteration + 1)
                break

        return R, t

    # ------------------------------------------------------------------
    # Нормированная отбраковка выбросов
    # ------------------------------------------------------------------

    def _filter_outliers(
        self,
        base_xyz: np.ndarray,
        mov_xyz: np.ndarray,
        weights: np.ndarray,
        sigma_total: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ):
        """
        Нормированные остатки: r̃_i = ‖r_i‖ / sigma_total_i.

        Точка отбраковывается если r̃_i > k_sigma.
        Это учитывает точность точки: грубая точка допускает большой
        геометрический остаток до отбраковки.
        """
        residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)

        # Защита от деления на ноль
        safe_sigma = np.where(sigma_total > 1e-12, sigma_total, 1.0)
        normalized = residuals / safe_sigma

        mask = normalized <= self.k_sigma
        n_removed = int((~mask).sum())

        if n_removed > 0:
            logger.info(
                "Отбраковано по нормированным остаткам: %d / %d  "
                "(max r̃ = %.2f, порог = %.1f)",
                n_removed, len(base_xyz),
                float(normalized.max()), self.k_sigma,
            )

        return (
            base_xyz[mask],
            mov_xyz[mask],
            weights[mask],
            sigma_total[mask],
        )

    # ------------------------------------------------------------------
    # Вычисление остатков
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_residuals(
        base_xyz: np.ndarray,
        mov_xyz: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        transformed = (R @ mov_xyz.T).T + t
        return np.linalg.norm(base_xyz - transformed, axis=1)

    # ------------------------------------------------------------------
    # Вспомогательный лог
    # ------------------------------------------------------------------

    @staticmethod
    def _log_weight_summary(names, weights, variances):
        lines = ["  Веса точек (нормированные):"]
        for n, w, var in zip(names, weights, variances):
            sigma_str = (
                f"σ_total={np.sqrt(var)*1000:.2f} мм"
                if var is not None
                else "σ=N/A (fallback)"
            )
            lines.append(f"    {n:20s}  w={w:.6f}  {sigma_str}")
        logger.info("\n".join(lines))
