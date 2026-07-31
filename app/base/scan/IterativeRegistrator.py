from __future__ import annotations
import logging
import numpy as np
from app.base.scan.WeightedPointCloudRegistrator import WeightedPointCloudRegistrator
from app.base.scan.SpatialTransformation import SpatialTransformation

logger = logging.getLogger(__name__)


class IterativeRegistrator(WeightedPointCloudRegistrator):
    """
    Последовательное итерационное уравнивание с пошаговой отбраковкой.

    Алгоритм (Danish method / sequential robust estimation):
    ---------------------------------------------------------
    1. Уравниваем по всем общим точкам → R, t.
    2. Вычисляем нормированные остатки: r̃ᵢ = ‖rᵢ‖ / σᵢ
    3. Если max(r̃) > k_sigma → удаляем ОДНУ наихудшую точку.
    4. Повторяем с п.1 до тех пор, пока все r̃ ≤ k_sigma
       или осталось < min_points точек.

    Принципиальное отличие от PointCloudRegistrator:
    - Отбраковка поэтапная (по одной точке), а не одним проходом.
    - После каждого удаления пересчитывается вся трансформация.
    - Это предотвращает «маскировку» выброса другим выбросом.
    """

    def __init__(self, *args, min_points: int = 3, **kwargs):
        # Отключаем однопроходную отбраковку родителя,
        # управляем ею сами итерационно
        kwargs["k_sigma"] = None
        super().__init__(*args, **kwargs)
        self.k_sigma_iter = kwargs.get("k_sigma_iter", 3.0)
        self.min_points = min_points

    def compute(self) -> SpatialTransformation:
        base_xyz, mov_xyz, weights, sigma_total, n_common = (
            self._match_common_points()
        )

        if n_common < self.min_points:
            raise ValueError(
                f"Найдено только {n_common} общих точек — нужно минимум {self.min_points}"
            )

        iteration = 0
        while True:
            # ── уравнивание на текущем наборе ─────────────────────────
            R, t = self._fit(base_xyz, mov_xyz, weights)
            residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)

            # ── нормированные остатки ─────────────────────────────────
            safe_sigma = np.where(sigma_total > 1e-12, sigma_total, 1.0)
            r_norm = residuals / safe_sigma

            worst_idx = int(np.argmax(r_norm))
            worst_val = float(r_norm[worst_idx])

            logger.info(
                "Итерация %d | n=%d | RMSE=%.4f мм | "
                "max r̃=%.2f (т. %d, порог %.1f)",
                iteration, len(base_xyz),
                float(np.sqrt(np.mean(residuals ** 2))) * 1000,
                worst_val, worst_idx, self.k_sigma_iter,
            )

            # ── критерий останова ─────────────────────────────────────
            if worst_val <= self.k_sigma_iter:
                logger.info("Сошлось: все r̃ ≤ %.1f", self.k_sigma_iter)
                break

            if len(base_xyz) - 1 < self.min_points:
                logger.warning(
                    "Достигнут минимум точек (%d), останов без выполнения "
                    "критерия отбраковки (max r̃=%.2f)",
                    self.min_points, worst_val,
                )
                break

            # ── удаляем ОДНУ наихудшую точку ──────────────────────────
            mask = np.ones(len(base_xyz), dtype=bool)
            mask[worst_idx] = False
            base_xyz    = base_xyz[mask]
            mov_xyz     = mov_xyz[mask]
            weights     = weights[mask] / weights[mask].sum()
            sigma_total = sigma_total[mask]

            iteration += 1

        n_used = len(base_xyz)
        residuals = self._compute_residuals(base_xyz, mov_xyz, R, t)

        logger.info(
            "Итог | итераций=%d | n_used=%d / n_common=%d | "
            "RMSE=%.4f мм | MAE=%.4f мм | max=%.4f мм",
            iteration, n_used, n_common,
            float(np.sqrt(np.mean(residuals**2))) * 1000,
            float(np.mean(np.abs(residuals))) * 1000,
            float(np.max(np.abs(residuals))) * 1000,
        )

        return SpatialTransformation(
            R=R, t=t,
            method=self.method,
            n_common=n_common,
            n_used=n_used,
            residuals=residuals,
        )
