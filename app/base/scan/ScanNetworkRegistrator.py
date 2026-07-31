"""
Итерационная сетевая регистрация набора сканов по общим CrossPoint
с итерационной отбраковкой плохих точек на уровне сети.

Алгоритм
--------
1. Каждому скану сопоставляется словарь {name: CrossPoint} — только точки
   со статусом GOOD и reliable_accuracy=True.
2. Строится граф смежности: для каждой пары сканов считается число общих
   точек и суммарная «точностная ценность» (1/tr(Σ)).
3. Итерационно (внешний цикл — фиксация сканов):
   a. Из незафиксированных сканов выбирается пара (anchor, moving)
      с наибольшим числом общих хороших точек, где anchor уже
      зафиксирован в целевой СК.
   b. Внутренний цикл отбраковки (до max_rejection_iter итераций):
      - WeightedPointCloudRegistrator вычисляет T = (R, t) по текущему
        набору точек пары.
      - Для каждой точки считается нормированный остаток:
            r̃_i = ||r_i|| / sqrt(tr(Σ_anchor_i) + tr(Σ_moving_i))
      - Точки с r̃_i > k_sigma_network помечаются как отбракованные
        для данной пары (global_rejected[name] += 1).
      - Если отбракованных нет — цикл завершается.
      - Если после отбраковки осталось < min_common — регистрация
        для данной пары считается неудачной.
   c. Все CrossPoint скана moving трансформируются в систему anchor.
   d. Скан помечается как зафиксированный.
4. Первым опорным сканом (anchor_0) выбирается скан с наибольшей
   суммарной связностью (или задаётся явно).
5. Результат: RegistrationResult со статистикой и отбракованными точками.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterator

import numpy as np

from app.base.scan.IterativeRegistrator import IterativeRegistrator
from app.base.scan.SpatialTransformation import SpatialTransformation
from app.base.scan.WeightedPointCloudRegistrator import WeightedPointCloudRegistrator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Структуры данных
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RejectedPointInfo:
    """Информация об отбракованной точке."""
    name: str
    anchor_scan: str
    moving_scan: str
    residual_m: float          # геометрический остаток, м
    normalized_residual: float  # нормированный остаток r̃ = ||r|| / σ_total
    sigma_total_mm: float       # σ_total в мм


@dataclass
class RegistrationEdge:
    """Ребро графа: пара сканов с оценкой качества связи."""
    anchor_name: str
    moving_name: str
    n_common: int          # число общих хороших точек
    quality_score: float   # сумма весов (1/tr(Σ)) общих точек

    def __lt__(self, other: "RegistrationEdge") -> bool:
        return (self.n_common, self.quality_score) < (other.n_common, other.quality_score)


@dataclass
class RegistrationResult:
    """Результат сетевой регистрации."""
    anchor_scan: str
    transforms: dict[str, SpatialTransformation | None]   # None = опорный скан
    order: list[str]                                       # порядок фиксации
    edges_used: list[RegistrationEdge]
    failed_scans: list[str]
    rejected_points: list[RejectedPointInfo]               # все отбракованные точки
    registered_points: dict[str, dict[str, object]]        # scan → {name: CrossPoint}

    def summary(self) -> str:
        lines = [
            "=== ScanNetworkRegistrator result ===",
            f"Опорный скан   : {self.anchor_scan}",
            f"Зафиксировано  : {len(self.order)} / {len(self.transforms)}",
            f"Не удалось     : {self.failed_scans or '—'}",
            f"Отбраковано    : {len(self.rejected_points)} точек",
            "",
            f"{'Скан':<30} {'Метод':<6} {'N_common':>8} {'N_used':>6} "
            f"{'RMSE, м':>10} {'MAE, м':>10} {'max, м':>10}",
            "-" * 82,
        ]
        for name in self.order:
            T = self.transforms[name]
            if T is None:
                lines.append(
                    f"{name:<30} {'—':<6} {'—':>8} {'—':>6} "
                    f"{'0 (anchor)':>10} {'—':>10} {'—':>10}"
                )
            else:
                lines.append(
                    f"{name:<30} {T.method:<6} {T.n_common:>8} {T.n_used:>6} "
                    f"{T.rmse:>10.6f} {T.mae:>10.6f} {T.max_res:>10.6f}"
                )

        if self.rejected_points:
            lines += [
                "",
                f"{'Отбракованная точка':<25} {'Anchor':<25} {'Moving':<25} "
                f"{'||r||, мм':>10} {'r̃':>8} {'σ_total, мм':>12}",
                "-" * 105,
            ]
            for rp in self.rejected_points:
                lines.append(
                    f"{rp.name:<25} {rp.anchor_scan:<25} {rp.moving_scan:<25} "
                    f"{rp.residual_m * 1000:>10.3f} {rp.normalized_residual:>8.2f} "
                    f"{rp.sigma_total_mm:>12.3f}"
                )

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Основной класс
# ──────────────────────────────────────────────────────────────────────

class ScanNetworkRegistrator:
    """
    Итерационная сетевая регистрация набора сканов с отбраковкой точек.

    Параметры
    ---------
    cross_points_map      : dict[str, list[CrossPoint]]
    method                : 'LSM' или 'L1'
    k_sigma               : порог отбраковки внутри WeightedPointCloudRegistrator
    k_sigma_network       : порог нормированного остатка на уровне сети
                            (None — не выполнять сетевую отбраковку)
    max_rejection_iter    : макс. итераций внутреннего цикла отбраковки
    min_common            : мин. число точек для регистрации пары
    anchor_scan           : имя опорного скана (None — авто)
    missing_cov_strategy  : стратегия для точек без ковариации
    deep_copy_points      : если True — deepcopy CrossPoint перед работой
                            (защита исходных объектов от мутации)
    """

    def __init__(
        self,
        cross_points_map: dict[str, list],
        method: str = "LSM",
        k_sigma: float | None = 3.0,
        k_sigma_network: float | None = 3.0,
        max_rejection_iter: int = 10,
        min_common: int = 3,
        anchor_scan: str | None = None,
        missing_cov_strategy: str = "median",
        deep_copy_points: bool = True,
    ):
        self.method = method.upper()
        self.k_sigma = k_sigma
        self.k_sigma_network = k_sigma_network
        self.max_rejection_iter = max_rejection_iter
        self.min_common = min_common
        self.missing_cov_strategy = missing_cov_strategy
        self.deep_copy_points = deep_copy_points

        if self.method not in {"LSM", "L1"}:
            raise ValueError(f"method должен быть 'LSM' или 'L1', получено: {self.method!r}")

        self._good_points: dict[str, dict[str, object]] = {
            scan_name: self._filter_good(pts)
            for scan_name, pts in cross_points_map.items()
        }

        self.scan_names = list(self._good_points.keys())
        self.anchor_scan = anchor_scan or self._choose_anchor()

        logger.info(
            "ScanNetworkRegistrator | сканов: %d | опорный: %s | "
            "метод: %s | k_sigma_network: %s",
            len(self.scan_names), self.anchor_scan,
            self.method, self.k_sigma_network,
        )
        self._log_point_counts()

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def run(self) -> RegistrationResult:
        """Запускает итерационную сетевую регистрацию с отбраковкой."""

        # Рабочие копии: deepcopy защищает исходные CrossPoint от мутации
        if self.deep_copy_points:
            working_points: dict[str, dict[str, object]] = {
                name: {k: copy.deepcopy(v) for k, v in pts.items()}
                for name, pts in self._good_points.items()
            }
        else:
            working_points = {
                name: dict(pts)
                for name, pts in self._good_points.items()
            }

        transforms: dict[str, SpatialTransformation | None] = {
            name: None for name in self.scan_names
        }
        fixed: set[str] = {self.anchor_scan}
        order: list[str] = [self.anchor_scan]
        edges_used: list[RegistrationEdge] = []
        failed: list[str] = []
        all_rejected: list[RejectedPointInfo] = []

        remaining = set(self.scan_names) - fixed
        iteration = 0

        while remaining:
            iteration += 1

            candidates = list(self._candidate_edges(fixed, remaining, working_points))

            if not candidates:
                logger.warning(
                    "Нет доступных рёбер. Изолированные сканы: %s", sorted(remaining)
                )
                failed.extend(sorted(remaining))
                break

            best = max(candidates)
            logger.info(
                "Итерация %d | %s → %s | общих точек: %d | score: %.4g",
                iteration, best.anchor_name, best.moving_name,
                best.n_common, best.quality_score,
            )

            # ── Внутренний цикл итерационной отбраковки ──────────────
            T, rejected = self._register_with_rejection(
                anchor_name=best.anchor_name,
                moving_name=best.moving_name,
                working_points=working_points,
            )
            all_rejected.extend(rejected)

            if T is None:
                logger.error(
                    "Регистрация %s → %s не удалась после отбраковки",
                    best.anchor_name, best.moving_name,
                )
                failed.append(best.moving_name)
                remaining.discard(best.moving_name)
                continue

            # ── Трансформируем все точки moving → система anchor ─────
            moving_pts_all = list(working_points[best.moving_name].values())
            transformed_list = T.transform_points(moving_pts_all)
            working_points[best.moving_name] = {p.name: p for p in transformed_list}

            transforms[best.moving_name] = T
            fixed.add(best.moving_name)
            remaining.discard(best.moving_name)
            order.append(best.moving_name)
            edges_used.append(best)

            logger.info(
                "  Зафиксирован: %s | RMSE=%.6f m | MAE=%.6f m | "
                "n_used=%d | отбраковано точек: %d",
                best.moving_name, T.rmse, T.mae, T.n_used, len(rejected),
            )

        result = RegistrationResult(
            anchor_scan=self.anchor_scan,
            transforms=transforms,
            order=order,
            edges_used=edges_used,
            failed_scans=failed,
            rejected_points=all_rejected,
            registered_points=working_points,
        )
        logger.info("\n%s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Внутренний цикл итерационной отбраковки для одной пары сканов
    # ------------------------------------------------------------------

    def _register_with_rejection(
        self,
        anchor_name: str,
        moving_name: str,
        working_points: dict[str, dict[str, object]],
    ) -> tuple[SpatialTransformation | None, list[RejectedPointInfo]]:
        """
        Итерационная регистрация пары (anchor, moving) с отбраковкой точек.

        Алгоритм каждой итерации:
        1. WeightedPointCloudRegistrator → T = (R, t).
        2. Вычислить нормированные остатки r̃_i для всех общих точек.
        3. Отбраковать точки с r̃_i > k_sigma_network из working_points[moving].
        4. Если отбракованных 0 — выход. Если осталось < min_common — ошибка.

        Возвращает (SpatialTransformation | None, list[RejectedPointInfo]).
        """
        rejected_infos: list[RejectedPointInfo] = []

        # Рабочий набор имён точек (будет уменьшаться при отбраковке)
        active_anchor = dict(working_points[anchor_name])
        active_moving = dict(working_points[moving_name])

        for rej_iter in range(self.max_rejection_iter):
            common_names = sorted(set(active_anchor.keys()) & set(active_moving.keys()))
            n_common = len(common_names)

            if n_common < self.min_common:
                logger.warning(
                    "  [%s→%s] итер. отбраковки %d: осталось %d точек < min_common=%d",
                    anchor_name, moving_name, rej_iter, n_common, self.min_common,
                )
                return None, rejected_infos

            anchor_pts = [active_anchor[n] for n in common_names]
            moving_pts = [active_moving[n] for n in common_names]

            # ── Вычисляем трансформацию ──────────────────────────────
            try:
                reg = IterativeRegistrator(
                    base_points=anchor_pts,
                    transform_points=moving_pts,
                    method=self.method,
                    k_sigma=self.k_sigma,          # внутренняя отбраковка WCPR
                    missing_cov_strategy=self.missing_cov_strategy,
                )
                T = reg.compute()
            except ValueError as exc:
                logger.error(
                    "  [%s→%s] IterativeRegistrator: %s", anchor_name, moving_name, exc
                )
                return None, rejected_infos

            # ── Нет сетевой отбраковки — сразу возвращаем ────────────
            if self.k_sigma_network is None:
                return T, rejected_infos

            # ── Нормированные остатки по всем общим точкам ───────────
            base_xyz = np.array([[p.x, p.y, p.z] for p in anchor_pts], dtype=float)
            mov_xyz  = np.array([[p.x, p.y, p.z] for p in moving_pts], dtype=float)

            transformed = (T.R @ mov_xyz.T).T + T.t
            raw_residuals = np.linalg.norm(base_xyz - transformed, axis=1)

            sigma_total = self._compute_sigma_total(anchor_pts, moving_pts)
            safe_sigma = np.where(sigma_total > 1e-12, sigma_total, 1.0)
            norm_residuals = raw_residuals / safe_sigma

            # ── Отбраковка ────────────────────────────────────────────
            bad_mask = norm_residuals > self.k_sigma_network
            n_bad = int(bad_mask.sum())

            if n_bad == 0:
                logger.debug(
                    "  [%s→%s] сходимость на итерации %d (нет выбросов)",
                    anchor_name, moving_name, rej_iter,
                )
                return T, rejected_infos

            # Собираем информацию об отбракованных точках
            for i, name in enumerate(common_names):
                if bad_mask[i]:
                    rejected_infos.append(RejectedPointInfo(
                        name=name,
                        anchor_scan=anchor_name,
                        moving_scan=moving_name,
                        residual_m=float(raw_residuals[i]),
                        normalized_residual=float(norm_residuals[i]),
                        sigma_total_mm=float(sigma_total[i]) * 1000.0,
                    ))
                    # Удаляем из активного набора moving (не anchor!)
                    active_moving.pop(name, None)

            logger.info(
                "  [%s→%s] итер. %d: отбракованы %d точек: %s",
                anchor_name, moving_name, rej_iter, n_bad,
                [common_names[i] for i in range(n_common) if bad_mask[i]],
            )

        # Исчерпан лимит итераций — возвращаем последнюю трансформацию
        logger.warning(
            "  [%s→%s] исчерпан лимит итераций отбраковки (%d)",
            anchor_name, moving_name, self.max_rejection_iter,
        )
        return T, rejected_infos

    # ------------------------------------------------------------------
    # Граф смежности
    # ------------------------------------------------------------------

    def _candidate_edges(
        self,
        fixed: set[str],
        remaining: set[str],
        working_points: dict[str, dict[str, object]],
    ) -> Iterator[RegistrationEdge]:
        for anchor in fixed:
            anchor_names = set(working_points[anchor].keys())
            for moving in remaining:
                moving_names = set(working_points[moving].keys())
                common_names = anchor_names & moving_names
                n_common = len(common_names)
                if n_common < self.min_common:
                    continue
                quality = self._edge_quality(common_names, working_points[anchor], working_points[moving])
                yield RegistrationEdge(
                    anchor_name=anchor,
                    moving_name=moving,
                    n_common=n_common,
                    quality_score=quality,
                )

    @staticmethod
    def _edge_quality(
        common_names: set[str],
        anchor_pts: dict[str, object],
        moving_pts: dict[str, object],
    ) -> float:
        """Q = Σ_i 1 / (tr(Σ_anchor_i) + tr(Σ_moving_i))"""
        total = 0.0
        for name in common_names:
            ap, mp = anchor_pts[name], moving_pts[name]
            cov_a = getattr(ap, "cov_xyz", None)
            cov_m = getattr(mp, "cov_xyz", None)
            tr_a = float(np.trace(cov_a)) if cov_a is not None else None
            tr_m = float(np.trace(cov_m)) if cov_m is not None else None
            if tr_a is not None and tr_m is not None:
                denom = tr_a + tr_m
            elif tr_a is not None:
                denom = tr_a
            elif tr_m is not None:
                denom = tr_m
            else:
                denom = None
            total += (1.0 / denom) if (denom and denom > 1e-20) else 1.0
        return total

    # ------------------------------------------------------------------
    # Вычисление σ_total для массива пар точек
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sigma_total(
        anchor_pts: list,
        moving_pts: list,
    ) -> np.ndarray:
        """
        σ_total_i = sqrt(tr(Σ_anchor_i) + tr(Σ_moving_i)).
        Если ковариация отсутствует — используется 1.0 мм в качестве
        минимального порога (точка получает максимальную чувствительность).
        """
        sigma = np.ones(len(anchor_pts), dtype=float) * 1e-3  # 1 мм fallback

        for i, (ap, mp) in enumerate(zip(anchor_pts, moving_pts)):
            cov_a = getattr(ap, "cov_xyz", None)
            cov_m = getattr(mp, "cov_xyz", None)
            tr_a = float(np.trace(cov_a)) if cov_a is not None else 0.0
            tr_m = float(np.trace(cov_m)) if cov_m is not None else 0.0
            total_var = tr_a + tr_m
            if total_var > 1e-20:
                sigma[i] = np.sqrt(total_var)

        return sigma

    # ------------------------------------------------------------------
    # Выбор опорного скана
    # ------------------------------------------------------------------

    def _choose_anchor(self) -> str:
        connectivity: dict[str, int] = {name: 0 for name in self.scan_names}
        for a, b in combinations(self.scan_names, 2):
            n = len(set(self._good_points[a].keys()) & set(self._good_points[b].keys()))
            connectivity[a] += n
            connectivity[b] += n
        best = max(connectivity, key=lambda k: connectivity[k])
        logger.info("Автовыбор опорного скана: %s (связность=%d)", best, connectivity[best])
        return best

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_good(points: list) -> dict[str, object]:
        return {
            p.name: p
            for p in points
            if getattr(p, "status", None) == "GOOD"
            and getattr(p, "reliable_accuracy", True)
        }

    def _log_point_counts(self):
        lines = ["  Хороших точек на скан:"]
        for name in self.scan_names:
            pts = self._good_points[name]
            names_str = ", ".join(sorted(pts.keys())[:6])
            if len(pts) > 6:
                names_str += f", ... (+{len(pts) - 6})"
            lines.append(f"    {name:<30} : {len(pts):>3} [{names_str}]")
        logger.info("\n".join(lines))
