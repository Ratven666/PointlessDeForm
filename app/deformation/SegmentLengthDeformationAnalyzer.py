from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LengthDeformationResult:
    """
    Результат анализа деформации точки только по изменениям длин связанных сегментов.
    """

    point_name: str
    n_obs_initial: int
    n_obs_used: int

    used_neighbors: list
    rejected_neighbors: list

    delta_lengths: np.ndarray
    sigma_delta_lengths: np.ndarray
    weights: np.ndarray
    normalized_residuals: np.ndarray

    mean_delta_length: float
    rms_delta_length: float
    max_abs_delta_length: float

    test_statistic: Optional[float]
    p_value: Optional[float]
    significant: Optional[bool]

    reliable: bool
    message: str

    @property
    def mean_delta_length_mm(self):
        return self.mean_delta_length * 1000.0

    @property
    def rms_delta_length_mm(self):
        return self.rms_delta_length * 1000.0

    @property
    def max_abs_delta_length_mm(self):
        return self.max_abs_delta_length * 1000.0

    def as_dict(self):
        return {
            "point_name": self.point_name,
            "n_obs_initial": self.n_obs_initial,
            "n_obs_used": self.n_obs_used,
            "n_rejected": len(self.rejected_neighbors),
            "used_neighbors": ", ".join(self.used_neighbors),
            "rejected_neighbors": ", ".join(self.rejected_neighbors),
            "mean_delta_length": self.mean_delta_length,
            "mean_delta_length_mm": self.mean_delta_length_mm,
            "rms_delta_length": self.rms_delta_length,
            "rms_delta_length_mm": self.rms_delta_length_mm,
            "max_abs_delta_length": self.max_abs_delta_length,
            "max_abs_delta_length_mm": self.max_abs_delta_length_mm,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "reliable": self.reliable,
            "message": self.message,
        }

    def __str__(self):
        if not self.reliable:
            return (
                f"LengthDeformationResult(point={self.point_name}, "
                f"reliable=False, message={self.message})"
            )

        return (
            f"LengthDeformationResult(point={self.point_name}, "
            f"n_obs={self.n_obs_used}/{self.n_obs_initial}, "
            f"mean_dl={self.mean_delta_length:.6f} m, "
            f"rms_dl={self.rms_delta_length:.6f} m, "
            f"max_abs_dl={self.max_abs_delta_length:.6f} m, "
            f"stat={self.test_statistic:.6f}, p={self.p_value:.6g}, "
            f"significant={self.significant}, "
            f"rejected={self.rejected_neighbors})"
        )


class SegmentLengthDeformationAnalyzer:
    """
    Анализ деформации по изменениям длин сегментов без использования направлений.

    Инвариантность:
    - инвариантен к общему переносу системы координат;
    - инвариантен к общему повороту системы координат;
    - НЕ инвариантен к изменению масштаба.

    Отбраковка:
    - итерационно удаляется сегмент с максимальным нормированным отклонением
      z_i = |Δl_i| / σ(Δl_i),
      если z_i > rejection_threshold.
    """

    def __init__(
        self,
        alpha=0.05,
        use_weights=True,
        enable_rejection=False,
        rejection_threshold=3.0,
        min_obs=3,
        max_rejections=None,
    ):
        self.alpha = alpha
        self.use_weights = use_weights
        self.enable_rejection = enable_rejection
        self.rejection_threshold = rejection_threshold
        self.min_obs = min_obs
        self.max_rejections = max_rejections

    def analyze_for_point(self, point_name, seg_set_epoch1, seg_set_epoch2):
        segs1 = self._collect_segments_for_point(seg_set_epoch1, point_name)
        segs2 = self._collect_segments_for_point(seg_set_epoch2, point_name)

        if len(segs1) == 0 or len(segs2) == 0:
            return self._empty_result(point_name, "Нет сегментов для данной точки в одной из эпох")

        common_neighbors = sorted(set(segs1.keys()) & set(segs2.keys()))
        if len(common_neighbors) == 0:
            return self._empty_result(point_name, "Нет общих парных сегментов между эпохами")

        observations = []
        for neighbor_name in common_neighbors:
            seg1 = segs1[neighbor_name]
            seg2 = segs2[neighbor_name]

            dl = float(seg2.length - seg1.length)

            s1 = seg1.sigma_length
            s2 = seg2.sigma_length

            if s1 is not None and s2 is not None:
                sigma_dl = float(np.sqrt(max(s1 ** 2 + s2 ** 2, 0.0)))
            else:
                sigma_dl = np.nan

            if self.use_weights and np.isfinite(sigma_dl) and sigma_dl > 1e-16:
                w = 1.0 / (sigma_dl ** 2)
            else:
                w = 1.0

            observations.append({
                "neighbor": neighbor_name,
                "delta_length": dl,
                "sigma_delta_length": sigma_dl,
                "weight": w,
            })

        n_obs_initial = len(observations)

        if self.enable_rejection:
            observations, rejected_neighbors = self._apply_iterative_rejection(observations)
        else:
            rejected_neighbors = []

        if len(observations) < self.min_obs:
            return self._empty_result(
                point_name,
                f"После отбраковки осталось недостаточно наблюдений: {len(observations)} < {self.min_obs}",
                n_obs_initial=n_obs_initial,
                rejected_neighbors=rejected_neighbors,
            )

        used_neighbors = [obs["neighbor"] for obs in observations]
        delta_lengths = np.asarray([obs["delta_length"] for obs in observations], dtype=float)
        sigma_delta_lengths = np.asarray([obs["sigma_delta_length"] for obs in observations], dtype=float)
        weights = np.asarray([obs["weight"] for obs in observations], dtype=float)

        normalized_residuals = self._calc_normalized_residuals(delta_lengths, sigma_delta_lengths)

        mean_delta_length = float(np.mean(delta_lengths))
        rms_delta_length = float(np.sqrt(np.mean(delta_lengths ** 2)))
        max_abs_delta_length = float(np.max(np.abs(delta_lengths)))

        # Глобальный критерий на отсутствие деформации по длинам:
        # H0: все Δl_i = 0
        # T = Σ w_i * Δl_i² ~ χ²(n)
        n = len(delta_lengths)
        test_statistic = float(np.sum(weights * delta_lengths ** 2))
        p_value = float(1.0 - stats.chi2.cdf(test_statistic, df=n))
        significant = p_value < self.alpha

        reliable = True
        if not np.all(np.isfinite(delta_lengths)):
            reliable = False

        return LengthDeformationResult(
            point_name=point_name,
            n_obs_initial=n_obs_initial,
            n_obs_used=n,
            used_neighbors=used_neighbors,
            rejected_neighbors=rejected_neighbors,
            delta_lengths=delta_lengths,
            sigma_delta_lengths=sigma_delta_lengths,
            weights=weights,
            normalized_residuals=normalized_residuals,
            mean_delta_length=mean_delta_length,
            rms_delta_length=rms_delta_length,
            max_abs_delta_length=max_abs_delta_length,
            test_statistic=test_statistic,
            p_value=p_value,
            significant=significant,
            reliable=reliable,
            message="ok" if reliable else "Некорректные значения в наблюдениях",
        )

    def analyze_for_all_points(self, seg_set_epoch1, seg_set_epoch2):
        point_names = sorted(
            self._extract_all_point_names(seg_set_epoch1) &
            self._extract_all_point_names(seg_set_epoch2)
        )

        results = []
        for point_name in point_names:
            result = self.analyze_for_point(point_name, seg_set_epoch1, seg_set_epoch2)
            results.append(result)

        return results

    @staticmethod
    def results_to_dataframe(results):
        return pd.DataFrame([r.as_dict() for r in results])

    def _apply_iterative_rejection(self, observations):
        """
        Итерационная отбраковка по нормированным изменениям длин:
            z_i = |Δl_i| / σ(Δl_i)

        На каждом шаге удаляется одно худшее наблюдение.
        """
        obs = list(observations)
        rejected_neighbors = []
        n_rejections = 0

        while True:
            if len(obs) <= self.min_obs:
                break

            delta_lengths = np.asarray([o["delta_length"] for o in obs], dtype=float)
            sigma_delta_lengths = np.asarray([o["sigma_delta_length"] for o in obs], dtype=float)
            z = self._calc_normalized_residuals(delta_lengths, sigma_delta_lengths)

            if len(z) == 0 or not np.any(np.isfinite(z)):
                break

            worst_idx = int(np.nanargmax(z))
            worst_val = z[worst_idx]

            if worst_val <= self.rejection_threshold:
                break

            rejected_neighbors.append(obs[worst_idx]["neighbor"])
            del obs[worst_idx]
            n_rejections += 1

            if self.max_rejections is not None and n_rejections >= self.max_rejections:
                break

        return obs, rejected_neighbors

    @staticmethod
    def _calc_normalized_residuals(delta_lengths, sigma_delta_lengths):
        """
        z_i = |Δl_i| / σ(Δl_i)
        Если σ неизвестно или невалидно, используем NaN.
        """
        z = np.full_like(delta_lengths, np.nan, dtype=float)

        mask = np.isfinite(sigma_delta_lengths) & (sigma_delta_lengths > 1e-16)
        z[mask] = np.abs(delta_lengths[mask]) / sigma_delta_lengths[mask]

        return z

    @staticmethod
    def _collect_segments_for_point(seg_set, point_name):
        out = {}
        for seg in seg_set:
            if seg.p1.name == point_name:
                out[seg.p2.name] = seg
            elif seg.p2.name == point_name:
                out[seg.p1.name] = seg
        return out

    @staticmethod
    def _extract_all_point_names(seg_set):
        names = set()
        for seg in seg_set:
            names.add(seg.p1.name)
            names.add(seg.p2.name)
        return names

    def _empty_result(self, point_name, message, n_obs_initial=0, rejected_neighbors=None):
        if rejected_neighbors is None:
            rejected_neighbors = []

        return LengthDeformationResult(
            point_name=point_name,
            n_obs_initial=n_obs_initial,
            n_obs_used=0,
            used_neighbors=[],
            rejected_neighbors=rejected_neighbors,
            delta_lengths=np.array([], dtype=float),
            sigma_delta_lengths=np.array([], dtype=float),
            weights=np.array([], dtype=float),
            normalized_residuals=np.array([], dtype=float),
            mean_delta_length=np.nan,
            rms_delta_length=np.nan,
            max_abs_delta_length=np.nan,
            test_statistic=None,
            p_value=None,
            significant=None,
            reliable=False,
            message=message,
        )


if __name__ == "__main__":
    from tabulate import tabulate

    from app.cross_points.CrossPointListRestorer import CrossPointListRestorer
    from app.cross_points.CrossPointSegmentSet import CrossPointSegmentSet

    points_path_epoch1 = "/data/8_floors_wall/output/total_scan/cross_points_good_filtered_by_ellipsoid.csv"
    points_path_epoch2 = "/data/8_floors_wall/output/scan_2335_filt/cross_points_good_filtered_by_ellipsoid.csv"

    points1 = CrossPointListRestorer(points_path_epoch1).restore_all()
    points2 = CrossPointListRestorer(points_path_epoch2).restore_all()

    seg_set_1 = CrossPointSegmentSet.from_all_pairs(points1)
    seg_set_2 = CrossPointSegmentSet.from_all_pairs(points2)

    analyzer = SegmentLengthDeformationAnalyzer(
        alpha=0.05,
        use_weights=True,
        enable_rejection=True,
        rejection_threshold=3.0,
        min_obs=3,
        max_rejections=10,
    )

    one_result = analyzer.analyze_for_point("2_1_vl", seg_set_1, seg_set_2)
    print(one_result)
    print("Rejected:", one_result.rejected_neighbors)
    print("Used:", one_result.used_neighbors)

    results = analyzer.analyze_for_all_points(seg_set_1, seg_set_2)
    df = analyzer.results_to_dataframe(results)

    # print(df[[
    #     "point_name",
    #     "n_obs_initial",
    #     "n_obs_used",
    #     "n_rejected",
    #     "mean_delta_length_mm",
    #     "rms_delta_length_mm",
    #     "max_abs_delta_length_mm",
    #     "p_value",
    #     "significant",
    #     "rejected_neighbors",
    # ]])
    print(tabulate(df.sort_values("mean_delta_length_mm", ascending=True),
                   tablefmt="pretty", headers="keys"))