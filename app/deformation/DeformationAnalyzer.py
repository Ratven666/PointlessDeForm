from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DeformationResult:
    """
    Результат анализа смещения одной точки между двумя эпохами.
    """
    name: str

    # Вектор смещения и его СКП
    delta: np.ndarray            # (3,) вектор d = X2 - X1
    displacement: float          # |d|
    sigma_displacement: float    # СКП |d|
    cov_delta: np.ndarray        # (3,3) ковариация вектора смещения

    # Компоненты
    sigma_dx: float
    sigma_dy: float
    sigma_dz: float

    # 1D t-тест (по длине смещения)
    t_value: float
    p_value_t: float             # p-значение (двустороннее)
    significant_t: bool          # значимо по t-тесту

    # 3D chi^2-тест (congruence test)
    chi2_value: Optional[float]
    p_value_chi2: Optional[float]
    significant_chi2: Optional[bool]

    # Уровень значимости, использованный при тесте
    alpha: float

    # Надёжность оценки
    reliable: bool

    @property
    def displacement_mm(self):
        return self.displacement * 1000.0

    @property
    def sigma_displacement_mm(self):
        return self.sigma_displacement * 1000.0

    def as_dict(self):
        return {
            "name": self.name,
            "dx": float(self.delta[0]),
            "dy": float(self.delta[1]),
            "dz": float(self.delta[2]),
            "displacement": self.displacement,
            "displacement_mm": self.displacement_mm,
            "sigma_dx": self.sigma_dx,
            "sigma_dy": self.sigma_dy,
            "sigma_dz": self.sigma_dz,
            "sigma_displacement": self.sigma_displacement,
            "sigma_displacement_mm": self.sigma_displacement_mm,
            "t_value": self.t_value,
            "p_value_t": self.p_value_t,
            "significant_t": self.significant_t,
            "chi2_value": self.chi2_value,
            "p_value_chi2": self.p_value_chi2,
            "significant_chi2": self.significant_chi2,
            "alpha": self.alpha,
            "reliable": self.reliable,
        }

    def __str__(self):
        sig_t = "SIGNIFICANT" if self.significant_t else "not significant"
        sig_chi2 = (
            "n/a" if self.significant_chi2 is None
            else ("SIGNIFICANT" if self.significant_chi2 else "not significant")
        )
        return (
            f"[{self.name}] "
            f"d={self.displacement_mm:.2f} mm ± {self.sigma_displacement_mm:.2f} mm | "
            f"T={self.t_value:.3f} p={self.p_value_t:.4f} ({sig_t}) | "
            f"chi2={self.chi2_value:.3f} p={self.p_value_chi2:.4f} ({sig_chi2})"
            if self.chi2_value is not None
            else (
                f"[{self.name}] "
                f"d={self.displacement_mm:.2f} mm ± {self.sigma_displacement_mm:.2f} mm | "
                f"T={self.t_value:.3f} p={self.p_value_t:.4f} ({sig_t}) | "
                f"chi2=n/a"
            )
        )


class DeformationAnalyzer:
    """
    Анализирует пространственные смещения точек между двумя (или более) эпохами.

    Алгоритм для каждой пары одноимённых точек:
        1. Вектор смещения d = X2 - X1
        2. Ковариация Σ_d = Σ1 + Σ2 - C12 - C12^T
        3. СКП компонент и длины смещения (Якоби)
        4. 1D t-тест: T = |d| / σ(|d|), нулевая гипотеза |d|=0
        5. 3D chi^2-тест: χ² = d^T Σ_d^{-1} d, df=3 (если Σ_d обратима)
        6. p-значения и флаги значимости

    Parameters
    ----------
    alpha : float
        Уровень значимости (по умолчанию 0.05 → доверие 95%).
    cross_cov_map : dict | None
        Взаимные ковариации Cov(X_epoch1, X_epoch2) по имени точки:
        {point_name: cov_12 (3,3)}.
    """

    def __init__(self, alpha=0.05, cross_cov_map=None):
        self.alpha = alpha
        self.cross_cov_map = cross_cov_map or {}
        self._results: list[DeformationResult] = []

    # ------------------------------------------------------------------
    # Основные методы
    # ------------------------------------------------------------------

    def analyze_point_sets(self, points_epoch1, points_epoch2):
        """
        Анализ смещений по двум спискам точек CrossPoint.
        Сопоставление — по атрибуту .name.
        """
        map1 = {p.name: p for p in points_epoch1}
        map2 = {p.name: p for p in points_epoch2}
        common_names = sorted(set(map1.keys()) & set(map2.keys()))

        self._results = []
        for name in common_names:
            result = self._analyze_single_point(map1[name], map2[name])
            self._results.append(result)

        return self

    def analyze_segment_sets(self, seg_set_epoch1, seg_set_epoch2):
        """
        Анализ смещений по двум CrossPointSegmentSet.
        Ищет отрезки с одинаковыми именами в обоих наборах.
        Каждый отрезок рассматривается как точка с координатами
        midpoint и ковариацией midpoint (для демонстрации API).

        Рекомендуемый путь — analyze_point_sets() напрямую по точкам.
        """
        names1 = {(seg.p1.name, seg.p2.name) for seg in seg_set_epoch1}
        names2 = {(seg.p1.name, seg.p2.name) for seg in seg_set_epoch2}
        common = names1 & names2

        self._results = []
        for name_pair in sorted(common):
            try:
                seg1 = seg_set_epoch1.get_segment_by_point_names(*name_pair)
                seg2 = seg_set_epoch2.get_segment_by_point_names(*name_pair)
            except KeyError:
                continue

            result = self._analyze_segment_pair(seg1, seg2)
            self._results.append(result)

        return self

    # ------------------------------------------------------------------
    # Ядро вычислений
    # ------------------------------------------------------------------

    def _analyze_single_point(self, p1, p2):
        """
        p1 — точка в эпоху 1, p2 — точка в эпоху 2 (один и тот же объект).
        """
        name = p1.name
        x1 = np.array([p1.x, p1.y, p1.z], dtype=float)
        x2 = np.array([p2.x, p2.y, p2.z], dtype=float)
        delta = x2 - x1
        displacement = float(np.linalg.norm(delta))

        cov1 = getattr(p1, "cov_xyz", None)
        cov2 = getattr(p2, "cov_xyz", None)
        cross_cov = self.cross_cov_map.get(name, np.zeros((3, 3), dtype=float))

        reliable = (cov1 is not None) and (cov2 is not None)

        if not reliable:
            return self._make_unreliable_result(name, delta, displacement)

        cov1 = np.asarray(cov1, dtype=float)
        cov2 = np.asarray(cov2, dtype=float)
        cross_cov = np.asarray(cross_cov, dtype=float)

        cov_delta = cov1 + cov2 - cross_cov - cross_cov.T
        cov_delta = 0.5 * (cov_delta + cov_delta.T)

        return self._compute_tests(name, delta, displacement, cov_delta, reliable=True)

    def _analyze_segment_pair(self, seg1, seg2):
        """
        Анализ смещения по паре одноимённых сегментов.
        Смещение определяется как разность длин с оценкой погрешности.
        """
        name = seg1.name
        delta_length = seg2.length - seg1.length

        reliable = seg1.reliable_accuracy and seg2.reliable_accuracy

        if not reliable:
            delta_vec = np.array([delta_length, 0.0, 0.0])
            return self._make_unreliable_result(name, delta_vec, abs(delta_length))

        # Вектор смещения в системе первого сегмента: вдоль направления
        delta_vec = (seg2.length - seg1.length) * seg1.direction

        # Упрощённая ковариация: только вдоль оси сегмента
        sigma2 = seg1.sigma_length ** 2 + seg2.sigma_length ** 2
        cov_delta = np.outer(seg1.direction, seg1.direction) * sigma2

        return self._compute_tests(name, delta_vec, abs(delta_length), cov_delta, reliable=True)

    def _compute_tests(self, name, delta, displacement, cov_delta, reliable):
        diag = np.maximum(np.diag(cov_delta), 0.0)
        sigma_xyz = np.sqrt(diag)

        # СКП длины смещения
        if displacement > 1e-16:
            g = (delta / displacement).reshape(1, 3)
            var_d = (g @ cov_delta @ g.T).item()
            sigma_displacement = float(np.sqrt(max(var_d, 0.0)))
        else:
            sigma_displacement = float(np.sqrt(np.trace(cov_delta) / 3.0))

        # 1D t-тест
        if sigma_displacement > 1e-16:
            t_value = displacement / sigma_displacement
        else:
            t_value = 0.0

        # двустороннее p-значение (df=∞ → стандартное нормальное)
        p_value_t = float(2.0 * (1.0 - stats.norm.cdf(abs(t_value))))
        significant_t = p_value_t < self.alpha

        # 3D chi^2-тест
        chi2_value = None
        p_value_chi2 = None
        significant_chi2 = None

        try:
            cov_inv = np.linalg.inv(cov_delta)
            chi2_value = float(delta @ cov_inv @ delta)
            p_value_chi2 = float(1.0 - stats.chi2.cdf(chi2_value, df=3))
            significant_chi2 = p_value_chi2 < self.alpha
        except np.linalg.LinAlgError:
            pass

        return DeformationResult(
            name=name,
            delta=delta,
            displacement=displacement,
            sigma_displacement=sigma_displacement,
            cov_delta=cov_delta,
            sigma_dx=float(sigma_xyz[0]),
            sigma_dy=float(sigma_xyz[1]),
            sigma_dz=float(sigma_xyz[2]),
            t_value=t_value,
            p_value_t=p_value_t,
            significant_t=significant_t,
            chi2_value=chi2_value,
            p_value_chi2=p_value_chi2,
            significant_chi2=significant_chi2,
            alpha=self.alpha,
            reliable=reliable,
        )

    @staticmethod
    def _make_unreliable_result(name, delta, displacement):
        return DeformationResult(
            name=name,
            delta=delta,
            displacement=float(displacement),
            sigma_displacement=float("nan"),
            cov_delta=np.full((3, 3), float("nan")),
            sigma_dx=float("nan"),
            sigma_dy=float("nan"),
            sigma_dz=float("nan"),
            t_value=float("nan"),
            p_value_t=float("nan"),
            significant_t=False,
            chi2_value=None,
            p_value_chi2=None,
            significant_chi2=None,
            alpha=0.05,
            reliable=False,
        )

    # ------------------------------------------------------------------
    # Результаты
    # ------------------------------------------------------------------

    @property
    def results(self):
        return list(self._results)

    @property
    def significant_results(self):
        return [r for r in self._results if r.significant_t]

    @property
    def n_significant(self):
        return sum(r.significant_t for r in self._results)

    def to_dataframe(self):
        return pd.DataFrame([r.as_dict() for r in self._results])

    def to_csv(self, file_path, index=False):
        self.to_dataframe().to_csv(file_path, index=index)

    def print_summary(self):
        n = len(self._results)
        n_sig = self.n_significant
        print(f"\n{'='*60}")
        print(f"DeformationAnalyzer: {n} точек, значимых смещений: {n_sig}/{n}")
        print(f"Уровень значимости α = {self.alpha}")
        print(f"{'='*60}")
        for r in sorted(self._results, key=lambda x: x.displacement, reverse=True):
            print(f"  {r}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    from tabulate import tabulate

    from app.cross_points.CrossPointListRestorer import CrossPointListRestorer

    path_epoch1 = "/data/8_floors_wall/output/total_scan/cross_points_good_filtered_by_ellipsoid.csv"
    path_epoch2 = "/data/8_floors_wall/output/scan_2335_filt/cross_points_good_filtered_by_ellipsoid.csv"

    points_e1 = CrossPointListRestorer(path_epoch1).restore_all()
    points_e2 = CrossPointListRestorer(path_epoch2).restore_all()

    analyzer = DeformationAnalyzer(alpha=0.05)
    analyzer.analyze_point_sets(points_e1, points_e2)

    analyzer.print_summary()

    df = analyzer.to_dataframe()
    # print(tabulate(df[["name", "displacement_mm", "sigma_displacement_mm", "t_value", "p_value_t", "significant_t"]], tablefmt="pretty"))
    print(tabulate(df.sort_values("displacement"), tablefmt="pretty", headers="keys") )