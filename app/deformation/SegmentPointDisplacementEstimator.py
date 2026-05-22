from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SegmentDisplacementResult:
    """
    Результат оценки смещения одной точки по изменениям сегментов.
    """
    point_name: str
    n_obs: int

    dx: float
    dy: float
    dz: float
    displacement: float

    cov_xyz: np.ndarray
    sigma_x: float
    sigma_y: float
    sigma_z: float
    sigma_displacement: float

    residuals: np.ndarray
    sigma0: float

    t_x: Optional[float]
    t_y: Optional[float]
    t_z: Optional[float]
    t_disp: Optional[float]

    p_x: Optional[float]
    p_y: Optional[float]
    p_z: Optional[float]
    p_disp: Optional[float]

    significant_x: Optional[bool]
    significant_y: Optional[bool]
    significant_z: Optional[bool]
    significant_disp: Optional[bool]

    chi2_value: Optional[float]
    p_chi2: Optional[float]
    significant_chi2: Optional[bool]

    reliable: bool
    message: str

    @property
    def displacement_mm(self):
        return self.displacement * 1000.0

    @property
    def sigma_displacement_mm(self):
        return self.sigma_displacement * 1000.0

    def as_dict(self):
        return {
            "point_name": self.point_name,
            "n_obs": self.n_obs,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "displacement": self.displacement,
            "displacement_mm": self.displacement_mm,
            "sigma_x": self.sigma_x,
            "sigma_y": self.sigma_y,
            "sigma_z": self.sigma_z,
            "sigma_displacement": self.sigma_displacement,
            "sigma_displacement_mm": self.sigma_displacement_mm,
            "sigma0": self.sigma0,
            "t_x": self.t_x,
            "t_y": self.t_y,
            "t_z": self.t_z,
            "t_disp": self.t_disp,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "p_z": self.p_z,
            "p_disp": self.p_disp,
            "significant_x": self.significant_x,
            "significant_y": self.significant_y,
            "significant_z": self.significant_z,
            "significant_disp": self.significant_disp,
            "chi2_value": self.chi2_value,
            "p_chi2": self.p_chi2,
            "significant_chi2": self.significant_chi2,
            "reliable": self.reliable,
            "message": self.message,
        }

    def __str__(self):
        if not self.reliable:
            return (
                f"SegmentDisplacementResult(point={self.point_name}, "
                f"reliable=False, message={self.message})"
            )

        return (
            f"SegmentDisplacementResult(point={self.point_name}, n_obs={self.n_obs}, "
            f"d=({self.dx:.6f}, {self.dy:.6f}, {self.dz:.6f}) m, "
            f"|d|={self.displacement:.6f} m, "
            f"s=({self.sigma_x:.6f}, {self.sigma_y:.6f}, {self.sigma_z:.6f}) m, "
            f"s|d|={self.sigma_displacement:.6f} m, "
            f"chi2={self.chi2_value}, p={self.p_chi2})"
        )


class SegmentPointDisplacementEstimator:
    """
    Оценивает смещение точки по изменениям длин сегментов, выходящих из неё,
    между двумя наборами отрезков.

    Модель:
        Δl_i = u_i^T * dX + v_i

    где:
        Δl_i  = l_i(epoch2) - l_i(epoch1)
        u_i   = единичный вектор направления сегмента в epoch1
        dX    = [dx, dy, dz]^T — искомое смещение точки
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def estimate_for_point(self, point_name, seg_set_epoch1, seg_set_epoch2, use_weights=True):
        segs1 = self._collect_segments_for_point(seg_set_epoch1, point_name)
        segs2 = self._collect_segments_for_point(seg_set_epoch2, point_name)

        if len(segs1) == 0 or len(segs2) == 0:
            return self._empty_result(point_name, "Нет сегментов для данной точки в одной из эпох")

        common_keys = sorted(set(segs1.keys()) & set(segs2.keys()))
        if len(common_keys) < 3:
            return self._empty_result(
                point_name,
                "Недостаточно общих сегментов. Для 3D-оценки нужно минимум 3 независимых направления"
            )

        A_rows = []
        L_rows = []
        P_diag = []

        for key in common_keys:
            seg1 = segs1[key]
            seg2 = segs2[key]

            u = self._get_point_outward_unit_vector(seg1, point_name)
            dl = seg2.length - seg1.length

            A_rows.append(u)
            L_rows.append(dl)

            if use_weights and seg1.sigma_length is not None and seg2.sigma_length is not None:
                var_dl = seg1.sigma_length ** 2 + seg2.sigma_length ** 2
                if var_dl > 1e-16:
                    P_diag.append(1.0 / var_dl)
                else:
                    P_diag.append(1.0)
            else:
                P_diag.append(1.0)

        A = np.asarray(A_rows, dtype=float)
        L = np.asarray(L_rows, dtype=float).reshape(-1, 1)
        P = np.diag(P_diag)

        rank = np.linalg.matrix_rank(A)
        if rank < 3:
            return self._empty_result(
                point_name,
                "Матрица коэффициентов вырождена: направления сегментов не обеспечивают 3D-решение"
            )

        N = A.T @ P @ A
        Qxx = np.linalg.inv(N)
        X = Qxx @ A.T @ P @ L
        V = A @ X - L

        n = A.shape[0]
        u = 3
        dof = n - u

        if dof > 0:
            sigma0_sq = float((V.T @ P @ V).item() / dof)
        else:
            sigma0_sq = float((V.T @ P @ V).item())

        cov_xyz = sigma0_sq * Qxx
        cov_xyz = 0.5 * (cov_xyz + cov_xyz.T)

        dx, dy, dz = X.flatten()
        disp_vec = np.array([dx, dy, dz], dtype=float)
        displacement = float(np.linalg.norm(disp_vec))

        sigma_x = float(np.sqrt(max(cov_xyz[0, 0], 0.0)))
        sigma_y = float(np.sqrt(max(cov_xyz[1, 1], 0.0)))
        sigma_z = float(np.sqrt(max(cov_xyz[2, 2], 0.0)))

        if displacement > 1e-16:
            g = (disp_vec / displacement).reshape(1, 3)
            sigma_disp_sq = float((g @ cov_xyz @ g.T).item())
            sigma_displacement = float(np.sqrt(max(sigma_disp_sq, 0.0)))
        else:
            sigma_displacement = 0.0

        t_x, p_x, sig_x = self._calc_t_test(dx, sigma_x)
        t_y, p_y, sig_y = self._calc_t_test(dy, sigma_y)
        t_z, p_z, sig_z = self._calc_t_test(dz, sigma_z)
        t_disp, p_disp, sig_disp = self._calc_t_test(displacement, sigma_displacement)

        chi2_value = None
        p_chi2 = None
        sig_chi2 = None
        try:
            chi2_value = float(disp_vec.T @ np.linalg.inv(cov_xyz) @ disp_vec)
            p_chi2 = float(1.0 - stats.chi2.cdf(chi2_value, df=3))
            sig_chi2 = p_chi2 < self.alpha
        except np.linalg.LinAlgError:
            pass

        return SegmentDisplacementResult(
            point_name=point_name,
            n_obs=n,
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            displacement=displacement,
            cov_xyz=cov_xyz,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            sigma_displacement=sigma_displacement,
            residuals=V.flatten(),
            sigma0=float(np.sqrt(max(sigma0_sq, 0.0))),
            t_x=t_x,
            t_y=t_y,
            t_z=t_z,
            t_disp=t_disp,
            p_x=p_x,
            p_y=p_y,
            p_z=p_z,
            p_disp=p_disp,
            significant_x=sig_x,
            significant_y=sig_y,
            significant_z=sig_z,
            significant_disp=sig_disp,
            chi2_value=chi2_value,
            p_chi2=p_chi2,
            significant_chi2=sig_chi2,
            reliable=True,
            message="ok",
        )

    def estimate_for_all_points(self, seg_set_epoch1, seg_set_epoch2, use_weights=True):
        point_names = sorted(
            self._extract_all_point_names(seg_set_epoch1) & self._extract_all_point_names(seg_set_epoch2)
        )

        results = []
        for point_name in point_names:
            result = self.estimate_for_point(
                point_name=point_name,
                seg_set_epoch1=seg_set_epoch1,
                seg_set_epoch2=seg_set_epoch2,
                use_weights=use_weights,
            )
            results.append(result)

        return results

    @staticmethod
    def results_to_dataframe(results):
        return pd.DataFrame([r.as_dict() for r in results])

    @staticmethod
    def _collect_segments_for_point(seg_set, point_name):
        out = {}
        for seg in seg_set:
            if seg.p1.name == point_name:
                other = seg.p2.name
                out[other] = seg
            elif seg.p2.name == point_name:
                other = seg.p1.name
                out[other] = seg
        return out

    @staticmethod
    def _get_point_outward_unit_vector(seg, point_name):
        """
        Возвращает единичный вектор направления от оцениваемой точки к соседней.
        """
        if seg.p1.name == point_name:
            return np.array(seg.direction, dtype=float)
        elif seg.p2.name == point_name:
            return -np.array(seg.direction, dtype=float)
        raise ValueError(f"Сегмент {seg.name} не связан с точкой {point_name}")

    @staticmethod
    def _extract_all_point_names(seg_set):
        names = set()
        for seg in seg_set:
            names.add(seg.p1.name)
            names.add(seg.p2.name)
        return names

    def _calc_t_test(self, value, sigma):
        if sigma is None or not np.isfinite(sigma) or sigma <= 1e-16:
            return None, None, None

        t_value = float(abs(value) / sigma)
        p_value = float(2.0 * (1.0 - stats.norm.cdf(t_value)))
        significant = p_value < self.alpha
        return t_value, p_value, significant

    def _empty_result(self, point_name, message):
        nan3 = np.full((3, 3), np.nan, dtype=float)
        return SegmentDisplacementResult(
            point_name=point_name,
            n_obs=0,
            dx=np.nan,
            dy=np.nan,
            dz=np.nan,
            displacement=np.nan,
            cov_xyz=nan3,
            sigma_x=np.nan,
            sigma_y=np.nan,
            sigma_z=np.nan,
            sigma_displacement=np.nan,
            residuals=np.array([], dtype=float),
            sigma0=np.nan,
            t_x=None,
            t_y=None,
            t_z=None,
            t_disp=None,
            p_x=None,
            p_y=None,
            p_z=None,
            p_disp=None,
            significant_x=None,
            significant_y=None,
            significant_z=None,
            significant_disp=None,
            chi2_value=None,
            p_chi2=None,
            significant_chi2=None,
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

    estimator = SegmentPointDisplacementEstimator(alpha=0.05)

    result = estimator.estimate_for_point("7_7_vl", seg_set_1, seg_set_2)
    print(result)

    results = estimator.estimate_for_all_points(seg_set_1, seg_set_2)
    df = estimator.results_to_dataframe(results)
    print(tabulate(df[[
        "point_name", "n_obs", "dx", "dy", "dz",
        "displacement_mm", "sigma_displacement_mm",
        "p_disp", "significant_disp", "p_chi2", "significant_chi2"
    ]], headers=["point_name", "n_obs", "dx", "dy", "dz",
        "displacement_mm", "sigma_displacement_mm",
        "p_disp", "significant_disp", "p_chi2", "significant_chi2"]))