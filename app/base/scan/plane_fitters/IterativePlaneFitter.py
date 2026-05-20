import numpy as np

from app.base.scan.Scan import Scan
from app.base.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC
from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.base.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter


class IterativePlaneFitter(PlaneFitterABC):
    """
    Итерационный фиттер плоскости:
    1) оценивает плоскость робастным базовым фиттером,
    2) отбраковывает выбросы по правилу mean + k_sigma * std,
    3) повторяет до достижения mse_threshold или max_iteration,
    4) после завершения ОБЯЗАТЕЛЬНО выполняет финальную подгонку по обычному МНК
       на очищенном наборе точек.

    Зачем нужен финальный МНК:
    -------------------------
    Робастный/итерационный фиттер хорошо очищает выборку от выбросов, но его
    оценка ковариации параметров плоскости обычно не является строгой и может
    быть нестабильной. Поэтому после очистки данных мы строим окончательную
    плоскость через PlaneLSMFitter на финальном подмножестве точек.

    Именно от этой финальной LSM-плоскости наружу пробрасываются:
        - cov_params
        - sigma0
        - mse

    Таким образом достигается схема:
        robust trimming -> clean inliers -> final LSM fit -> reliable covariance
    """

    def __init__(self, scan: Scan):
        super().__init__(scan)
        self.cov_params = None
        self.sigma0 = None
        self.final_plane = None
        self.filtered_scan = None

    def fit_plane(
        self,
        *args,
        mse_threshold=0.001,
        max_iteration=20,
        k_sigma=3,
        base_fitter=PlaneL1Fitter,
        final_fitter=PlaneLSMFitter,
        min_points=6,
        **kwargs,
    ):
        from app.base.scan.ScanPlane import ScanPlane

        current_scan = self.scan
        robust_plane = None

        for _ in range(max_iteration):
            if len(current_scan) < 3:
                raise RuntimeError("После фильтрации осталось меньше 3 точек")

            robust_plane = ScanPlane.fit_plane_to_scan(
                scan=current_scan,
                fitter=base_fitter,
                *args,
                **kwargs,
            )

            if robust_plane.mse <= mse_threshold:
                break

            next_scan = self._filter_outliers_by_k_sigma(
                current_scan=current_scan,
                current_plane=robust_plane,
                k_sigma=k_sigma,
            )

            if len(next_scan) < min_points:
                break

            if len(next_scan) == len(current_scan):
                current_scan = next_scan
                break

            current_scan = next_scan

        if robust_plane is None:
            raise RuntimeError("Не удалось оценить плоскость: robust_plane is None")

        self.filtered_scan = current_scan

        final_plane = ScanPlane.fit_plane_to_scan(
            scan=current_scan,
            fitter=final_fitter,
            *args,
            **kwargs,
        )

        self.final_plane = final_plane
        self.cov_params = getattr(final_plane, "cov_params", None)
        self.sigma0 = getattr(final_plane, "sigma0", None)
        self.mse = getattr(final_plane, "mse", None)

        return current_scan, final_plane.normal, final_plane.point, final_plane.d

    @staticmethod
    def _filter_outliers_by_k_sigma(current_scan, current_plane, k_sigma):
        """
        Отбраковка выбросов по правилу:
            dist <= mean + k_sigma * std

        Важно:
        dist берётся как ортогональное расстояние до текущей робастной плоскости.
        """
        pts = np.array([[p.x, p.y, p.z] for p in current_scan], dtype=float)
        dists = current_plane.distance_to_point(pts)

        mean = float(np.mean(dists))
        std = float(np.std(dists))
        threshold = mean + k_sigma * std

        filtered_points = [
            p for p, dist in zip(current_scan, dists)
            if dist <= threshold
        ]

        f_scan = Scan(scan_name=f"{current_scan.name}_filtered")
        f_scan._points = filtered_points
        f_scan.borders = f_scan._get_borders_dict(f_scan._points)

        return f_scan
