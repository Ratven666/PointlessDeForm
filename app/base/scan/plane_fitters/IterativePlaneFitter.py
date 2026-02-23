import numpy as np

from app.base.scan.Scan import Scan
from app.base.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC
from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter


class IterativePlaneFitter(PlaneFitterABC):

    def fit_plane(self, *args, mse_threshold=0.001,
                  max_iteration=20, k_sigma=3, base_fitter=PlaneL1Fitter, **kwargs):
        from app.base.scan.ScanPlane import ScanPlane
        current_scan = self.scan
        current_plane = None
        for _ in range(max_iteration):
            # 1. Оценка плоскости
            current_plane = ScanPlane.fit_plane_to_scan(scan=current_scan,
                                                        fitter=base_fitter,
                                                        args=args, kwargs=kwargs)
            # 2. Критерий остановки по MSE
            if current_plane.mse <= mse_threshold:
                break
            # 3. Отбраковка выбросов
            current_scan = self._filter_outliers_by_k_sigma(current_scan=current_scan,
                                                            current_plane=current_plane,
                                                            k_sigma=k_sigma,
                                                            )
        return current_scan, current_plane.normal, current_plane.point, current_plane.d

    @staticmethod
    def _filter_outliers_by_k_sigma(current_scan, current_plane, k_sigma):
        """
        Отбраковка выбросов по правилу k * sigma:
        оставляем точки с dist <= mean + k * std.
        """
        pts = np.array([[p.x, p.y, p.z] for p in current_scan], dtype=float)
        dists = current_plane.distance_to_point(pts)

        mean = float(np.mean(dists))
        std = float(np.std(dists))
        threshold = mean + k_sigma * std

        filtered_points = [p for p, dist in zip(current_scan, dists) if dist <= threshold]

        f_scan = Scan(scan_name=f"{current_scan.name}_filtered")
        f_scan._points = filtered_points
        f_scan.borders = f_scan._get_borders_dict(f_scan._points)

        return f_scan
