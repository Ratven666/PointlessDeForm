import numpy as np
from loguru import logger

from app.base.Plane import Plane
from app.base.scan.Scan import Scan
from app.base.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter


class ScanPlane(Plane):
    """
    Плоскость, подогнанная к облаку точек.

    Атрибуты точности (заполняются только если фиттер вернул cov_params):
        cov_params  – np.ndarray (4,4): ковариация параметров (A,B,C,D)
        sigma0      – float: оценка СКП единицы веса (из фиттера)
        mse         – float: RMSE расстояний точек до плоскости
    """

    def __init__(self, normal, point_on_plane, d):
        super().__init__(normal, point_on_plane, d)
        self.scan: Scan | None = None
        self.mse: float | None = None
        self.cov_params: np.ndarray | None = None   # (4,4) ковариация (A,B,C,D)
        self.sigma0: float | None = None            # СКП единицы веса

    # ------------------------------------------------------------------
    # Фабричный метод
    # ------------------------------------------------------------------
    @classmethod
    def fit_plane_to_scan(cls, scan: Scan, *args, fitter=PlaneLSMFitter, **kwargs):
        fitter_instance = fitter(scan=scan)
        result = fitter_instance.fit_plane(*args, **kwargs)
        scan_out, normal, point_on_plane, d = result

        scan_plane = cls(normal, point_on_plane, d)
        scan_plane._compute_mse_for_scan(scan=scan_out)
        scan_plane.scan = scan_out

        # Забираем ковариацию и sigma0, если фиттер их посчитал
        scan_plane.cov_params = getattr(fitter_instance, "cov_params", None)
        scan_plane.sigma0 = getattr(fitter_instance, "sigma0", None)

        logger.info(
            "Finished fit_plane_to_scan | fitter={} | scan_len={} | mse={:.6f}",
            getattr(fitter, "__name__", str(fitter)),
            len(scan_plane.scan),
            scan_plane.mse,
        )
        return scan_plane

    # ------------------------------------------------------------------
    # Точность плоскости
    # ------------------------------------------------------------------
    def _compute_mse_for_scan(self, scan: Scan) -> float:
        """RMSE расстояний точек скана до плоскости."""
        pts = np.array([[p.x, p.y, p.z] for p in scan], dtype=float)
        dists = self.distance_to_point(pts)
        mse = float(np.sqrt(np.mean(dists ** 2)))
        self.mse = mse
        return mse

    def has_covariance(self) -> bool:
        """True, если ковариация параметров доступна."""
        return self.cov_params is not None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self):
        A, B, C, D = self.equation

        cov_str = "None"
        if self.cov_params is not None:
            cov_str = np.array2string(
                self.cov_params,
                precision=6,
                suppress_small=True
            )
        sigma0_str = "None" if self.sigma0 is None else f"{self.sigma0:.6f}"
        return (
            f"{self.__class__.__name__} (\n"
            f"  mse={self.mse:.6f}, sigma0={sigma0_str}, scan_len={len(self.scan)},\n"
            f"  A={A:.6f}, B={B:.6f}, C={C:.6f}, D={D:.6f},\n"
            f"  cov_params=\n{cov_str}\n"
            f")"
        )
