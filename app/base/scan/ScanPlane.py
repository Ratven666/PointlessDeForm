import numpy as np
from loguru import logger

from app.base.Plane import Plane
from app.base.scan.Scan import Scan
from app.base.scan.plane_fitters.IterativePlaneFitter import IterativePlaneFitter
from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.base.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter


class ScanPlane(Plane):
    def __init__(self, normal, point_on_plane, d):
        super().__init__(normal, point_on_plane, d)
        self.scan = None
        self.mse = None

    @classmethod
    def fit_plane_to_scan(cls, scan: Scan, *args, fitter=PlaneLSMFitter, **kwargs):
        fitter_instance = fitter(scan=scan)
        scan, normal, point_on_plane, d = fitter_instance.fit_plane(args, **kwargs)

        scan_plane = cls(normal, point_on_plane, d)
        scan_plane._compute_mse_for_scan(scan=scan)
        scan_plane.scan = scan

        logger.info(
            "Finished fit_plane_to_scan | fitter={} | scan_len={} | mse={:.6f}",
            getattr(fitter, "__name__", str(fitter)),
            len(scan_plane.scan),
            scan_plane.mse,
        )

        return scan_plane

    def _compute_mse_for_scan(self, scan: Scan) -> float:
        """
        Считает среднеквадратическое отклонение (СКО) расстояний
        точек скана от плоскости.
        scan: Scan – скан с точками (ScanPoint/Point с атрибутами x, y, z)
        """
        pts = np.array([[p.x, p.y, p.z] for p in scan], dtype=float)
        dists = self.distance_to_point(pts)  # shape (N,)
        mse = float(np.mean(dists ** 2)) ** 0.5
        self.mse = mse
        return mse

    def __repr__(self):
        A, B, C, D = self.equation
        return (
            f"{self.__class__.__name__} (mse={self.mse:.6f}, "
            f"scan_len={len(self.scan)}, "
            f"A={A:.6f}, B={B:.6f}, C={C:.6f}, D={D:.6f})"
        )


if __name__ == '__main__':
    # logger.add("scan_plane.log", rotation="10 MB")

    scan = Scan("TestPlane")
    scan.import_points_from_file("../../../src/Lt1predobr_label_0_label_14.txt")
    logger.info("Loaded scan | name={} | len={}", scan.name, len(scan))

    print(scan)

    s_plane = ScanPlane.fit_plane_to_scan(scan=scan, fitter=PlaneLSMFitter)
    logger.info("Result LSM plane: {}", s_plane)
    print(s_plane)

    s_plane = ScanPlane.fit_plane_to_scan(scan=scan, fitter=PlaneL1Fitter)
    logger.info("Result L1 plane: {}", s_plane)
    print(s_plane)

    s_plane = ScanPlane.fit_plane_to_scan(
        scan=scan,
        fitter=IterativePlaneFitter,
        base_fitter=PlaneL1Fitter,
        mse_threshold=0.0002,
        max_iteration=10,
        k_sigma=2,
    )
    logger.info("Result Iterative L1 plane: {}", s_plane)
    print(s_plane)

    s_plane = ScanPlane.fit_plane_to_scan(
        scan=scan,
        fitter=IterativePlaneFitter,
        base_fitter=PlaneLSMFitter,
        mse_threshold=0.0002,
        max_iteration=10,
        k_sigma=2,
    )
    logger.info("Result Iterative LSM plane: {}", s_plane)
    print(s_plane)
    # s_plane.scan.plot()
