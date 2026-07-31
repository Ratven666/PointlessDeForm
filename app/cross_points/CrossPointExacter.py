import os
import logging

import numpy as np

from app.base.scan.Scan import Scan
from app.base.scan.ScanPlane import ScanPlane
from app.base.scan.filters.ScanFilterByDBSCAN import ScanFilterByDBSCAN
from app.base.scan.plane_fitters.IterativePlaneFitter import IterativePlaneFitter
from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.base.scan.plotters.ScanPlotterWithLabelsMPL import ScanPlotterWithLabelsMPL
from app.base.scan.utils.ScanNormalsDirectionClassifier import ScanNormalsDirectionClassifier
from app.base.scan.utils.ScanSplitterByLabels import ScanSplitterByLabels
from app.cross_points.CrossPoint import CrossPoint

logger = logging.getLogger(__name__)

COND_THRESHOLD = 1_000.0
PARALLEL_ANGLE_TOL = np.deg2rad(10.0)


class PlaneGeometryStatus:
    GOOD = "GOOD"
    PARALLEL = "PARALLEL"
    ILL_CONDITIONED = "ILL_CONDITIONED"
    SINGULAR = "SINGULAR"


class PlaneGeometryDiagnostics:
    def __init__(self, planes,
                 cond_threshold: float = COND_THRESHOLD,
                 angle_tol_rad: float = PARALLEL_ANGLE_TOL):
        self.cond_threshold = cond_threshold
        self.angle_tol_rad = angle_tol_rad

        self.N = np.array([[p.A, p.B, p.C] for p in planes], dtype=float)
        self.det = float(np.linalg.det(self.N))
        _, s, _ = np.linalg.svd(self.N)
        self.singular_values = s
        self.cond = float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf")
        self.has_parallel = self._check_parallel(planes)
        self.messages: list[str] = []
        self.status = self._evaluate()

    def _check_parallel(self, planes) -> bool:
        normals = np.array([p.normal for p in planes], dtype=float)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 1e-15, norms, 1.0)
        cos_tol = np.cos(self.angle_tol_rad)
        n = len(normals)

        for i in range(n):
            for j in range(i + 1, n):
                if abs(np.dot(normals[i], normals[j])) >= cos_tol:
                    return True
        return False

    def _evaluate(self) -> str:
        if abs(self.det) < 1e-10:
            self.messages.append(
                f"det(N)={self.det:.3e} — матрица нормалей вырождена"
            )
            return PlaneGeometryStatus.SINGULAR

        if self.has_parallel:
            self.messages.append(
                f"Обнаружены почти параллельные плоскости (допуск {np.rad2deg(self.angle_tol_rad):.1f}°)"
            )
            return PlaneGeometryStatus.PARALLEL

        if self.cond > self.cond_threshold:
            self.messages.append(
                f"cond(N)={self.cond:.1f} > {self.cond_threshold:.0f} — геометрия плохо обусловлена"
            )
            return PlaneGeometryStatus.ILL_CONDITIONED

        self.messages.append(
            f"det(N)={self.det:.4f}, cond(N)={self.cond:.1f} — геометрия устойчива"
        )
        return PlaneGeometryStatus.GOOD

    @property
    def is_reliable(self) -> bool:
        return self.status == PlaneGeometryStatus.GOOD

    def __str__(self):
        lines = [
            "PlaneGeometryDiagnostics:",
            f"  status          = {self.status}",
            f"  det(N)          = {self.det:.6f}",
            f"  cond(N)         = {self.cond:.2f}",
            f"  singular_values = {self.singular_values}",
            f"  has_parallel    = {self.has_parallel}",
        ]
        for msg in self.messages:
            lines.append(f"  [!] {msg}")
        return "\n".join(lines)


class CrossPointExacter:
    def __init__(self, file_path: str,
                 choose_scan_directly_from_dbscan: bool = True,
                 show_scans=True,
                 labels=None,
                 eps=0.01):
        self.base_scan = self.__init_scan(file_path)
        self.choose_scan_directly_from_dbscan = choose_scan_directly_from_dbscan
        self.plane_scans = self.__separate_plane_scans(show_scans=show_scans,
                                                       labels=labels,
                                                       eps=eps)
        self.planes = None
        self.cross_point = None
        self.geometry_diagnostics: PlaneGeometryDiagnostics | None = None

    @staticmethod
    def __init_scan(file_path: str):
        scan_name = os.path.basename(file_path).split(".")[0]
        scan = Scan(scan_name)
        scan.import_points_from_file(file_path)
        scan.compute_normals(k=8)
        s_normals_c = ScanNormalsDirectionClassifier(scan)
        s_normals_c.classify_normals(n_classes=3, unify_hemisphere=True)
        return scan

    def __separate_plane_scans(self, show_scans, eps, labels=None):
        def choose_plane_scan(scan, label=None, eps=eps, min_samples=5, min_cluster_size=100):
            scan.filter_scan(filter_cls=ScanFilterByDBSCAN,
                             eps=eps,
                             min_samples=min_samples,
                             min_cluster_size=min_cluster_size)
            if show_scans:
                scan.plot(plotter=ScanPlotterWithLabelsMPL)
            scans = ScanSplitterByLabels(scan).split()
            if not scans:
                raise ValueError(f"После DBSCAN не осталось кластеров в скане '{scan.name}'")
            if label is None:
                # Автоматически выбираем крупнейший кластер
                label = max(scans, key=lambda lbl: len(list(scans[lbl])))
                logger.debug("Авто-выбор кластера %s для скана '%s'", label, scan.name)
            if label not in scans:
                raise KeyError(
                    f"Кластер {label} не найден в скане '{scan.name}'. "
                    f"Доступные: {sorted(scans.keys())}"
                )
            return scans[label]

        scans = ScanSplitterByLabels(self.base_scan).split()
        plane_scan = []
        for idx, scan in enumerate(scans.values()):
            label = labels[idx] if labels is not None else None
            if self.choose_scan_directly_from_dbscan:
                scan = choose_plane_scan(scan, label=label)
            plane_scan.append(scan)
            if show_scans:
                scan.plot()
        return plane_scan

    def calculate_planes(self,
                         base_fitter=PlaneL1Fitter,
                         mse_threshold=0.0001,
                         max_iteration=20,
                         k_sigma=2):
        scan_planes = []
        for scan in self.plane_scans:
            scan_plane = ScanPlane.fit_plane_to_scan(
                scan=scan,
                fitter=IterativePlaneFitter,
                base_fitter=base_fitter,
                mse_threshold=mse_threshold,
                max_iteration=max_iteration,
                k_sigma=k_sigma,
            )
            scan_planes.append(scan_plane)
        self.planes = scan_planes
        return scan_planes

    def diagnose_geometry(self,
                          cond_threshold: float = COND_THRESHOLD,
                          angle_tol_rad: float = PARALLEL_ANGLE_TOL) -> PlaneGeometryDiagnostics:
        diag = PlaneGeometryDiagnostics(
            self.planes,
            cond_threshold=cond_threshold,
            angle_tol_rad=angle_tol_rad,
        )
        self.geometry_diagnostics = diag
        logger.info("PlaneGeometryDiagnostics | %s | det=%.4f | cond=%.1f",
                    diag.status, diag.det, diag.cond)
        return diag

    @staticmethod
    def _fallback_cov_from_mse(plane: ScanPlane) -> np.ndarray:
        sigma2 = float(plane.mse ** 2)
        return np.eye(4, dtype=float) * sigma2

    @staticmethod
    def _propagate_covariance(planes) -> np.ndarray:
        N_mat = np.array([[p.A, p.B, p.C] for p in planes], dtype=float)
        d_vec = np.array([p.D for p in planes], dtype=float)

        xyz = np.linalg.solve(N_mat, -d_vec)
        X, Y, Z = xyz
        M = np.linalg.inv(N_mat)

        Sigma_p = np.zeros((12, 12), dtype=float)
        for i, plane in enumerate(planes):
            cov_i = (plane.cov_params
                     if getattr(plane, "cov_params", None) is not None
                     else CrossPointExacter._fallback_cov_from_mse(plane))
            Sigma_p[4 * i:4 * i + 4, 4 * i:4 * i + 4] = cov_i

        J = np.zeros((3, 12), dtype=float)
        for i in range(3):
            col = M[:, i]
            j0 = 4 * i
            J[:, j0 + 0] = -X * col
            J[:, j0 + 1] = -Y * col
            J[:, j0 + 2] = -Z * col
            J[:, j0 + 3] = -col

        return J @ Sigma_p @ J.T

    def calculate_intersect_point(self,
                                  cond_threshold: float = COND_THRESHOLD,
                                  angle_tol_rad: float = PARALLEL_ANGLE_TOL) -> CrossPoint:
        diag = self.diagnose_geometry(
            cond_threshold=cond_threshold,
            angle_tol_rad=angle_tol_rad,
        )

        A_mat = np.array([[p.A, p.B, p.C] for p in self.planes], dtype=float)
        b_vec = np.array([-p.D for p in self.planes], dtype=float)

        if diag.status == PlaneGeometryStatus.SINGULAR:
            raise ValueError(
                f"Плоскости не имеют единственной точки пересечения: {diag.messages[0]}"
            )

        x = np.linalg.solve(A_mat, b_vec)

        self.cross_point = CrossPoint(
            name=self.base_scan.name,
            x=float(x[0]),
            y=float(x[1]),
            z=float(x[2]),
        )
        self.cross_point.status = diag.status

        plane_mses = [plane.mse for plane in self.planes]
        self.cross_point.load_mses(plane_mses=plane_mses)

        if diag.is_reliable:
            cov_xyz = self._propagate_covariance(self.planes)
            self.cross_point.load_covariance(cov_xyz)
        else:
            self.cross_point.mark_unreliable_accuracy()
            logger.warning(
                "Ковариация точки %s не вычислена: %s | cond(N)=%.1f",
                self.base_scan.name, diag.status, diag.cond,
            )

        return self.cross_point

    def get_result_str(self) -> str:
        cp = self.cross_point
        parts = [
            self.base_scan.name,
            f"x={cp.x:.6f}",
            f"y={cp.y:.6f}",
            f"z={cp.z:.6f}",
            f"status={cp.status}",
        ]

        if cp.reliable_accuracy and cp.sigma_xyz is not None:
            sx, sy, sz = cp.sigma_xyz
            parts.append(f"sx={sx:.6f}")
            parts.append(f"sy={sy:.6f}")
            parts.append(f"sz={sz:.6f}")
        else:
            parts.append("accuracy=UNRELIABLE")

        if self.geometry_diagnostics is not None:
            diag = self.geometry_diagnostics
            parts.append(f"cond(N)={diag.cond:.1f}")
            parts.append(f"det(N)={diag.det:.6f}")

        return ", ".join(parts)


if __name__ == "__main__":
    base_path = "../../data/200226/сканер/lazpredobr"
    file_path = os.path.join(base_path, "Lt1predobr.las")

    cpe = CrossPointExacter(file_path, labels=[14, 5, 2], show_scans=False)
    cpe.calculate_planes()

    for plane in cpe.planes:
        print(plane)

    print("\n--- Диагностика геометрии ---")
    print(cpe.diagnose_geometry())

    print("\n--- Точка пересечения ---")
    point = cpe.calculate_intersect_point()
    print(point)
    print(cpe.get_result_str())
