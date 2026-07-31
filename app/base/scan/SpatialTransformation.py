from __future__ import annotations

import numpy as np


class SpatialTransformation:
    """
    Матрица пространственных трансформаций (жёсткое тело, без масштаба).

    Параметризация:
    ---------------
    X' = R @ X + t

    Атрибуты:
        R          – матрица вращения (3×3)
        t          – вектор трансляции (3,)
        T          – однородная матрица (4×4): [[R | t], [0 | 1]]
        T_matrix   – alias для T (совместимость)
        method     – метод оценки: 'LSM' или 'L1'
        n_common   – число общих точек
        n_used     – число точек, использованных в оценке
        residuals  – вектор (n,): ортогональные расстояния после трансформации
        rmse       – RMSE остатков (м)
        mae        – MAE  остатков (м)
        max_res    – максимальный остаток (м)
        omega, phi, kappa – углы вращения (рад): Rx, Ry, Rz
        tx, ty, tz        – трансляции
    """

    def __init__(self,
                 R: np.ndarray,
                 t: np.ndarray,
                 method: str,
                 n_common: int,
                 n_used: int,
                 residuals: np.ndarray):
        self.R = np.asarray(R, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.method = method
        self.n_common = n_common
        self.n_used = n_used
        self.residuals = np.asarray(residuals, dtype=float)

        self.T = self._build_T(self.R, self.t)

        self.rmse = float(np.sqrt(np.mean(self.residuals ** 2)))
        self.mae = float(np.mean(np.abs(self.residuals)))
        self.max_res = float(np.max(np.abs(self.residuals)))

        self.omega, self.phi, self.kappa = self._rotation_to_angles(self.R)
        self.tx, self.ty, self.tz = self.t

    # ------------------------------------------------------------------
    @property
    def T_matrix(self) -> np.ndarray:
        """Alias для T — однородная матрица 4×4 (обратная совместимость)."""
        return self.T

    # ------------------------------------------------------------------
    @staticmethod
    def _build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def _rotation_to_angles(R: np.ndarray):
        """Euler ZYX -> omega(Rx), phi(Ry), kappa(Rz) в радианах."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            omega = np.arctan2(R[2, 1], R[2, 2])
            phi = np.arctan2(-R[2, 0], sy)
            kappa = np.arctan2(R[1, 0], R[0, 0])
        else:
            omega = np.arctan2(-R[1, 2], R[1, 1])
            phi = np.arctan2(-R[2, 0], sy)
            kappa = 0.0
        return float(omega), float(phi), float(kappa)

    # ------------------------------------------------------------------
    def transform_point(self, xyz: np.ndarray) -> np.ndarray:
        """Применяет трансформацию к одной точке или массиву (N,3)."""
        xyz = np.asarray(xyz, dtype=float)
        return (self.R @ xyz.T).T + self.t

    def transform_points(self, points: list) -> list:
        """
        Принимает list[CrossPoint], возвращает новый list[CrossPoint]
        с трансформированными координатами и перенесённой ковариацией.
        """
        from app.cross_points.CrossPoint import CrossPoint

        result = []
        for p in points:
            xyz = np.array([p.x, p.y, p.z], dtype=float)
            xyz_new = self.R @ xyz + self.t

            new_p = CrossPoint(
                name=p.name,
                x=float(xyz_new[0]),
                y=float(xyz_new[1]),
                z=float(xyz_new[2]),
            )

            new_p.status = getattr(p, "status", None)
            new_p.planes_mse = getattr(p, "planes_mse", None)
            new_p.reliable_accuracy = getattr(p, "reliable_accuracy", True)

            old_cov = getattr(p, "cov_xyz", None)
            if old_cov is not None and getattr(p, "reliable_accuracy", True):
                old_cov = np.asarray(old_cov, dtype=float)
                new_cov = self.R @ old_cov @ self.R.T

                old_conf = 0.95
                if getattr(p, "ellipsoid", None) is not None:
                    old_conf = p.ellipsoid.get("confidence", 0.95)

                new_p.load_covariance(new_cov, confidence=old_conf)
                new_p.planes_mse = getattr(p, "planes_mse", None)
            else:
                new_p.mse = getattr(p, "mse", None)
                if not getattr(p, "reliable_accuracy", True):
                    new_p.mark_unreliable_accuracy()

            result.append(new_p)

        return result

    def transform_scan(self, scan, inplace=False, rotate_normals=True, scan_name=None):
        return scan.transform_scan(
            transformation=self,
            inplace=inplace,
            rotate_normals=rotate_normals,
            scan_name=scan_name,
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Сериализует трансформацию в словарь (для JSON/pickle)."""
        return {
            "method":       self.method,
            "n_common":     self.n_common,
            "n_used":       self.n_used,
            "rmse_m":       self.rmse,
            "mae_m":        self.mae,
            "max_res_m":    self.max_res,
            "rotation_deg": {
                "omega": float(np.rad2deg(self.omega)),
                "phi":   float(np.rad2deg(self.phi)),
                "kappa": float(np.rad2deg(self.kappa)),
            },
            "translation_mm": {
                "tx": float(self.tx * 1000),
                "ty": float(self.ty * 1000),
                "tz": float(self.tz * 1000),
            },
            "R_matrix": self.R.tolist(),
            "T_matrix": self.T.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpatialTransformation:
        """Восстанавливает объект из словаря (например, загруженного из JSON)."""
        T = np.asarray(d["T_matrix"], dtype=float)
        R = T[:3, :3]
        t = T[:3, 3]
        # residuals не сохраняются — восстанавливаем нулевой вектор нужной длины
        n_used = d.get("n_used", 1)
        rmse = d.get("rmse_m", 0.0)
        fake_residuals = np.full(n_used, rmse)
        return cls(R=R, t=t, method=d["method"],
                   n_common=d.get("n_common", n_used),
                   n_used=n_used,
                   residuals=fake_residuals)

    # ------------------------------------------------------------------
    def __str__(self):
        lines = [
            f"SpatialTransformation (method={self.method})",
            f"  n_common={self.n_common}, n_used={self.n_used}",
            f"  RMSE={self.rmse:.6f} m  MAE={self.mae:.6f} m  max={self.max_res:.6f} m",
            "  Translation (m):",
            f"    tx={self.tx:.6f}  ty={self.ty:.6f}  tz={self.tz:.6f}",
            "  Rotation (deg):",
            f"    omega={np.rad2deg(self.omega):.6f}  "
            f"phi={np.rad2deg(self.phi):.6f}  "
            f"kappa={np.rad2deg(self.kappa):.6f}",
            "  Rotation matrix R:",
            np.array2string(self.R, precision=9, suppress_small=True),
            "  Full T (4×4):",
            np.array2string(self.T, precision=9, suppress_small=True),
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (f"SpatialTransformation(method={self.method!r}, "
                f"n_used={self.n_used}, rmse={self.rmse:.6f})")