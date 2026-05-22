import numpy as np

from app.base.NamedPoint import NamedPoint


class CrossPoint(NamedPoint):
    """
    Точка пересечения трёх плоскостей.

    Атрибуты точности:
        mse               – скалярная обобщённая СКП
        planes_mse        – список MSE трёх плоскостей
        sigma_xyz         – np.ndarray (3,): СКП по X, Y, Z
        cov_xyz           – np.ndarray (3,3): ковариационная матрица координат
        ellipsoid         – dict: полуоси и направления эллипсоида погрешности
        reliable_accuracy – bool: можно ли доверять числовой оценке точности
    """

    def __init__(self, name, x, y, z=0):
        super().__init__(name, x, y, z)
        self.status: str | None = None
        self.mse: float | None = None
        self.planes_mse: list[float] | None = None

        self.sigma_xyz: np.ndarray | None = None
        self.cov_xyz: np.ndarray | None = None
        self.ellipsoid: dict | None = None
        self.reliable_accuracy: bool = True

    def load_mses(self, plane_mses: list[float]):
        self.planes_mse = plane_mses
        self.mse = float(sum(m ** 2 for m in plane_mses) ** 0.5)

    def load_covariance(self, cov_xyz: np.ndarray, confidence: float = 0.95):
        from scipy.stats import chi2

        self.cov_xyz = np.asarray(cov_xyz, dtype=float)
        self.sigma_xyz = np.sqrt(np.maximum(np.diag(self.cov_xyz), 0.0))
        self.mse = float(np.sqrt(np.trace(self.cov_xyz)))

        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_xyz)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k = chi2.ppf(confidence, df=3)
        semi_axes = np.sqrt(np.maximum(eigenvalues, 0.0) * k)

        self.ellipsoid = {
            "semi_axes": semi_axes,
            "directions": eigenvectors,
            "confidence": confidence,
        }
        self.reliable_accuracy = True

    def mark_unreliable_accuracy(self):
        self.reliable_accuracy = False
        self.sigma_xyz = None
        self.cov_xyz = None
        self.ellipsoid = None

    def __str__(self):
        parts = [
            f"{self.__class__.__name__} (name={self.name}, status={self.status}",
            f"x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f}",
        ]

        if self.planes_mse is not None:
            parts.append(f"plane_mses={[round(m, 6) for m in self.planes_mse]}")

        if self.reliable_accuracy:
            if self.mse is not None:
                parts.append(f"mse={self.mse:.6f}")

            if self.sigma_xyz is not None:
                sx, sy, sz = self.sigma_xyz
                parts.append(f"sigma_xyz=({sx:.6f}, {sy:.6f}, {sz:.6f})")

            if self.cov_xyz is not None:
                parts.append(
                    "cov_xyz=\n" + np.array2string(
                        self.cov_xyz,
                        precision=6,
                        suppress_small=True,
                    )
                )

            if self.ellipsoid is not None:
                a, b, c = self.ellipsoid["semi_axes"]
                parts.append(f"ellipsoid_axes=({a:.6f}, {b:.6f}, {c:.6f})")
        else:
            parts.append("accuracy=UNRELIABLE")
            if self.mse is not None:
                parts.append(f"plane_mse_total={self.mse:.6f}")

        return ", ".join(parts) + ")"

    def __repr__(self):
        acc_info = "acc=ok" if self.reliable_accuracy else "acc=unreliable"
        return (
            f"({self.name}, status={self.status}, "
            f"{self.x:.3f}, {self.y:.3f}, {self.z:.3f}, "
            f"mse={self.mse:.5f}, {acc_info})"
        )
