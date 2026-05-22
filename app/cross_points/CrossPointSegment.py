import numpy as np


class CrossPointSegment:
    """
    Отрезок между двумя CrossPoint с оценкой точности длины и направления.

    Параметры:
        p1, p2       – объекты CrossPoint
        cross_cov_12 – взаимная ковариация точек Cov(X1, X2), np.ndarray (3,3) или None

    Геометрия:
        delta        – вектор p2 - p1
        length       – длина отрезка
        direction    – единичный вектор направления

    Точность:
        cov_delta         – ковариация вектора delta
        sigma_xyz         – СКП компонент delta
        sigma_length      – СКП длины
        cov_direction     – ковариация единичного вектора направления
        azimuth           – азимут в XY, рад
        elevation         – угол возвышения, рад
        sigma_azimuth     – СКП азимута, рад
        sigma_elevation   – СКП угла возвышения, рад
        reliable_accuracy – можно ли доверять оценке точности
    """

    def __init__(self, p1, p2, cross_cov_12=None):
        self.p1 = p1
        self.p2 = p2
        self.name = f"{p1.name}-{p2.name}"

        self.cross_cov_12 = None if cross_cov_12 is None else np.asarray(cross_cov_12, dtype=float)

        self.delta = None
        self.length = None
        self.direction = None

        self.cov_delta = None
        self.sigma_xyz = None
        self.sigma_length = None
        self.cov_direction = None

        self.azimuth = None
        self.elevation = None
        self.sigma_azimuth = None
        self.sigma_elevation = None

        self.reliable_accuracy = True

        self._validate_inputs()
        self._compute_geometry()
        self._compute_accuracy()

    # ------------------------------------------------------------------
    # Проверки
    # ------------------------------------------------------------------
    def _validate_inputs(self):
        for idx, p in enumerate((self.p1, self.p2), start=1):
            for attr in ("x", "y", "z", "name"):
                if not hasattr(p, attr):
                    raise AttributeError(f"У точки p{idx} отсутствует атрибут '{attr}'")

        if self.cross_cov_12 is not None:
            if self.cross_cov_12.shape != (3, 3):
                raise ValueError(
                    f"cross_cov_12 должна иметь форму (3,3), получено {self.cross_cov_12.shape}"
                )

    # ------------------------------------------------------------------
    # Геометрия
    # ------------------------------------------------------------------
    def _compute_geometry(self):
        x1 = np.array([self.p1.x, self.p1.y, self.p1.z], dtype=float)
        x2 = np.array([self.p2.x, self.p2.y, self.p2.z], dtype=float)

        self.delta = x2 - x1
        self.length = float(np.linalg.norm(self.delta))

        if self.length <= 0.0:
            raise ValueError("Невозможно построить отрезок нулевой длины")

        self.direction = self.delta / self.length

        dx, dy, dz = self.delta
        horiz = np.hypot(dx, dy)

        self.azimuth = float(np.arctan2(dy, dx))
        self.elevation = float(np.arctan2(dz, horiz))

    # ------------------------------------------------------------------
    # Точность
    # ------------------------------------------------------------------
    def _compute_accuracy(self):
        cov1 = getattr(self.p1, "cov_xyz", None)
        cov2 = getattr(self.p2, "cov_xyz", None)

        if cov1 is None or cov2 is None:
            self.reliable_accuracy = False
            return

        cov1 = np.asarray(cov1, dtype=float)
        cov2 = np.asarray(cov2, dtype=float)

        if cov1.shape != (3, 3):
            raise ValueError(f"cov_xyz первой точки должна иметь форму (3,3), получено {cov1.shape}")
        if cov2.shape != (3, 3):
            raise ValueError(f"cov_xyz второй точки должна иметь форму (3,3), получено {cov2.shape}")

        if self.cross_cov_12 is None:
            cross_cov = np.zeros((3, 3), dtype=float)
        else:
            cross_cov = self.cross_cov_12

        # Cov(delta) = Cov(X2 - X1) = Cov(X2) + Cov(X1) - Cov(X2,X1) - Cov(X1,X2)
        self.cov_delta = cov2 + cov1 - cross_cov - cross_cov.T

        # На случай накопления численных асимметрий
        self.cov_delta = 0.5 * (self.cov_delta + self.cov_delta.T)

        self.sigma_xyz = np.sqrt(np.maximum(np.diag(self.cov_delta), 0.0))

        dx, dy, dz = self.delta
        L = self.length
        horiz2 = dx ** 2 + dy ** 2
        horiz = np.sqrt(horiz2)

        # 1) Погрешность длины
        g_L = (self.delta / L).reshape(1, 3)
        var_L = self._quad_form(g_L, self.cov_delta)
        self.sigma_length = float(np.sqrt(max(var_L, 0.0)))

        # 2) Ковариация единичного вектора направления u = delta / |delta|
        I = np.eye(3, dtype=float)
        J_u = (I - np.outer(self.direction, self.direction)) / L
        self.cov_direction = J_u @ self.cov_delta @ J_u.T
        self.cov_direction = 0.5 * (self.cov_direction + self.cov_direction.T)

        # 3) Погрешность азимута: az = atan2(dy, dx)
        if horiz2 > 1e-16:
            J_az = np.array([[-dy / horiz2, dx / horiz2, 0.0]], dtype=float)
            var_az = self._quad_form(J_az, self.cov_delta)
            self.sigma_azimuth = float(np.sqrt(max(var_az, 0.0)))
        else:
            self.sigma_azimuth = None

        # 4) Погрешность угла возвышения: el = atan2(dz, horiz)
        if horiz > 1e-16 and L > 1e-16:
            J_el = np.array([[
                -(dz * dx) / (horiz * L ** 2),
                -(dz * dy) / (horiz * L ** 2),
                horiz / (L ** 2),
            ]], dtype=float)
            var_el = self._quad_form(J_el, self.cov_delta)
            self.sigma_elevation = float(np.sqrt(max(var_el, 0.0)))
        else:
            self.sigma_elevation = None

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------
    @staticmethod
    def _quad_form(J: np.ndarray, C: np.ndarray) -> float:
        """
        Квадратичная форма J C J^T для J формы (1,n).
        Возвращает Python float, корректно для NumPy 2.x.
        """
        return (J @ C @ J.T).item()

    @property
    def length_mm(self):
        return self.length * 1000.0

    @property
    def sigma_length_mm(self):
        if self.sigma_length is None:
            return None
        return self.sigma_length * 1000.0

    @property
    def midpoint(self):
        return 0.5 * (
            np.array([self.p1.x, self.p1.y, self.p1.z], dtype=float) +
            np.array([self.p2.x, self.p2.y, self.p2.z], dtype=float)
        )

    def direction_angles_deg(self):
        return {
            "azimuth_deg": float(np.degrees(self.azimuth)),
            "elevation_deg": float(np.degrees(self.elevation)),
            "sigma_azimuth_deg": None if self.sigma_azimuth is None else float(np.degrees(self.sigma_azimuth)),
            "sigma_elevation_deg": None if self.sigma_elevation is None else float(np.degrees(self.sigma_elevation)),
        }

    def as_dict(self):
        return {
            "name": self.name,
            "p1_name": self.p1.name,
            "p2_name": self.p2.name,
            "length": self.length,
            "length_mm": self.length_mm,
            "delta_x": float(self.delta[0]),
            "delta_y": float(self.delta[1]),
            "delta_z": float(self.delta[2]),
            "dir_x": float(self.direction[0]),
            "dir_y": float(self.direction[1]),
            "dir_z": float(self.direction[2]),
            "azimuth_rad": self.azimuth,
            "elevation_rad": self.elevation,
            "azimuth_deg": float(np.degrees(self.azimuth)),
            "elevation_deg": float(np.degrees(self.elevation)),
            "reliable_accuracy": self.reliable_accuracy,
            "sigma_dx": None if self.sigma_xyz is None else float(self.sigma_xyz[0]),
            "sigma_dy": None if self.sigma_xyz is None else float(self.sigma_xyz[1]),
            "sigma_dz": None if self.sigma_xyz is None else float(self.sigma_xyz[2]),
            "sigma_length": self.sigma_length,
            "sigma_length_mm": self.sigma_length_mm,
            "sigma_azimuth_rad": self.sigma_azimuth,
            "sigma_elevation_rad": self.sigma_elevation,
            "sigma_azimuth_deg": None if self.sigma_azimuth is None else float(np.degrees(self.sigma_azimuth)),
            "sigma_elevation_deg": None if self.sigma_elevation is None else float(np.degrees(self.sigma_elevation)),
        }

    def __str__(self):
        parts = [
            f"CrossPointSegment(name={self.name})",
            f"length={self.length:.6f} m",
            f"delta=({self.delta[0]:.6f}, {self.delta[1]:.6f}, {self.delta[2]:.6f})",
            f"direction=({self.direction[0]:.6f}, {self.direction[1]:.6f}, {self.direction[2]:.6f})",
            f"azimuth={np.degrees(self.azimuth):.6f} deg",
            f"elevation={np.degrees(self.elevation):.6f} deg",
        ]

        if self.reliable_accuracy:
            if self.sigma_length is not None:
                parts.append(f"sigma_length={self.sigma_length:.6f} m")
            if self.sigma_xyz is not None:
                parts.append(
                    f"sigma_delta=({self.sigma_xyz[0]:.6f}, "
                    f"{self.sigma_xyz[1]:.6f}, {self.sigma_xyz[2]:.6f})"
                )
            if self.sigma_azimuth is not None:
                parts.append(f"sigma_azimuth={np.degrees(self.sigma_azimuth):.6f} deg")
            if self.sigma_elevation is not None:
                parts.append(f"sigma_elevation={np.degrees(self.sigma_elevation):.6f} deg")
        else:
            parts.append("accuracy=UNRELIABLE")

        return ", ".join(parts)

    def __repr__(self):
        return (
            f"CrossPointSegment({self.p1.name!r}, {self.p2.name!r}, "
            f"length={self.length:.4f}, reliable_accuracy={self.reliable_accuracy})"
        )


if __name__ == "__main__":
    from app.cross_points.CrossPointListRestorer import CrossPointListRestorer

    points_path = "/data/8_floors_wall/output/scan_2334_filt/cross_points_good_filtered_by_ellipsoid.csv"

    base_points_restorer = CrossPointListRestorer(points_path)
    base_points = base_points_restorer.restore_all()

    seg = CrossPointSegment(base_points[0], base_points[1])

    print(seg.length)
    print(seg.sigma_length)
    print(seg.direction)
    print(seg.direction_angles_deg())
    print(seg)
    print(seg.as_dict())