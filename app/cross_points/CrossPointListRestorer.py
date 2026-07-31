import json

import numpy as np
import pandas as pd

from app.cross_points.CrossPoint import CrossPoint


class CrossPointListRestorer:
    """
    Восстанавливает список объектов CrossPoint из CSV,
    сформированного пакетным обработчиком.

    Из каждой строки таблицы восстанавливаются:
    - name, x, y, z
    - status
    - mse
    - planes_mse
    - sigma_xyz
    - cov_xyz
    - ellipsoid
    - reliable_accuracy
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def restore_all(self) -> list[CrossPoint]:
        points = []
        for _, row in self.df.iterrows():
            point = self._restore_cross_point(row)
            points.append(point)
        return points

    def restore_by_index(self, row_index: int) -> CrossPoint:
        row = self.df.iloc[row_index]
        return self._restore_cross_point(row)

    def restore_by_name(self, cross_point_name: str) -> CrossPoint:
        subset = self.df[self.df["cross_point_name"] == cross_point_name]
        if subset.empty:
            raise KeyError(f"Точка с именем '{cross_point_name}' не найдена")
        if len(subset) > 1:
            raise ValueError(
                f"Найдено несколько строк для '{cross_point_name}'. Используйте restore_all() или restore_by_index()."
            )
        return self._restore_cross_point(subset.iloc[0])

    def _restore_cross_point(self, row: pd.Series) -> CrossPoint:
        cp = CrossPoint(
            name=row["cross_point_name"],
            x=float(row["cross_point_x"]),
            y=float(row["cross_point_y"]),
            z=float(row["cross_point_z"]),
        )

        cp.status = row.get("cross_point_status")
        cp.reliable_accuracy = self._safe_bool(
            row.get("cross_point_reliable_accuracy"),
            default=False,
        )

        planes_mse = self._json_loads(row.get("planes_mse_json"))
        if planes_mse is not None:
            cp.planes_mse = [float(v) for v in planes_mse]

        sigma_x = self._safe_float(row.get("sigma_x"))
        sigma_y = self._safe_float(row.get("sigma_y"))
        sigma_z = self._safe_float(row.get("sigma_z"))
        if sigma_x is not None and sigma_y is not None and sigma_z is not None:
            cp.sigma_xyz = np.array([sigma_x, sigma_y, sigma_z], dtype=float)

        cov_xyz = self._json_to_ndarray(row.get("cov_xyz_json"))
        if cov_xyz is not None:
            cp.cov_xyz = cov_xyz

        ellipsoid_confidence = self._safe_float(row.get("ellipsoid_confidence"))
        ellipsoid_axis_a = self._safe_float(row.get("ellipsoid_axis_a"))
        ellipsoid_axis_b = self._safe_float(row.get("ellipsoid_axis_b"))
        ellipsoid_axis_c = self._safe_float(row.get("ellipsoid_axis_c"))
        ellipsoid_directions = self._json_to_ndarray(row.get("ellipsoid_directions_json"))

        if all(v is not None for v in [ellipsoid_confidence, ellipsoid_axis_a, ellipsoid_axis_b, ellipsoid_axis_c]):
            cp.ellipsoid = {
                "confidence": ellipsoid_confidence,
                "semi_axes": np.array([
                    ellipsoid_axis_a,
                    ellipsoid_axis_b,
                    ellipsoid_axis_c,
                ], dtype=float),
                "directions": ellipsoid_directions,
            }

        # mse восстанавливаем аккуратно:
        # 1) если есть cov_xyz -> как sqrt(trace(cov_xyz))
        # 2) иначе берём сохранённое cross_point_mse
        if cp.cov_xyz is not None:
            cp.mse = float(np.sqrt(np.trace(cp.cov_xyz)))
        else:
            cp.mse = self._safe_float(row.get("cross_point_mse"))

        return cp

    @staticmethod
    def _json_loads(value):
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, (list, dict)):
            return value

        s = str(value).strip()
        if s == "" or s.lower() == "nan":
            return None
        return json.loads(s)

    def _json_to_ndarray(self, value):
        parsed = self._json_loads(value)
        if parsed is None:
            return None
        return np.array(parsed, dtype=float)

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        try:
            s = str(value).strip()
            if s.lower() == "nan":
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_bool(value, default=None):
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        if isinstance(value, bool):
            return value

        s = str(value).strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        return default


if __name__ == "__main__":
    csv_path = "/data/8_floors_wall/output/scan_2334_filt/cross_points_good_filtered_by_ellipsoid.csv"

    restorer = CrossPointListRestorer(csv_path)

    points = restorer.restore_all()
    print(f"restored points: {len(points)}")
    print(points[0])

    # p = restorer.restore_by_name("2_10_vl")
    # print(p)

    print(*points, sep="\n")

