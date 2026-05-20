import os
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.util.CrossPointExacter import CrossPointExacter

logger = logging.getLogger(__name__)


class BatchCrossPointProcessor:
    """
    Пакетная обработка сканов в папке.

    Для каждого файла:
    1. Строит плоскости и точку пересечения.
    2. Собирает максимально полную информацию о точке, плоскостях,
       ковариациях, эллипсоиде и диагностике геометрии.
    3. Формирует полный DataFrame.
    4. Формирует подмножество только хороших точек пересечения.
    5. Дополнительно фильтрует хорошие точки по максимальной полуоси эллипсоида.
    6. Сохраняет результаты в CSV / XLSX / JSON.

    Особенности:
    - Прогресс-бар построен на tqdm.
    - Логи перенаправляются через tqdm, поэтому не ломают отображение прогресса.
    - Ошибки по отдельным файлам не останавливают весь пакетный запуск.
    """

    def __init__(self,
                 input_dir: str,
                 output_dir: str = "output",
                 extensions=(".las", ".laz", ".txt", ".xyz", ".pts"),
                 max_ellipsoid_axis: float | None = None,
                 labels_map: dict | None = None,
                 eps: float = 0.01,
                 show_scans: bool = False,
                 choose_scan_directly_from_dbscan: bool = True,
                 mse_threshold: float = 0.0001,
                 max_iteration: int = 20,
                 k_sigma: float = 2.0,
                 base_fitter=PlaneL1Fitter):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extensions = tuple(ext.lower() for ext in extensions)
        self.max_ellipsoid_axis = max_ellipsoid_axis
        self.labels_map = labels_map or {}
        self.eps = eps
        self.show_scans = show_scans
        self.choose_scan_directly_from_dbscan = choose_scan_directly_from_dbscan
        self.mse_threshold = mse_threshold
        self.max_iteration = max_iteration
        self.k_sigma = k_sigma
        self.base_fitter = base_fitter

    # ------------------------------------------------------------------
    # Служебные методы
    # ------------------------------------------------------------------
    def _scan_files(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Папка не найдена: {self.input_dir}")

        files = []
        for path in sorted(self.input_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in self.extensions:
                files.append(path)
        return files

    @staticmethod
    def _to_list(value):
        if value is None:
            return None
        return np.asarray(value).tolist()

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _json_dumps(value):
        return json.dumps(value, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Сбор информации о плоскости
    # ------------------------------------------------------------------
    def _plane_to_dict(self, plane, idx: int):
        normal = getattr(plane, "normal", None)
        point = getattr(plane, "point", None)
        equation = getattr(plane, "equation", None)
        cov_params = getattr(plane, "cov_params", None)

        return {
            f"plane_{idx}_mse": self._safe_float(getattr(plane, "mse", None)),
            f"plane_{idx}_sigma0": self._safe_float(getattr(plane, "sigma0", None)),
            f"plane_{idx}_A": self._safe_float(getattr(plane, "A", None)),
            f"plane_{idx}_B": self._safe_float(getattr(plane, "B", None)),
            f"plane_{idx}_C": self._safe_float(getattr(plane, "C", None)),
            f"plane_{idx}_D": self._safe_float(getattr(plane, "D", None)),
            f"plane_{idx}_equation_json": self._json_dumps(self._to_list(equation)),
            f"plane_{idx}_normal_x": self._safe_float(normal[0]) if normal is not None else None,
            f"plane_{idx}_normal_y": self._safe_float(normal[1]) if normal is not None else None,
            f"plane_{idx}_normal_z": self._safe_float(normal[2]) if normal is not None else None,
            f"plane_{idx}_point_x": self._safe_float(point[0]) if point is not None else None,
            f"plane_{idx}_point_y": self._safe_float(point[1]) if point is not None else None,
            f"plane_{idx}_point_z": self._safe_float(point[2]) if point is not None else None,
            f"plane_{idx}_cov_params_json": self._json_dumps(self._to_list(cov_params)),
            f"plane_{idx}_has_covariance": cov_params is not None,
            f"plane_{idx}_scan_len": len(plane.scan) if getattr(plane, "scan", None) is not None else None,
        }

    # ------------------------------------------------------------------
    # Сбор информации о точке пересечения
    # ------------------------------------------------------------------
    def _cross_point_to_dict(self, scan_file: Path, cpe: CrossPointExacter):
        cp = cpe.cross_point
        diag = getattr(cpe, "geometry_diagnostics", None)

        sigma_xyz = getattr(cp, "sigma_xyz", None)
        cov_xyz = getattr(cp, "cov_xyz", None)
        ellipsoid = getattr(cp, "ellipsoid", None)

        row = {
            "scan_file": scan_file.name,
            "scan_path": str(scan_file),
            "scan_stem": scan_file.stem,
            "cross_point_name": getattr(cp, "name", None),
            "cross_point_status": getattr(cp, "status", None),
            "cross_point_x": self._safe_float(getattr(cp, "x", None)),
            "cross_point_y": self._safe_float(getattr(cp, "y", None)),
            "cross_point_z": self._safe_float(getattr(cp, "z", None)),
            "cross_point_mse": self._safe_float(getattr(cp, "mse", None)),
            "cross_point_reliable_accuracy": bool(getattr(cp, "reliable_accuracy", False)),
            "planes_mse_json": self._json_dumps(self._to_list(getattr(cp, "planes_mse", None))),
            "sigma_x": self._safe_float(sigma_xyz[0]) if sigma_xyz is not None else None,
            "sigma_y": self._safe_float(sigma_xyz[1]) if sigma_xyz is not None else None,
            "sigma_z": self._safe_float(sigma_xyz[2]) if sigma_xyz is not None else None,
            "cov_xyz_json": self._json_dumps(self._to_list(cov_xyz)),
            "ellipsoid_confidence": self._safe_float(ellipsoid.get("confidence")) if ellipsoid is not None else None,
            "ellipsoid_axis_a": self._safe_float(ellipsoid.get("semi_axes", [None, None, None])[0]) if ellipsoid is not None else None,
            "ellipsoid_axis_b": self._safe_float(ellipsoid.get("semi_axes", [None, None, None])[1]) if ellipsoid is not None else None,
            "ellipsoid_axis_c": self._safe_float(ellipsoid.get("semi_axes", [None, None, None])[2]) if ellipsoid is not None else None,
            "ellipsoid_max_axis": self._safe_float(np.max(ellipsoid.get("semi_axes"))) if ellipsoid is not None else None,
            "ellipsoid_directions_json": self._json_dumps(self._to_list(ellipsoid.get("directions"))) if ellipsoid is not None else None,
            "geometry_status": getattr(diag, "status", None),
            "geometry_det": self._safe_float(getattr(diag, "det", None)),
            "geometry_cond": self._safe_float(getattr(diag, "cond", None)),
            "geometry_has_parallel": getattr(diag, "has_parallel", None),
            "geometry_singular_values_json": self._json_dumps(self._to_list(getattr(diag, "singular_values", None))),
            "geometry_messages_json": self._json_dumps(getattr(diag, "messages", None)),
            "n_planes": len(cpe.planes) if getattr(cpe, "planes", None) is not None else None,
        }

        for idx, plane in enumerate(cpe.planes, start=1):
            row.update(self._plane_to_dict(plane, idx))

        return row

    # ------------------------------------------------------------------
    # Обработка одного файла
    # ------------------------------------------------------------------
    def process_one(self, scan_file: Path):
        labels = self.labels_map.get(scan_file.name)

        logger.info("Начало обработки файла: %s", scan_file.name)

        cpe = CrossPointExacter(
            file_path=str(scan_file),
            choose_scan_directly_from_dbscan=self.choose_scan_directly_from_dbscan,
            show_scans=self.show_scans,
            labels=labels,
            eps=self.eps,
        )

        cpe.calculate_planes(
            base_fitter=self.base_fitter,
            mse_threshold=self.mse_threshold,
            max_iteration=self.max_iteration,
            k_sigma=self.k_sigma,
        )
        cpe.calculate_intersect_point()

        row = self._cross_point_to_dict(scan_file, cpe)

        logger.info(
            "Готово: %s | status=%s | reliable=%s | max_axis=%s",
            scan_file.name,
            row.get("cross_point_status"),
            row.get("cross_point_reliable_accuracy"),
            f"{row['ellipsoid_max_axis']:.6f}" if row.get("ellipsoid_max_axis") is not None else "None",
        )

        return row

    # ------------------------------------------------------------------
    # Пакетный запуск
    # ------------------------------------------------------------------
    def run(self):
        files = self._scan_files()
        rows = []
        errors = []

        logger.info("Найдено файлов для обработки: %d", len(files))

        with logging_redirect_tqdm():
            with tqdm(files, desc="Обработка сканов", dynamic_ncols=True, unit="file") as pbar:
                for scan_file in pbar:
                    pbar.set_postfix_str(scan_file.name)

                    try:
                        row = self.process_one(scan_file)
                        rows.append(row)

                        pbar.set_postfix({
                            "file": scan_file.name,
                            "status": row.get("cross_point_status"),
                            "good": sum(1 for r in rows if r.get("cross_point_status") == "GOOD"),
                            "errors": len(errors),
                        })

                    except Exception as exc:
                        errors.append({
                            "scan_file": scan_file.name,
                            "scan_path": str(scan_file),
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                        logger.exception("Ошибка обработки файла: %s", scan_file.name)

                        pbar.set_postfix({
                            "file": scan_file.name,
                            "status": "ERROR",
                            "good": sum(1 for r in rows if r.get("cross_point_status") == "GOOD"),
                            "errors": len(errors),
                        })

        df_all = pd.DataFrame(rows)
        df_errors = pd.DataFrame(errors)

        if not df_all.empty:
            df_good = df_all[df_all["cross_point_status"] == "GOOD"].copy()
        else:
            df_good = pd.DataFrame()

        if self.max_ellipsoid_axis is not None and not df_good.empty:
            df_good_filtered = df_good[
                df_good["ellipsoid_max_axis"].notna() &
                (df_good["ellipsoid_max_axis"] <= self.max_ellipsoid_axis)
            ].copy()
        else:
            df_good_filtered = df_good.copy()

        all_csv = self.output_dir / "cross_points_all.csv"
        good_csv = self.output_dir / "cross_points_good.csv"
        filtered_csv = self.output_dir / "cross_points_good_filtered_by_ellipsoid.csv"
        errors_csv = self.output_dir / "cross_points_errors.csv"
        xlsx_path = self.output_dir / "cross_points_report.xlsx"
        meta_json = self.output_dir / "cross_points_run_metadata.json"

        df_all.to_csv(all_csv, index=False)
        df_good.to_csv(good_csv, index=False)
        df_good_filtered.to_csv(filtered_csv, index=False)
        df_errors.to_csv(errors_csv, index=False)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="all_points", index=False)
            df_good.to_excel(writer, sheet_name="good_points", index=False)
            df_good_filtered.to_excel(writer, sheet_name="filtered_points", index=False)
            df_errors.to_excel(writer, sheet_name="errors", index=False)

        metadata = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "extensions": list(self.extensions),
            "max_ellipsoid_axis": self.max_ellipsoid_axis,
            "eps": self.eps,
            "show_scans": self.show_scans,
            "choose_scan_directly_from_dbscan": self.choose_scan_directly_from_dbscan,
            "mse_threshold": self.mse_threshold,
            "max_iteration": self.max_iteration,
            "k_sigma": self.k_sigma,
            "base_fitter": getattr(self.base_fitter, "__name__", str(self.base_fitter)),
            "n_all": int(len(df_all)),
            "n_good": int(len(df_good)),
            "n_good_filtered": int(len(df_good_filtered)),
            "n_errors": int(len(df_errors)),
            "output_files": {
                "all_csv": str(all_csv),
                "good_csv": str(good_csv),
                "filtered_csv": str(filtered_csv),
                "errors_csv": str(errors_csv),
                "xlsx": str(xlsx_path),
                "meta_json": str(meta_json),
            },
        }

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "Пакетная обработка завершена | all=%d | good=%d | filtered=%d | errors=%d",
            len(df_all), len(df_good), len(df_good_filtered), len(df_errors)
        )

        return {
            "df_all": df_all,
            "df_good": df_good,
            "df_good_filtered": df_good_filtered,
            "df_errors": df_errors,
            "metadata": metadata,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    )

    input_dir = "../../data/200226/сканер/lazpredobr"
    output_dir = "output"

    processor = BatchCrossPointProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        extensions=(".las", ".laz"),
        max_ellipsoid_axis=0.01,
        eps=0.05,
        show_scans=False,
        choose_scan_directly_from_dbscan=True,
        mse_threshold=0.0001,
        max_iteration=20,
        k_sigma=2.0,
        base_fitter=PlaneL1Fitter,
    )

    result = processor.run()

    logger.info("Всего обработано: %d", len(result["df_all"]))
    logger.info("Хороших точек: %d", len(result["df_good"]))
    logger.info("Хороших после фильтра по эллипсоиду: %d", len(result["df_good_filtered"]))
    logger.info("Ошибок: %d", len(result["df_errors"]))
