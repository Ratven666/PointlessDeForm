import json
import logging
from pathlib import Path

import math

from app.base.scan.ScanNetworkRegistrator import ScanNetworkRegistrator
from app.cross_points.CrossPointListRestorer import CrossPointListRestorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR     = Path("../data/8_floors_dvor")
CROSS_DIR    = BASE_DIR / "output" / "cross_points"   # ← новый путь
RESULTS_DIR  = BASE_DIR / "output" / "registration"
CSV_FILENAME = "cross_points_good_filtered_by_ellipsoid.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# FOLDERS      = ["1", "2"]
FOLDERS      = ["all"]
METHOD       = "LSM"    # 'LSM' или 'L1'
K_SIGMA      = 3.0
MIN_COMMON   = 3        # минимум общих точек для регистрации пары



def load_cross_points_for_folder(folder_name: str) -> dict[str, list]:
    """
    Структура:
      data/8_floors_dvor/output/cross_points/1/360_o_las/cross_points_good_filtered_by_ellipsoid.csv
                                              ^folder    ^scan_stem
    Возвращает dict {scan_stem: [CrossPoint, ...]}
    """
    folder_dir = CROSS_DIR / folder_name

    if not folder_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder_dir}")

    result: dict[str, list] = {}

    for scan_subdir in sorted(folder_dir.iterdir()):
        if not scan_subdir.is_dir():
            continue

        csv_path = scan_subdir / CSV_FILENAME
        if not csv_path.exists():
            logger.warning("Нет CSV: %s — пропускаю", csv_path)
            continue

        restorer = CrossPointListRestorer(str(csv_path))
        points = restorer.restore_all()
        result[scan_subdir.name] = points

        logger.info(
            "  %-30s : %d точек  (GOOD+reliable: %d)",
            scan_subdir.name,
            len(points),
            sum(1 for p in points
                if getattr(p, "status", None) == "GOOD"
                and getattr(p, "reliable_accuracy", False)),
        )

    logger.info(
        "Папка '%s': %d сканов загружено, %d точек суммарно",
        folder_name,
        len(result),
        sum(len(v) for v in result.values()),
    )
    return result


def save_transforms(folder_name: str, result):
    """Сохраняет трансформации в JSON для последующего использования."""
    out = {}
    for scan_name, T in result.transforms.items():
        if T is None:
            out[scan_name] = {"anchor": True}
        else:
            out[scan_name] = {
                "anchor": False,
                "method": T.method,
                "n_common": T.n_common,
                "n_used": T.n_used,
                "rmse": T.rmse,
                "mae": T.mae,
                "max_res": T.max_res,
                "R": T.R.tolist(),
                "t": T.t.tolist(),
                "omega_deg": float(T.omega * 180 / math.pi),
                "phi_deg":   float(T.phi   * 180 / math.pi),
                "kappa_deg": float(T.kappa * 180 / math.pi),
                "tx": T.tx, "ty": T.ty, "tz": T.tz,
            }

    json_path = RESULTS_DIR / f"transforms_folder_{folder_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("Трансформации сохранены: %s", json_path)


def main():
    for folder_name in FOLDERS:
        logger.info("=" * 60)
        logger.info("Регистрация сканов в папке: %s", folder_name)
        logger.info("=" * 60)

        cross_points_map = load_cross_points_for_folder(folder_name)

        if not cross_points_map:
            logger.warning("Нет данных для папки %s, пропускаю", folder_name)
            continue

        registrator = ScanNetworkRegistrator(
            cross_points_map=cross_points_map,
            method=METHOD,
            k_sigma=K_SIGMA,
            min_common=MIN_COMMON,
            anchor_scan=None,          # автовыбор опорного скана
        )

        result = registrator.run()

        save_transforms(folder_name, result)

        # Сохраняем выровненные точки для последующего анализа деформаций
        for scan_name, pts_dict in result.registered_points.items():
            pts_file = RESULTS_DIR / f"registered_points_{folder_name}_{scan_name}.csv"
            with open(pts_file, "w", encoding="utf-8") as f:
                f.write("name,x,y,z,sigma_x,sigma_y,sigma_z\n")
                for pt in pts_dict.values():
                    sx, sy, sz = ("", "", "")
                    if getattr(pt, "sigma_xyz", None) is not None:
                        sx, sy, sz = pt.sigma_xyz
                    f.write(
                        f"{pt.name},{pt.x:.9f},{pt.y:.9f},{pt.z:.9f},"
                        f"{sx},{sy},{sz}\n"
                    )

        if result.failed_scans:
            logger.warning(
                "Папка %s: не удалось зарегистрировать сканы: %s",
                folder_name, result.failed_scans
            )

    logger.info("Готово.")


if __name__ == "__main__":
    main()
