import logging
from pathlib import Path

from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.util.batch_cross_points import BatchCrossPointProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

# SUBSCANS_DIR   = Path("../data/8_floors_dvor/output/subscans")
# OUTPUT_BASE    = Path("../data/8_floors_dvor/output/cross_points")
SUBSCANS_DIR   = Path("../data/8_floors_2/output/subscans")
OUTPUT_BASE    = Path("../data/8_floors_2/output/cross_points")

SCAN_DIRS      = ["1", "2"]
# SCAN_DIRS      = ["all"]
# SCAN_DIRS      = ["0_1", "0_2"]

PROCESSOR_CONFIG = dict(
    extensions        = (".las", ".laz", ".txt", ".xyz", ".pts"),
    max_ellipsoid_axis= 0.05,
    eps               = 0.05,
    show_scans        = False,
    choose_scan_directly_from_dbscan = True,
    mse_threshold     = 0.0001,
    max_iteration     = 20,
    k_sigma           = 2.0,
    base_fitter       = PlaneL1Fitter,
)

def main():
    for folder_name in SCAN_DIRS:
        scans_root = SUBSCANS_DIR / folder_name

        if not scans_root.exists():
            logger.warning("Папка субсканов не найдена, пропускаю: %s", scans_root)
            continue

        # Каждый скан — отдельная подпапка вида subscans/<folder>/<scan_stem>/
        scan_subdirs = sorted(p for p in scans_root.iterdir() if p.is_dir())

        logger.info(
            "Папка '%s': найдено %d сканов для обработки",
            folder_name, len(scan_subdirs)
        )

        for scan_subdir in scan_subdirs:
            output_dir = OUTPUT_BASE / folder_name / scan_subdir.name

            logger.info(
                "  Обработка: %s → %s", scan_subdir, output_dir
            )

            processor = BatchCrossPointProcessor(
                input_dir  = str(scan_subdir),
                output_dir = str(output_dir),
                **PROCESSOR_CONFIG,
            )

            result = processor.run()

            logger.info(
                "  %s | всего=%d | хороших=%d | после фильтра=%d | ошибок=%d",
                scan_subdir.name,
                len(result["df_all"]),
                len(result["df_good"]),
                len(result["df_good_filtered"]),
                len(result["df_errors"]),
            )

    logger.info("Готово. Результаты сохранены в: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
