# thesis_giab/1_get_subscans_around_points.py

import logging
from pathlib import Path

from app.base.scan.Scan import Scan
from app.util.sub_scan_separator import (
    get_named_point_list,
    split_scan_by_points,
    save_sub_scans,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

# BASE_DATA_DIR = Path("../data/8_floors_dvor")
BASE_DATA_DIR = Path("../data/8_floors_2")
POINTS_FILE   = BASE_DATA_DIR / "vse_tochki.txt"
OUTPUT_DIR    = BASE_DATA_DIR / "output" / "subscans"
CUBE_SIZE     = 0.5
SCAN_DIRS     = ["1", "2"]
# SCAN_DIRS     = ["0_1", "0_2"]
EXTENSIONS    = {".las", ".laz", ".txt", ".xyz", ".pts"}


def main():
    # 1. Загружаем список опорных точек
    if not POINTS_FILE.exists():
        raise FileNotFoundError(f"Файл точек не найден: {POINTS_FILE}")

    named_points = get_named_point_list(str(POINTS_FILE))
    logger.info("Загружено опорных точек: %d", len(named_points))

    # 2. Обходим папки 1 и 2
    for folder_name in SCAN_DIRS:
        scan_dir = BASE_DATA_DIR / folder_name

        if not scan_dir.exists():
            logger.warning("Папка не найдена, пропускаю: %s", scan_dir)
            continue

        las_files = sorted(
            p for p in scan_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        logger.info(
            "Папка '%s': найдено %d файлов для обработки",
            folder_name, len(las_files)
        )

        for scan_file in las_files:
            logger.info("  Загрузка скана: %s", scan_file.name)

            # 3. Загружаем скан
            scan = Scan(scan_name=scan_file.stem)
            scan.import_points_from_file(str(scan_file))

            # 4. Разбиваем скан по опорным точкам
            sub_scans = split_scan_by_points(
                scan, named_points, cube_size=CUBE_SIZE
            )

            # 5. Фильтруем — оставляем только непустые субсканы
            non_empty = {
                name: ss
                for name, ss in sub_scans.items()
                if len(list(ss)) > 0
            }

            logger.info(
                "  Субсканов с точками: %d / %d",
                len(non_empty), len(sub_scans)
            )

            # 6. Сохраняем только непустые
            if non_empty:
                folder_output = OUTPUT_DIR / folder_name
                saved_dir = save_sub_scans(
                    non_empty, scan, base_dir=str(folder_output)
                )
                logger.info("  Сохранено в: %s", saved_dir)
            else:
                logger.warning(
                    "  Скан '%s' не перекрывается ни с одной опорной точкой, "
                    "файлы не сохраняются.", scan_file.name
                )

    logger.info("Готово. Все субсканы сохранены в: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()