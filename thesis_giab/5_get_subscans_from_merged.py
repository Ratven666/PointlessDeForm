import logging
import pandas as pd
from pathlib import Path

from app.base.scan.Scan import Scan
from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.util.sub_scan_separator import (
    get_named_point_list,
    split_scan_by_points,
    save_sub_scans,
)
from app.util.batch_cross_points import BatchCrossPointProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────

BASE_DATA_DIR = Path("../data/8_floors_dvor")
POINTS_FILE   = BASE_DATA_DIR / "vse_tochki.txt"

MERGED_SCANS = {
    "1": BASE_DATA_DIR / "output" / "merged" / "1" / "epoch_1_full.txt",
    "2": BASE_DATA_DIR / "output" / "merged" / "2" / "epoch_2_full.txt",
}

# MERGED_SCANS = {
#     "all": BASE_DATA_DIR / "output" / "merged" / "all" / "epoch_all_full.txt",
# }

SUBSCANS_DIR = BASE_DATA_DIR / "output" / "merged_subscans"
CROSS_DIR    = BASE_DATA_DIR / "output" / "merged_cross_points"

CUBE_SIZE = 0.5

PROCESSOR_CONFIG = dict(
    extensions                       = (".las", ".laz", ".txt", ".xyz", ".pts"),
    max_ellipsoid_axis               = 0.05,
    eps                              = 0.05,
    show_scans                       = False,
    choose_scan_directly_from_dbscan = True,
    mse_threshold                    = 0.0001,
    max_iteration                    = 20,
    k_sigma                          = 2.0,
    base_fitter                      = PlaneL1Fitter,
)


def process_epoch(folder_name: str, merged_file: Path, named_points: list):
    logger.info("=" * 60)
    logger.info("Эпоха '%s': %s", folder_name, merged_file.name)
    logger.info("=" * 60)

    if not merged_file.exists():
        logger.error("Файл не найден: %s", merged_file)
        return

    # ── ШАГ 1: загружаем объединённый скан ──────────────────────
    logger.info("  Загрузка объединённого скана...")
    scan = Scan(scan_name=f"epoch_{folder_name}_full")
    scan.import_points_from_file(str(merged_file))
    logger.info("  Точек загружено: %d", len(list(scan)))

    # ── ШАГ 2: нарезаем субсканы ────────────────────────────────
    logger.info("  Нарезка субсканов (cube_size=%.2f м)...", CUBE_SIZE)
    sub_scans = split_scan_by_points(scan, named_points, cube_size=CUBE_SIZE)

    non_empty = {
        name: ss for name, ss in sub_scans.items() if len(list(ss)) > 0
    }
    logger.info(
        "  Непустых субсканов: %d / %d",
        len(non_empty), len(sub_scans)
    )

    # save_sub_scans создаёт: SUBSCANS_DIR/<folder_name>/<point_name>.<ext>
    # или:                    SUBSCANS_DIR/<folder_name>/<scan_name>/<point_name>.<ext>
    subscans_root = SUBSCANS_DIR / folder_name
    save_sub_scans(non_empty, scan, base_dir=str(subscans_root))
    logger.info("  Субсканы сохранены в: %s", subscans_root)

    # ── ШАГ 3: расчёт кросс-точек ───────────────────────────────
    # BatchCrossPointProcessor._scan_files() делает iterdir() — НЕ рекурсивно.
    # Нужно найти все папки/директории, содержащие непосредственно файлы,
    # и запустить процессор для каждой из них.

    cross_out = CROSS_DIR / folder_name
    cross_out.mkdir(parents=True, exist_ok=True)

    # Собираем все директории, в которых есть хотя бы один файл субскана
    extensions = set(PROCESSOR_CONFIG["extensions"])
    dirs_with_files = sorted(set(
        p.parent for p in subscans_root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ))

    if not dirs_with_files:
        logger.error(
            "  Не найдено файлов субсканов в '%s'. "
            "Проверьте структуру папок и параметр CUBE_SIZE.", subscans_root
        )
        return

    logger.info(
        "  Найдено %d папок с субсканами для обработки.", len(dirs_with_files)
    )

    all_results = []

    for sub_dir in dirs_with_files:
        # Выходная папка зеркалит структуру входной
        relative = sub_dir.relative_to(subscans_root)
        out_sub = cross_out / relative
        out_sub.mkdir(parents=True, exist_ok=True)

        logger.info("    → обрабатываю: %s", sub_dir)

        processor = BatchCrossPointProcessor(
            input_dir  = str(sub_dir),
            output_dir = str(out_sub),
            **PROCESSOR_CONFIG,
        )
        result = processor.run()
        all_results.append(result)

    # ── Сводный отчёт по эпохе ───────────────────────────────────
    df_all_epoch      = pd.concat([r["df_all"]           for r in all_results], ignore_index=True)
    df_good_epoch     = pd.concat([r["df_good"]          for r in all_results], ignore_index=True)
    df_filtered_epoch = pd.concat([r["df_good_filtered"] for r in all_results], ignore_index=True)
    df_errors_epoch   = pd.concat([r["df_errors"]        for r in all_results], ignore_index=True)

    # Сохраняем сводные CSV в корень эпохи
    df_all_epoch.to_csv(cross_out / "cross_points_all.csv", index=False)
    df_good_epoch.to_csv(cross_out / "cross_points_good.csv", index=False)
    df_filtered_epoch.to_csv(cross_out / "cross_points_good_filtered_by_ellipsoid.csv", index=False)
    df_errors_epoch.to_csv(cross_out / "cross_points_errors.csv", index=False)

    logger.info(
        "  Эпоха '%s' итого | all=%d | good=%d | filtered=%d | errors=%d",
        folder_name,
        len(df_all_epoch),
        len(df_good_epoch),
        len(df_filtered_epoch),
        len(df_errors_epoch),
    )


def main():
    SUBSCANS_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_DIR.mkdir(parents=True, exist_ok=True)

    if not POINTS_FILE.exists():
        raise FileNotFoundError(f"Файл точек не найден: {POINTS_FILE}")

    named_points = get_named_point_list(str(POINTS_FILE))
    logger.info("Загружено опорных точек: %d", len(named_points))

    for folder_name, merged_file in MERGED_SCANS.items():
        process_epoch(folder_name, merged_file, named_points)

    logger.info("Всё готово.")
    logger.info("  Субсканы:    %s", SUBSCANS_DIR)
    logger.info("  Кросс-точки: %s", CROSS_DIR)


if __name__ == "__main__":
    main()