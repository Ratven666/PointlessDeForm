"""
Шаг 4 (упрощённый): объединение субсканов в единое облако эпохи.

Для каждой папки из FOLDERS:
  1. Обходит все подпапки subscans/<folder>/<scan_stem>/
  2. Склеивает субсканы одного скана в один Scan
  3. Сохраняет в  output/merged/<folder>/<scan_stem>_merged.txt
  4. Слёт все сканы папки → output/merged/<folder>/epoch_<folder>_full.txt

Структура:
  data/8_floors_dvor/
    output/
      subscans/<folder>/<scan_stem>/<point_name>.txt  ← входные данные
      merged/<folder>/<scan_stem>_merged.txt          ← скан целиком
      merged/<folder>/epoch_<folder>_full.txt         ← всё облако эпохи
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.base.scan.Scan import Scan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

BASE_DIR       = Path("../data/8_floors_dvor")
SUBSCANS_DIR   = BASE_DIR / "output" / "subscans"
MERGED_DIR     = BASE_DIR / "output" / "merged"
FOLDERS        = ["1", "2"]          # ["all"] для единой сессии
SCAN_EXTENSIONS = {".las", ".laz", ".txt", ".xyz", ".pts"}


# ──────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────

def collect_subscan_files(scan_subdir: Path) -> list[Path]:
    """Возвращает отсортированный список файлов субсканов в папке."""
    return sorted(
        p for p in scan_subdir.iterdir()
        if p.is_file() and p.suffix.lower() in SCAN_EXTENSIONS
    )


def merge_subscans_into_scan(scan_subdir: Path, scan_name: str) -> Scan:
    """
    Собирает все файлы из scan_subdir в один Scan.
    compute_normals=False — нормали не нужны для экспорта/деформаций.
    """
    merged = Scan(scan_name=scan_name)
    sub_files = collect_subscan_files(scan_subdir)

    if not sub_files:
        logger.warning("  [пусто] нет файлов субсканов в: %s", scan_subdir)
        return merged

    for sub_file in sub_files:
        tmp = Scan(scan_name=sub_file.stem)
        tmp.import_points_from_file(str(sub_file), compute_normals=False)
        for pt in tmp:
            merged.add_point(pt)

    logger.info(
        "  %-35s  %6d точек  (%d субсканов)",
        scan_name, len(merged), len(sub_files),
    )
    return merged


# ──────────────────────────────────────────────
# Основная логика
# ──────────────────────────────────────────────

def process_folder(folder_name: str) -> Scan:
    logger.info("=" * 60)
    logger.info("Папка: %s", folder_name)
    logger.info("=" * 60)

    subscans_root = SUBSCANS_DIR / folder_name
    if not subscans_root.exists():
        raise FileNotFoundError(
            f"Папка субсканов не найдена: {subscans_root}\n"
            f"Сначала запустите 1_get_subscans_around_points.py"
        )

    out_folder = MERGED_DIR / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    epoch_scan = Scan(scan_name=f"epoch_{folder_name}_full")

    scan_subdirs = sorted(d for d in subscans_root.iterdir() if d.is_dir())

    if not scan_subdirs:
        logger.warning("Нет подпапок сканов в: %s", subscans_root)
        return epoch_scan

    for scan_subdir in scan_subdirs:
        scan_stem = scan_subdir.name

        # 1. Собираем скан из субсканов
        scan = merge_subscans_into_scan(scan_subdir, scan_name=scan_stem)

        if len(scan) == 0:
            logger.warning("  Скан '%s' пуст — пропускаю", scan_stem)
            continue

        # 2. Сохраняем отдельный скан
        out_path = out_folder / f"{scan_stem}_merged.txt"
        scan.export_points_from_file(str(out_path))
        logger.info("  → %s", out_path.relative_to(BASE_DIR))

        # 3. Добавляем в общее облако эпохи
        for pt in scan:
            epoch_scan.add_point(pt)

    # 4. Сохраняем полное облако эпохи
    epoch_path = out_folder / f"epoch_{folder_name}_full.txt"
    epoch_scan.export_points_from_file(str(epoch_path))
    logger.info(
        "Эпоха '%s': %d точек → %s",
        folder_name, len(epoch_scan), epoch_path.relative_to(BASE_DIR),
    )

    return epoch_scan


def main():
    for folder_name in FOLDERS:
        process_folder(folder_name)
    logger.info("Готово. Результаты в: %s", MERGED_DIR)


if __name__ == "__main__":
    main()
