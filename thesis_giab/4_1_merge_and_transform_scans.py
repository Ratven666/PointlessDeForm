"""
Шаг 4: Сборка и выравнивание облаков точек.

Логика для каждой папки (1 и 2):
──────────────────────────────────────────────────────────────────────
  Для каждого скана (360_o_las, 362_o_las, …):
    1. Собрать общий скан из всех субсканов подпапки subscans/<folder>/<scan_stem>/
    2. Применить трансформацию из transforms_folder_<folder>.json
       (для опорного скана — тождественная, ничего не делаем)
    3. Сохранить выровненный скан в output/merged/<folder>/<scan_stem>_registered.txt

  Затем слить все выровненные сканы папки в один:
    output/merged/<folder>/epoch_<folder>_full.txt

Структура файлов:
  data/8_floors_dvor/
    output/
      subscans/<folder>/<scan_stem>/<point_name>.txt   ← субсканы
      registration/transforms_folder_<folder>.json     ← трансформации
      merged/<folder>/<scan_stem>_registered.txt       ← выровненные сканы
      merged/<folder>/epoch_<folder>_full.txt          ← общее облако эпохи
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.base.scan.Scan import Scan
from app.base.scan.SpatialTransformation import SpatialTransformation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────

BASE_DIR       = Path("../data/8_floors_dvor")
SUBSCANS_DIR   = BASE_DIR / "output" / "subscans"
REG_DIR        = BASE_DIR / "output" / "registration"
MERGED_DIR     = BASE_DIR / "output" / "merged"
# FOLDERS        = ["1", "2"]
FOLDERS        = ["all"]
SCAN_EXTENSION = {".las", ".laz", ".txt", ".xyz", ".pts"}


# ──────────────────────────────────────────────────────────────────────
# Загрузка трансформаций из JSON
# ──────────────────────────────────────────────────────────────────────

def load_transforms(folder_name: str) -> dict[str, SpatialTransformation | None]:
    """
    Читает transforms_folder_<folder>.json.
    Возвращает dict {scan_stem: SpatialTransformation | None}.
    None — для опорного скана (тождественная трансформация).
    """
    json_path = REG_DIR / f"transforms_folder_{folder_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Файл трансформаций не найден: {json_path}\n"
            f"Сначала запустите 3_register_scans.py"
        )

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    transforms: dict[str, SpatialTransformation | None] = {}
    for scan_stem, info in data.items():
        if info.get("anchor"):
            transforms[scan_stem] = None   # опорный скан — трансформация не нужна
        else:
            R = np.array(info["R"], dtype=float)
            t = np.array(info["t"], dtype=float)
            residuals = np.array(info.get("residuals", [0.0]), dtype=float)

            T = SpatialTransformation(
                R=R,
                t=t,
                method=info.get("method", "LSM"),
                n_common=int(info.get("n_common", 0)),
                n_used=int(info.get("n_used", 0)),
                residuals=residuals,
            )
            transforms[scan_stem] = T

    logger.info(
        "Папка '%s': загружено трансформаций: %d (anchor: %d)",
        folder_name,
        len(transforms),
        sum(1 for v in transforms.values() if v is None),
    )
    return transforms


# ──────────────────────────────────────────────────────────────────────
# Сборка скана из субсканов
# ──────────────────────────────────────────────────────────────────────

def merge_subscans(scan_subdir: Path, scan_name: str) -> Scan:
    """
    Собирает один Scan из всех файлов в папке scan_subdir.
    Нормали пересчитываются один раз после сборки (compute_normals=False при загрузке).
    """
    merged = Scan(scan_name=scan_name)

    sub_files = sorted(
        p for p in scan_subdir.iterdir()
        if p.is_file() and p.suffix.lower() in SCAN_EXTENSION
    )

    if not sub_files:
        logger.warning("  Нет файлов субсканов в: %s", scan_subdir)
        return merged

    for sub_file in sub_files:
        sub_scan = Scan(scan_name=sub_file.stem)
        # compute_normals=False — сэкономим время, нормали посчитаем после слияния
        sub_scan.import_points_from_file(str(sub_file), compute_normals=False)

        for point in sub_scan:
            merged.add_point(point)

    logger.info(
        "  Собран скан %-30s : %d точек из %d субсканов",
        scan_name, len(merged), len(sub_files),
    )
    return merged


# ──────────────────────────────────────────────────────────────────────
# Основная логика
# ──────────────────────────────────────────────────────────────────────

def process_folder(folder_name: str):
    logger.info("=" * 62)
    logger.info("Обработка папки: %s", folder_name)
    logger.info("=" * 62)

    # 1. Загружаем трансформации
    transforms = load_transforms(folder_name)

    # 2. Папка для выходных данных
    out_folder = MERGED_DIR / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    # 3. Общий скан эпохи
    epoch_scan = Scan(scan_name=f"epoch_{folder_name}_full")

    # 4. Обходим сканы
    subscans_folder = SUBSCANS_DIR / folder_name

    if not subscans_folder.exists():
        raise FileNotFoundError(
            f"Папка субсканов не найдена: {subscans_folder}\n"
            f"Сначала запустите 1_get_subscans_around_points.py"
        )

    # Порядок: сначала опорный (anchor), потом остальные
    anchor_stems = [s for s, T in transforms.items() if T is None]
    other_stems  = [s for s, T in transforms.items() if T is not None]
    ordered_stems = anchor_stems + other_stems

    # Сканы, которые есть в subscans_folder, но не попали в transforms
    known_stems = set(ordered_stems)
    extra_subdirs = [
        d for d in sorted(subscans_folder.iterdir())
        if d.is_dir() and d.name not in known_stems
    ]
    if extra_subdirs:
        logger.warning(
            "Следующие папки субсканов не имеют трансформации "
            "(пропускаются): %s",
            [d.name for d in extra_subdirs],
        )

    for scan_stem in ordered_stems:
        scan_subdir = subscans_folder / scan_stem

        if not scan_subdir.exists() or not scan_subdir.is_dir():
            logger.warning(
                "Подпапка субсканов не найдена: %s — пропускаю", scan_subdir
            )
            continue

        # ── 4a. Собираем скан из субсканов ───────────────────────────
        raw_scan = merge_subscans(scan_subdir, scan_name=scan_stem)

        if len(raw_scan) == 0:
            logger.warning("  Скан %s пуст — пропускаю", scan_stem)
            continue

        # ── 4b. Применяем трансформацию ──────────────────────────────
        T = transforms[scan_stem]

        if T is None:
            # Опорный скан — трансформация тождественная
            registered_scan = Scan(scan_name=f"{scan_stem}_registered")
            for pt in raw_scan:
                registered_scan.add_point(pt)
            logger.info(
                "  %-30s : anchor, трансформация не применяется",
                scan_stem,
            )
        else:
            registered_scan = raw_scan.transform_scan(
                transformation=T,
                inplace=False,
                rotate_normals=True,
                scan_name=f"{scan_stem}_registered",
            )
            logger.info(
                "  %-30s : трансформирован | RMSE=%.6f m | "
                "tx=%.4f ty=%.4f tz=%.4f",
                scan_stem, T.rmse, T.tx, T.ty, T.tz,
            )

        # ── 4c. Сохраняем выровненный скан ───────────────────────────
        out_path = out_folder / f"{scan_stem}_registered.txt"
        registered_scan.export_points_from_file(str(out_path))
        logger.info("  Сохранён: %s (%d точек)", out_path.name, len(registered_scan))

        # ── 4d. Добавляем в общий скан эпохи ─────────────────────────
        for pt in registered_scan:
            epoch_scan.add_point(pt)

    # 5. Сохраняем общее облако эпохи
    epoch_path = out_folder / f"epoch_{folder_name}_full.txt"
    epoch_scan.export_points_from_file(str(epoch_path))

    logger.info(
        "Эпоха '%s' собрана: %d точек → %s",
        folder_name, len(epoch_scan), epoch_path,
    )
    return epoch_scan


def main():
    for folder_name in FOLDERS:
        process_folder(folder_name)

    logger.info("Готово. Результаты в: %s", MERGED_DIR)


if __name__ == "__main__":
    main()
