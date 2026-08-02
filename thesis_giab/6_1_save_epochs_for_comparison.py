"""
Подготовка точек к деформационному анализу:
  — загрузка кросс-точек двух эпох из CSV
  — сохранение в output/cross_points/aligned/ без какого-либо преобразования координат
  — формирование сводной таблицы общих точек epochs_comparison.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.cross_points.CrossPointListRestorer import CrossPointListRestorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

BASE_DATA_DIR = Path("../data/8_floors_2")
CROSS_DIR     = BASE_DATA_DIR / "output" / "cross_points"
OUTPUT_DIR    = CROSS_DIR / "aligned"

# CSV_EPOCH_1 = CROSS_DIR / "0_1" / "scan_2335" / "cross_points_good_filtered_by_ellipsoid.csv"
# CSV_EPOCH_2 = CROSS_DIR / "0_2" / "367_o_las"  / "cross_points_good_filtered_by_ellipsoid.csv"

CSV_EPOCH_1 = CROSS_DIR / "1" / "367_o_las" / "cross_points_good_filtered_by_ellipsoid.csv"
CSV_EPOCH_2 = CROSS_DIR / "2" / "368_o_las"  / "cross_points_good_filtered_by_ellipsoid.csv"


# ──────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────

def load_cross_points(csv_path: Path) -> list:
    """Восстанавливает список CrossPoint из CSV через CrossPointListRestorer."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл кросс-точек не найден: {csv_path}")
    points = CrossPointListRestorer(str(csv_path)).restore_all()
    logger.info("  %s: загружено %d точек", csv_path.name, len(points))
    return points


def cross_points_to_df(points: list) -> pd.DataFrame:
    """Конвертирует список CrossPoint в DataFrame для сохранения."""
    rows = []
    for pt in points:
        sigma = getattr(pt, "sigma_xyz", None)
        rows.append({
            "name":              pt.name,
            "x":                 pt.x,
            "y":                 pt.y,
            "z":                 pt.z,
            "status":            getattr(pt, "status", None),
            "reliable_accuracy": getattr(pt, "reliable_accuracy", None),
            "sigma_x":           float(sigma[0]) if sigma is not None else None,
            "sigma_y":           float(sigma[1]) if sigma is not None else None,
            "sigma_z":           float(sigma[2]) if sigma is not None else None,
            "mse":               getattr(pt, "mse", None),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Основной пайплайн
# ──────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Загружаем точки
    logger.info("Загрузка кросс-точек...")
    pts_e1 = load_cross_points(CSV_EPOCH_1)
    pts_e2 = load_cross_points(CSV_EPOCH_2)

    # 2. Проверяем пересечение по именам
    names1 = {p.name for p in pts_e1}
    names2 = {p.name for p in pts_e2}
    common = names1 & names2
    only1  = names1 - names2
    only2  = names2 - names1

    logger.info(
        "Общих точек: %d | только в эпохе 1: %d | только в эпохе 2: %d",
        len(common), len(only1), len(only2),
    )
    if only1:
        logger.warning("  Только в эпохе 1: %s", sorted(only1))
    if only2:
        logger.warning("  Только в эпохе 2: %s", sorted(only2))

    # 3. Сохраняем точки эпохи 1 (опорные)
    df_e1 = cross_points_to_df(pts_e1)
    path_e1 = OUTPUT_DIR / "epoch1_reference.csv"
    df_e1.to_csv(path_e1, index=False)
    logger.info("  Эпоха 1 сохранена: %s (%d точек)", path_e1.name, len(df_e1))

    # 4. Сохраняем точки эпохи 2 (без трансформации)
    df_e2 = cross_points_to_df(pts_e2)
    path_e2 = OUTPUT_DIR / "epoch2_aligned.csv"
    df_e2.to_csv(path_e2, index=False)
    logger.info("  Эпоха 2 сохранена: %s (%d точек)", path_e2.name, len(df_e2))

    # 5. Сводная таблица общих точек рядом
    df_merged = pd.merge(
        df_e1.rename(columns={"x": "x1", "y": "y1", "z": "z1",
                               "sigma_x": "sx1", "sigma_y": "sy1", "sigma_z": "sz1",
                               "mse": "mse1"}),
        df_e2.rename(columns={"x": "x2", "y": "y2", "z": "z2",
                               "sigma_x": "sx2", "sigma_y": "sy2", "sigma_z": "sz2",
                               "mse": "mse2",
                               "status": "status_e2",
                               "reliable_accuracy": "reliable_accuracy_e2"}),
        on="name", how="inner",
    )
    path_merged = OUTPUT_DIR / "epochs_comparison.csv"
    df_merged.to_csv(path_merged, index=False)
    logger.info(
        "  Сводная таблица: %s (%d общих точек)", path_merged.name, len(df_merged)
    )

    logger.info("Готово. Результаты в: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
