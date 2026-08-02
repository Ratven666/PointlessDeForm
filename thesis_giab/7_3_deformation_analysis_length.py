# thesis_giab/7_3_deformation_analysis_length.py
"""
Анализ деформаций по изменениям длин сегментов между опорными точками.
Метод инвариантен к переносу и повороту СК.

Использует SegmentLengthDeformationAnalyzer:
  - χ²-тест H0: все Δl_i = 0  (T = Σ w_i·Δl_i² ~ χ²(n))
  - итерационная отбраковка аномальных сегментов z = |Δl| / σ(Δl) > rejection_threshold
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from app.cross_points.CrossPoint import CrossPoint
from app.cross_points.CrossPointSegmentSet import CrossPointSegmentSet
from app.deformation.SegmentLengthDeformationAnalyzer import SegmentLengthDeformationAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────

EPOCH1_CSV = Path("../data/8_floors_2/output/cross_points/aligned/epoch1_reference.csv")
EPOCH2_CSV = Path("../data/8_floors_2/output/cross_points/aligned/epoch2_aligned.csv")
OUTPUT_DIR = Path("../data/8_floors_2/output/deformation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA              = 0.05
USE_WEIGHTS        = True
ENABLE_REJECTION   = True
REJECTION_THRESH   = 3.0
# REJECTION_THRESH   = 4.0
MIN_OBS            = 3
MAX_REJECTIONS     = None   # None — без ограничения

DISPLAY_COLS = [
    "point_name", "n_obs_initial", "n_obs_used", "n_rejected",
    "mean_delta_length_mm", "rms_delta_length_mm", "max_abs_delta_length_mm",
    "test_statistic", "p_value", "significant",
    "reliable", "rejected_neighbors",
]

# ──────────────────────────────────────────────────────────
# Загрузка точек
# ──────────────────────────────────────────────────────────

def load_points(csv_path: Path) -> list[CrossPoint]:
    """
    Читает CSV с колонками name, x, y, z, sigma_x, sigma_y, sigma_z.
    Строит CrossPoint с sigma_xyz и диагональной cov_xyz.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    df = pd.read_csv(csv_path)
    points: list[CrossPoint] = []

    for _, row in df.iterrows():
        cp = CrossPoint(
            name=str(row["name"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
        )
        cp.status            = row.get("status", "GOOD")
        cp.reliable_accuracy = bool(row.get("reliable_accuracy", False))

        sx, sy, sz = float(row["sigma_x"]), float(row["sigma_y"]), float(row["sigma_z"])
        cp.sigma_xyz = np.array([sx, sy, sz], dtype=float)
        cp.cov_xyz   = np.diag([sx**2, sy**2, sz**2])

        points.append(cp)

    reliable = sum(1 for p in points if getattr(p, "reliable_accuracy", False))
    logger.info("  %s: загружено %d точек (%d надёжных)", csv_path.name, len(points), reliable)
    return points


# ──────────────────────────────────────────────────────────
# Основной пайплайн
# ──────────────────────────────────────────────────────────

def main():
    # 1. Загрузка
    logger.info("Загрузка данных...")
    points_e1 = load_points(EPOCH1_CSV)
    points_e2 = load_points(EPOCH2_CSV)

    # 2. Построение графов сегментов
    seg_set_e1 = CrossPointSegmentSet.from_all_pairs(points_e1)
    seg_set_e2 = CrossPointSegmentSet.from_all_pairs(points_e2)
    logger.info("Сегментов: эпоха 1 = %d  |  эпоха 2 = %d",
                len(seg_set_e1), len(seg_set_e2))

    # 3. Анализ
    logger.info(
        "Анализ деформаций (SegmentLengthDeformationAnalyzer, "
        "α=%.2f, rejection=%s @ %.1fσ, min_obs=%d)...",
        ALPHA, ENABLE_REJECTION, REJECTION_THRESH, MIN_OBS,
    )
    analyzer = SegmentLengthDeformationAnalyzer(
        alpha=ALPHA,
        use_weights=USE_WEIGHTS,
        enable_rejection=ENABLE_REJECTION,
        rejection_threshold=REJECTION_THRESH,
        min_obs=MIN_OBS,
        max_rejections=MAX_REJECTIONS,
    )
    results = analyzer.analyze_for_all_points(seg_set_e1, seg_set_e2)
    df = SegmentLengthDeformationAnalyzer.results_to_dataframe(results)

    # 4. Сортировка: значимые → по убыванию RMS
    df_sorted = df.sort_values(
        ["significant", "rms_delta_length_mm"],
        ascending=[False, False],
        na_position="last",
    )

    # 5. Вывод таблицы
    available = [c for c in DISPLAY_COLS if c in df_sorted.columns]
    print_df = df_sorted[available].copy()
    for col in print_df.select_dtypes("float").columns:
        print_df[col] = print_df[col].map(lambda v: f"{v:.4g}" if v == v else "NaN")

    print(f"\n{'=' * 70}")
    print("ДЕФОРМАЦИИ — изменения длин сегментов (Δl = l₂ − l₁)")
    print("=" * 70)
    print(tabulate(print_df, headers="keys", tablefmt="pretty", showindex=False))
    print("=" * 70)

    # 6. Краткая статистика
    reliable_df = df_sorted[df_sorted["reliable"]]
    n_sig = int(df_sorted["significant"].sum())
    logger.info("Итого: %d точек (%d надёжных)", len(df_sorted), len(reliable_df))
    logger.info("  Значимые деформации (χ²-тест, α=%.2f): %d / %d",
                ALPHA, n_sig, len(reliable_df))
    if len(reliable_df) > 0:
        logger.info(
            "  Медиана RMS(Δl): %.2f мм  |  Макс RMS(Δl): %.2f мм",
            reliable_df["rms_delta_length_mm"].median(),
            reliable_df["rms_delta_length_mm"].max(),
        )
        logger.info(
            "  Медиана mean(Δl): %.2f мм  |  Макс |mean(Δl)|: %.2f мм",
            reliable_df["mean_delta_length_mm"].median(),
            reliable_df["mean_delta_length_mm"].abs().max(),
        )

    # 7. Сохранение
    out_path = OUTPUT_DIR / "deformation_segment_lengths.csv"
    df_sorted.to_csv(out_path, index=False)
    logger.info("Результаты сохранены: %s", out_path)


if __name__ == "__main__":
    main()