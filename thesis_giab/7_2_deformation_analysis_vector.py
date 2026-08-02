"""
Анализ деформаций по МНК-оценке вектора смещения из изменений длин сегментов.

Модель: Δl_i = u_i^T · dX + v_i
  где u_i — единичный вектор сегмента (эпоха 1), dX = [dx, dy, dz]^T

Метод инвариантен к переносу и повороту СК.
Тесты: t-тест на каждую компоненту + χ²-тест конгруэнтности (df=3).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from app.cross_points.CrossPoint import CrossPoint
from app.cross_points.CrossPointSegmentSet import CrossPointSegmentSet
from app.deformation.SegmentPointDisplacementEstimator import SegmentPointDisplacementEstimator

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

ALPHA        = 0.05
USE_WEIGHTS  = True   # взвешивание по σ(Δl) = sqrt(σ_l1² + σ_l2²)

DISPLAY_COLS = [
    "point_name", "n_obs",
    "dx", "dy", "dz", "displacement_mm", "sigma_displacement_mm",
    "sigma_x", "sigma_y", "sigma_z",
    "sigma0",
    "t_x", "p_x", "significant_x",
    "t_y", "p_y", "significant_y",
    "t_z", "p_z", "significant_z",
    "t_disp", "p_disp", "significant_disp",
    "chi2_value", "p_chi2", "significant_chi2",
    "reliable", "message",
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

    # 3. МНК-оценка вектора смещения
    logger.info("МНК-оценка смещений (SegmentPointDisplacementEstimator, α=%.2f)...", ALPHA)
    estimator = SegmentPointDisplacementEstimator(alpha=ALPHA)
    results = estimator.estimate_for_all_points(
        seg_set_e1, seg_set_e2, use_weights=USE_WEIGHTS
    )
    df = SegmentPointDisplacementEstimator.results_to_dataframe(results)

    # 4. Сортировка: значимые по χ² → по убыванию |d|
    df_sorted = df.sort_values(
        ["significant_chi2", "displacement_mm"],
        ascending=[False, False],
        na_position="last",
    )

    # 5. Вывод таблицы
    available = [c for c in DISPLAY_COLS if c in df_sorted.columns]
    print_df = df_sorted[available].copy()
    for col in print_df.select_dtypes("float").columns:
        print_df[col] = print_df[col].map(lambda v: f"{v:.4g}" if v == v else "NaN")

    print(f"\n{'=' * 70}")
    print("ДЕФОРМАЦИИ — МНК-оценка вектора смещения по длинам сегментов")
    print("=" * 70)
    print(tabulate(print_df, headers="keys", tablefmt="pretty", showindex=False))
    print("=" * 70)

    # 6. Краткая статистика
    reliable_df = df_sorted[df_sorted["reliable"]]
    n_sig_t    = int(reliable_df["significant_disp"].sum())
    n_sig_chi2 = int(reliable_df["significant_chi2"].sum())
    logger.info("Итого: %d точек (%d надёжных)", len(df_sorted), len(reliable_df))
    logger.info("  Значимые смещения (t-тест, α=%.2f):  %d / %d",
                ALPHA, n_sig_t, len(reliable_df))
    logger.info("  Значимые смещения (χ²-тест, α=%.2f): %d / %d",
                ALPHA, n_sig_chi2, len(reliable_df))
    if len(reliable_df) > 0:
        logger.info(
            "  Медиана |d|: %.2f мм  |  Макс |d|: %.2f мм",
            reliable_df["displacement_mm"].median(),
            reliable_df["displacement_mm"].max(),
        )
        logger.info(
            "  Медиана σ₀: %.4f м  |  Макс σ₀: %.4f м",
            reliable_df["sigma0"].median(),
            reliable_df["sigma0"].max(),
        )

    # 7. Сохранение
    out_path = OUTPUT_DIR / "deformation_segment_displacement.csv"
    df_sorted.to_csv(out_path, index=False)
    logger.info("Результаты сохранены: %s", out_path)


if __name__ == "__main__":
    main()
