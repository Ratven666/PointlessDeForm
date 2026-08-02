# thesis_giab/2_deformation_analysis.py
"""
Анализ деформаций методом прямых разностей координат (d = X2 - X1).
Использует DeformationAnalyzer с t-тестом и χ²-тестом конгруэнтности.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from app.cross_points.CrossPoint import CrossPoint
from app.deformation.DeformationAnalyzer import DeformationAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

EPOCH1_CSV = Path("../data/8_floors_2/output/cross_points/aligned/epoch1_reference.csv")
EPOCH2_CSV = Path("../data/8_floors_2/output/cross_points/aligned/epoch2_aligned.csv")

OUTPUT_DIR = Path("../data/8_floors_2/output/deformation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05   # уровень значимости

# ──────────────────────────────────────────────
# Загрузка точек из CSV
# ──────────────────────────────────────────────

def load_points(csv_path: Path) -> list[CrossPoint]:
    """
    Читает CSV с колонками name, x, y, z, sigma_x, sigma_y, sigma_z.
    Восстанавливает список CrossPoint с диагональной ковариацией.
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
        cp.reliable_accuracy = bool(row.get("reliable_accuracy", True))

        sx = float(row["sigma_x"])
        sy = float(row["sigma_y"])
        sz = float(row["sigma_z"])
        cp.sigma_xyz = np.array([sx, sy, sz], dtype=float)
        cp.cov_xyz   = np.diag([sx**2, sy**2, sz**2])

        points.append(cp)

    reliable = sum(1 for p in points if getattr(p, "reliable_accuracy", False))
    logger.info("  %s: загружено %d точек (%d надёжных)",
                csv_path.name, len(points), reliable)
    return points


# ──────────────────────────────────────────────
# Основной пайплайн
# ──────────────────────────────────────────────

def main():
    logger.info("Загрузка данных...")
    points_e1 = load_points(EPOCH1_CSV)
    points_e2 = load_points(EPOCH2_CSV)

    # ── Анализ деформаций ────────────────────
    logger.info("Анализ деформаций (DeformationAnalyzer, α=%.2f)...", ALPHA)
    analyzer = DeformationAnalyzer(alpha=ALPHA)
    analyzer.analyze_point_sets(points_e1, points_e2)
    analyzer.print_summary()

    # ── Формирование DataFrame ───────────────
    df = analyzer.to_dataframe()

    DISPLAY_COLS = [
        "name",
        "displacement_mm", "sigma_displacement_mm",
        "dx", "dy", "dz",
        "sigma_dx", "sigma_dy", "sigma_dz",
        "t_value", "p_value_t",  "significant_t",
        "chi2_value", "p_value_chi2", "significant_chi2",
        "reliable",
    ]
    available = [c for c in DISPLAY_COLS if c in df.columns]

    df_sorted = df.sort_values("displacement", ascending=False)

    # Форматирование для печати
    print_df = df_sorted[available].copy()
    for col in print_df.select_dtypes("float").columns:
        print_df[col] = print_df[col].map(
            lambda v: f"{v:.5g}" if v == v else "NaN"
        )

    print("\n" + "=" * 70)
    print("ДЕФОРМАЦИИ — прямое сравнение координат")
    print("=" * 70)
    print(tabulate(print_df, headers="keys", tablefmt="pretty", showindex=False))
    print("=" * 70)

    # ── Краткая статистика ───────────────────
    n = len(df_sorted)
    n_sig_t    = df_sorted["significant_t"].sum()
    n_sig_chi2 = df_sorted["significant_chi2"].sum() if "significant_chi2" in df_sorted else 0
    logger.info("Итого: %d общих точек", n)
    logger.info("  Значимые смещения (t-тест,  α=%.2f): %d / %d", ALPHA, n_sig_t,    n)
    logger.info("  Значимые смещения (χ²-тест, α=%.2f): %d / %d", ALPHA, n_sig_chi2, n)
    logger.info(
        "  Медиана |d|: %.2f мм  |  Макс |d|: %.2f мм",
        df_sorted["displacement_mm"].median(),
        df_sorted["displacement_mm"].max(),
    )

    # ── Сохранение ───────────────────────────
    out_path = OUTPUT_DIR / "deformation_direct.csv"
    df_sorted.to_csv(out_path, index=False)
    logger.info("Результаты сохранены: %s", out_path)


if __name__ == "__main__":
    main()
