"""
Вычисление деформаций тремя методами:
  1. DeformationAnalyzer          — прямое сравнение координат (d = X2 - X1)
  2. SegmentLengthDeformationAnalyzer — изменения длин сегментов
  3. SegmentPointDisplacementEstimator — МНК-оценка вектора смещения по длинам
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from app.cross_points.CrossPoint import CrossPoint
from app.cross_points.CrossPointSegmentSet import CrossPointSegmentSet
from app.deformation.DeformationAnalyzer import DeformationAnalyzer
from app.deformation.SegmentLengthDeformationAnalyzer import SegmentLengthDeformationAnalyzer
from app.deformation.SegmentPointDisplacementEstimator import SegmentPointDisplacementEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────

# EPOCH1_CSV = Path("../data/8_floors_dvor/output/merged_cross_points/aligned/epoch1_reference.csv")
# EPOCH2_CSV = Path("../data/8_floors_dvor/output/merged_cross_points/aligned/epoch2_aligned.csv")

EPOCH1_CSV = Path("../data/8_floors_1/output/cross_points/aligned/epoch1_reference.csv")
EPOCH2_CSV = Path("../data/8_floors_1/output/cross_points/aligned/epoch2_aligned.csv")

# OUTPUT_DIR = Path("../data/8_floors_dvor/output/deformation")
OUTPUT_DIR = Path("../data/8_floors_1/output/deformation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

# SegmentLengthDeformationAnalyzer
SEG_LENGTH_USE_WEIGHTS      = True
SEG_LENGTH_ENABLE_REJECTION = True
SEG_LENGTH_REJECTION_THRESH = 3.0
SEG_LENGTH_MIN_OBS          = 3

# SegmentPointDisplacementEstimator
SEG_DISP_USE_WEIGHTS = True


# ──────────────────────────────────────────────────────────
# 0. Загрузка точек из CSV
#    CrossPointListRestorer ожидает колонки cross_point_name/x/y/z,
#    но наши CSV имеют name/x/y/z — читаем вручную.
# ──────────────────────────────────────────────────────────

def load_points(csv_path: Path) -> list[CrossPoint]:
    """
    Читает CSV с колонками:
        name, x, y, z, status, reliable_accuracy,
        sigma_x, sigma_y, sigma_z, mse
    Восстанавливает список CrossPoint с sigma_xyz и диагональной cov_xyz.
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
        cp.status = row.get("status", "GOOD")
        cp.reliable_accuracy = bool(row.get("reliable_accuracy", False))

        sx = float(row["sigma_x"])
        sy = float(row["sigma_y"])
        sz = float(row["sigma_z"])
        cp.sigma_xyz = np.array([sx, sy, sz], dtype=float)
        # Диагональная ковариационная матрица (внеосевые корреляции не сохранены)
        cp.cov_xyz = np.diag([sx**2, sy**2, sz**2])

        points.append(cp)

    reliable = sum(1 for p in points if getattr(p, "reliable_accuracy", False))
    logger.info("  %s: загружено %d точек (%d надёжных)", csv_path.name, len(points), reliable)
    return points


# ──────────────────────────────────────────────────────────
# Вспомогательная функция вывода и сохранения
# ──────────────────────────────────────────────────────────

def report(method_name: str, df: pd.DataFrame, cols: list[str], csv_name: str) -> None:
    logger.info("\n%s", "=" * 70)
    logger.info("  %s", method_name)
    logger.info("%s", "=" * 70)

    available = [c for c in cols if c in df.columns]
    sub = df[available].copy()

    for col in sub.select_dtypes("float").columns:
        sub[col] = sub[col].map(lambda x: f"{x:.5g}" if x == x else "NaN")

    print(tabulate(sub, headers="keys", tablefmt="pretty", showindex=False))

    out_path = OUTPUT_DIR / csv_name
    df.to_csv(out_path, index=False)
    logger.info("  CSV сохранён: %s", out_path)


DISPLAY_COLS = {
    "method1": [
        "name",
        "displacement_mm", "sigma_displacement_mm",
        "dx", "dy", "dz",
        "t_value", "p_value_t", "significant_t",
        "chi2_value", "p_value_chi2", "significant_chi2",
        "reliable",
    ],
    "method2": [
        "point_name", "n_obs_used", "n_rejected",
        "mean_delta_length_mm", "rms_delta_length_mm", "max_abs_delta_length_mm",
        "test_statistic", "p_value", "significant",
        "reliable",
    ],
    "method3": [
        "point_name", "n_obs",
        "dx", "dy", "dz", "displacement_mm", "sigma_displacement_mm",
        "sigma0",
        "t_disp", "p_disp", "significant_disp",
        "chi2_value", "p_chi2", "significant_chi2",
        "reliable",
    ],
}


# ──────────────────────────────────────────────────────────
# Загрузка данных
# ──────────────────────────────────────────────────────────

logger.info("Загрузка данных...")
points_e1 = load_points(EPOCH1_CSV)
points_e2 = load_points(EPOCH2_CSV)


# ──────────────────────────────────────────────────────────
# 1. Метод 1 — DeformationAnalyzer (прямое сравнение координат)
# ──────────────────────────────────────────────────────────

logger.info("\n[1/3] DeformationAnalyzer — прямое сравнение координат...")

analyzer = DeformationAnalyzer(alpha=ALPHA)
analyzer.analyze_point_sets(points_e1, points_e2)
analyzer.print_summary()

df1 = analyzer.to_dataframe()
df1_sorted = df1.sort_values("displacement", ascending=False)
report(
    "Метод 1 — Прямое сравнение координат (DeformationAnalyzer)",
    df1_sorted,
    DISPLAY_COLS["method1"],
    "method1_direct_comparison.csv",
)


# ──────────────────────────────────────────────────────────
# 2. Метод 2 — SegmentLengthDeformationAnalyzer (изменения длин)
# ──────────────────────────────────────────────────────────

logger.info("\n[2/3] SegmentLengthDeformationAnalyzer — изменения длин сегментов...")

seg_set_e1 = CrossPointSegmentSet.from_all_pairs(points_e1)
seg_set_e2 = CrossPointSegmentSet.from_all_pairs(points_e2)
logger.info(
    "  Сегментов: эпоха 1 = %d  |  эпоха 2 = %d",
    len(seg_set_e1), len(seg_set_e2),
)

len_analyzer = SegmentLengthDeformationAnalyzer(
    alpha=ALPHA,
    use_weights=SEG_LENGTH_USE_WEIGHTS,
    enable_rejection=SEG_LENGTH_ENABLE_REJECTION,
    rejection_threshold=SEG_LENGTH_REJECTION_THRESH,
    min_obs=SEG_LENGTH_MIN_OBS,
)
results2 = len_analyzer.analyze_for_all_points(seg_set_e1, seg_set_e2)
df2 = SegmentLengthDeformationAnalyzer.results_to_dataframe(results2)
df2_sorted = df2.sort_values(
    "mean_delta_length", ascending=False, key=lambda s: s.abs()
)
report(
    "Метод 2 — Изменения длин сегментов (SegmentLengthDeformationAnalyzer)",
    df2_sorted,
    DISPLAY_COLS["method2"],
    "method2_segment_lengths.csv",
)

n_sig2 = df2["significant"].sum()
logger.info("  Значимых деформаций (χ²): %d / %d", n_sig2, len(df2))


# ──────────────────────────────────────────────────────────
# 3. Метод 3 — SegmentPointDisplacementEstimator (МНК по длинам)
# ──────────────────────────────────────────────────────────

logger.info("\n[3/3] SegmentPointDisplacementEstimator — МНК-оценка вектора смещения...")

# use_weights — параметр метода estimate_for_all_points, не конструктора
estimator = SegmentPointDisplacementEstimator(alpha=ALPHA)
results3 = estimator.estimate_for_all_points(
    seg_set_e1, seg_set_e2, use_weights=SEG_DISP_USE_WEIGHTS
)
df3 = SegmentPointDisplacementEstimator.results_to_dataframe(results3)
df3_sorted = df3.sort_values("displacement", ascending=False)
report(
    "Метод 3 — МНК-оценка вектора смещения (SegmentPointDisplacementEstimator)",
    df3_sorted,
    DISPLAY_COLS["method3"],
    "method3_segment_displacement.csv",
)

n_sig3 = df3["significant_disp"].sum()
logger.info("  Значимых смещений (t-тест): %d / %d", n_sig3, len(df3))


# ──────────────────────────────────────────────────────────
# 4. Сводная таблица по всем методам
# ──────────────────────────────────────────────────────────

logger.info("\n[4/4] Сводная таблица...")

summary = (
    df1[["name", "displacement_mm", "significant_t", "significant_chi2"]]
    .rename(columns={
        "name":           "point",
        "displacement_mm": "m1_disp_mm",
        "significant_t":  "m1_sig_t",
        "significant_chi2": "m1_sig_chi2",
    })
)

m2 = (
    df2[["point_name", "mean_delta_length_mm", "significant"]]
    .rename(columns={
        "point_name":          "point",
        "mean_delta_length_mm": "m2_mean_dl_mm",
        "significant":         "m2_sig",
    })
)

m3 = (
    df3[["point_name", "displacement_mm", "significant_disp", "significant_chi2"]]
    .rename(columns={
        "point_name":      "point",
        "displacement_mm": "m3_disp_mm",
        "significant_disp": "m3_sig_t",
        "significant_chi2": "m3_sig_chi2",
    })
)

summary = summary.merge(m2, on="point", how="outer")
summary = summary.merge(m3, on="point", how="outer")
summary = summary.sort_values("m1_disp_mm", ascending=False, na_position="last")

summary_path = OUTPUT_DIR / "summary_all_methods.csv"
summary.to_csv(summary_path, index=False)
logger.info("  Сводная таблица сохранена: %s", summary_path)

print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА — ВСЕ ТРИ МЕТОДА")
print("=" * 70)
print(tabulate(summary, headers="keys", tablefmt="pretty", showindex=False))
print("=" * 70)
print(f"\nФайлы сохранены в: {OUTPUT_DIR}/")
