import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from app.base.scan.WeightedPointCloudRegistrator import WeightedPointCloudRegistrator
from app.base.scan.SpatialTransformation import SpatialTransformation
from app.cross_points.CrossPointListRestorer import CrossPointListRestorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────

BASE_DATA_DIR = Path("../data/8_floors_dvor")
CROSS_DIR     = BASE_DATA_DIR / "output" / "merged_cross_points"

# Входные CSV с хорошими (отфильтрованными) кросс-точками обеих эпох
CSV_EPOCH_1 = CROSS_DIR / "1" / "cross_points_good_filtered_by_ellipsoid.csv"
CSV_EPOCH_2 = CROSS_DIR / "2" / "cross_points_good_filtered_by_ellipsoid.csv"

# Куда кладём результаты выравнивания
OUTPUT_DIR = CROSS_DIR / "aligned"

# Параметры регистрации
METHOD   = "LSM"  # 'LSM' (взвешенный Кабш) | 'L1' (взвешенный IRLS)
K_SIGMA  = 5.0    # порог отбраковки по нормированным остаткам (None — отключить)
# K_SIGMA  = None
MAX_ITER = 100    # только для L1
EPS      = 1e-6
TOL      = 1e-9
MISSING_COV_STRATEGY = "min"  # 'median' | 'mean' | 'min' | 'unit'


# ──────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────

def load_cross_points(csv_path: Path) -> list:
    """Восстанавливает список CrossPoint из CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл кросс-точек не найден: {csv_path}")
    restorer = CrossPointListRestorer(str(csv_path))
    points = restorer.restore_all()
    logger.info("  Загружено точек из %s: %d", csv_path.name, len(points))
    return points


def transform_apply(points: list, T: SpatialTransformation) -> list:
    """
    Применяет SpatialTransformation к списку CrossPoint.
    Координаты обновляются in-place, ковариация переносится как R Σ R^T.
    """
    for pt in points:
        # трансформируем координаты
        xyz = np.array([pt.x, pt.y, pt.z])
        xyz_new = T.R @ xyz + T.t
        pt.x, pt.y, pt.z = float(xyz_new[0]), float(xyz_new[1]), float(xyz_new[2])

        # переносим ковариацию
        if getattr(pt, "cov_xyz", None) is not None:
            pt.cov_xyz = T.R @ pt.cov_xyz @ T.R.T

    return points


def cross_points_to_df(points: list, prefix: str = "") -> pd.DataFrame:
    """Конвертирует список CrossPoint в DataFrame для сохранения."""
    rows = []
    for pt in points:
        sigma_xyz = getattr(pt, "sigma_xyz", None)
        rows.append({
            "name":             pt.name,
            f"{prefix}x":      pt.x,
            f"{prefix}y":      pt.y,
            f"{prefix}z":      pt.z,
            "status":           getattr(pt, "status", None),
            "reliable_accuracy": getattr(pt, "reliable_accuracy", None),
            "sigma_x":          float(sigma_xyz[0]) if sigma_xyz is not None else None,
            "sigma_y":          float(sigma_xyz[1]) if sigma_xyz is not None else None,
            "sigma_z":          float(sigma_xyz[2]) if sigma_xyz is not None else None,
            "mse":              getattr(pt, "mse", None),
        })
    return pd.DataFrame(rows)


def save_transformation(T: SpatialTransformation, path: Path):
    """Сохраняет параметры трансформации в JSON."""
    data = {
        "method":         T.method,
        "n_common":       int(T.n_common),
        "n_used":         int(T.n_used),
        "rmse_m":         float(np.sqrt(np.mean(T.residuals ** 2))),
        "mae_m":          float(np.mean(np.abs(T.residuals))),
        "max_residual_m": float(np.max(np.abs(T.residuals))),
        "R":              T.R.tolist(),
        "t":              T.t.tolist(),
        "T_matrix":       T.T_matrix.tolist(),
        "omega_deg":      float(np.degrees(T.omega)),
        "phi_deg":        float(np.degrees(T.phi)),
        "kappa_deg":      float(np.degrees(T.kappa)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("  Трансформация сохранена: %s", path)


# ──────────────────────────────────────────────────────────────────────
# Главная логика
# ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Загружаем кросс-точки обеих эпох ──────────────────────────
    logger.info("Загрузка кросс-точек эпохи 1 (базовая)...")
    pts_epoch1 = load_cross_points(CSV_EPOCH_1)

    logger.info("Загрузка кросс-точек эпохи 2 (трансформируемая)...")
    pts_epoch2 = load_cross_points(CSV_EPOCH_2)

    # ── 2. Проверяем пересечение по именам ───────────────────────────
    names1 = {p.name for p in pts_epoch1}
    names2 = {p.name for p in pts_epoch2}
    common = names1 & names2
    only1  = names1 - names2
    only2  = names2 - names1

    logger.info("Общих точек: %d | только в эпохе 1: %d | только в эпохе 2: %d",
                len(common), len(only1), len(only2))

    if only1:
        logger.warning("  Только в эпохе 1: %s", sorted(only1))
    if only2:
        logger.warning("  Только в эпохе 2: %s", sorted(only2))

    if len(common) < 3:
        raise ValueError(
            f"Слишком мало общих точек для регистрации: {len(common)} (нужно >= 3)"
        )

    # ── 3. Регистрация: эпоха 2 → система эпохи 1 ────────────────────
    logger.info("=" * 60)
    logger.info("Регистрация эпохи 2 → эпоха 1  [метод=%s, k_sigma=%s]",
                METHOD, K_SIGMA)
    logger.info("=" * 60)

    registrator = WeightedPointCloudRegistrator(
        base_points      = pts_epoch1,
        transform_points = pts_epoch2,
        method           = METHOD,
        k_sigma          = K_SIGMA,
        max_iter         = MAX_ITER,
        eps              = EPS,
        tol              = TOL,
        missing_cov_strategy = MISSING_COV_STRATEGY,
    )

    T = registrator.compute()

    logger.info(
        "Результат регистрации:\n"
        "  RMSE     = %.4f мм\n"
        "  MAE      = %.4f мм\n"
        "  max_res  = %.4f мм\n"
        "  omega    = %.6f°\n"
        "  phi      = %.6f°\n"
        "  kappa    = %.6f°\n"
        "  tx=%.4f мм  ty=%.4f мм  tz=%.4f мм",
        float(np.sqrt(np.mean(T.residuals ** 2))) * 1000,
        float(np.mean(np.abs(T.residuals))) * 1000,
        float(np.max(np.abs(T.residuals))) * 1000,
        float(np.degrees(T.omega)),
        float(np.degrees(T.phi)),
        float(np.degrees(T.kappa)),
        float(T.t[0]) * 1000, float(T.t[1]) * 1000, float(T.t[2]) * 1000,
    )

    # ── 4. Применяем трансформацию к точкам эпохи 2 ──────────────────
    logger.info("Применение трансформации к кросс-точкам эпохи 2...")
    pts_epoch2_aligned = transform_apply(pts_epoch2, T)

    # ── 5. Сохраняем результаты ───────────────────────────────────────

    # Трансформация в JSON
    save_transformation(T, OUTPUT_DIR / "transformation_epoch2_to_epoch1.json")

    # Выровненные точки эпохи 2
    df_aligned = cross_points_to_df(pts_epoch2_aligned)
    df_aligned.to_csv(OUTPUT_DIR / "epoch2_aligned.csv", index=False)
    logger.info("  Выровненные точки эпохи 2: %s", OUTPUT_DIR / "epoch2_aligned.csv")

    # Точки эпохи 1 для удобства сравнения
    df_epoch1 = cross_points_to_df(pts_epoch1)
    df_epoch1.to_csv(OUTPUT_DIR / "epoch1_reference.csv", index=False)

    # Сводная таблица: общие точки обеих эпох рядом
    df_merged = pd.merge(
        df_epoch1.rename(columns={"name": "name",
                                  "x": "x1", "y": "y1", "z": "z1",
                                  "sigma_x": "sx1", "sigma_y": "sy1", "sigma_z": "sz1"}),
        df_aligned.rename(columns={"name": "name",
                                   "x": "x2", "y": "y2", "z": "z2",
                                   "sigma_x": "sx2", "sigma_y": "sy2", "sigma_z": "sz2"}),
        on="name", how="inner", suffixes=("_e1", "_e2"),
    )
    df_merged.to_csv(OUTPUT_DIR / "epochs_comparison.csv", index=False)
    logger.info("  Сводная таблица: %s", OUTPUT_DIR / "epochs_comparison.csv")

    logger.info("=" * 60)
    logger.info("Готово. Результаты в: %s", OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
