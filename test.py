from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.util.batch_cross_points import BatchCrossPointProcessor

processor = BatchCrossPointProcessor(
    input_dir="data/8_floors_wall/scan_2334_filt",
    output_dir="data/8_floors_wall/output",
    extensions=(".las", ".laz"),
    max_ellipsoid_axis=0.05,   # например 5 см
    eps=0.05,
    show_scans=False,
    choose_scan_directly_from_dbscan=True,
    mse_threshold=0.0001,
    max_iteration=20,
    k_sigma=2.0,
    base_fitter=PlaneL1Fitter,
)

result = processor.run()

df_all = result["df_all"]
df_good = result["df_good"]
df_good_filtered = result["df_good_filtered"]
df_errors = result["df_errors"]

print("Всего обработано:", len(df_all))
print("Хороших точек:", len(df_good))
print("Хороших после фильтра по эллипсоиду:", len(df_good_filtered))
print("Ошибок:", len(df_errors))