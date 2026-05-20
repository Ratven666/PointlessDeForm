# import os
#
# from app.util.CrossPointExacter import CrossPointExacter
#
# # base_path = "data/200226/сканер/lazpredobr"
# base_path = "data/lazpredobr"
# target_path = "src"
#
# files = [
#     f for f in os.listdir(base_path)
#     if os.path.isfile(os.path.join(base_path, f))
# ]
#
# with open(os.path.join(target_path, "cross_points.txt"), "a") as write_file:
#     write_file.write(f"Scan_name, X, Y, Z\n")
#     for file in files:
#         file_path = os.path.join(base_path, file)
#         cpe = CrossPointExacter(file_path=file_path, eps=0.005)
#         cpe.calculate_planes()
#         for plane in cpe.planes:
#             print(plane)
#         x = cpe.calculate_intersect_point()
#         print(x)
#         res_str = cpe.get_result_str()
#         write_file.write(f"{res_str}\n")

from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.util.batch_cross_points import BatchCrossPointProcessor

processor = BatchCrossPointProcessor(
    input_dir="data/8_floors_wall/scan_2336_filt",
    output_dir="data/8_floors_wall/output/scan_2336_filt",
    extensions=(".las", ".laz", ".txt"),
    max_ellipsoid_axis=0.005,   # например 5 см
    eps=0.05,
    show_scans=False,
    choose_scan_directly_from_dbscan=False,
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