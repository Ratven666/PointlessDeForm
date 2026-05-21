from app.base.scan.Scan import Scan
from app.util.sub_scan_separator import get_named_point_list, split_scan_by_points, save_sub_scans

# total_scan = Scan(scan_name="total_scan")
#
# total_scan.import_points_from_file(file_path="scan_2334_filt_transformed_transformed.txt")
# print(total_scan)
# total_scan.import_points_from_file(file_path="scan_2336_filt_transformed_transformed.txt")
# print(total_scan)
# total_scan.import_points_from_file(file_path="data/8_floors_wall/scan_2335_filt.txt")
# print(total_scan)
#
# point_list_1 = get_named_point_list("data/8_floors_wall/2334_tochki.txt")
# point_list_2 = get_named_point_list("data/8_floors_wall/2335_tochki.txt")
# point_list_3 = get_named_point_list("data/8_floors_wall/2336_tochki.txt")
#
# point_dict = {}
#
# for point_list in [point_list_1, point_list_2, point_list_3]:
#     for point in point_list:
#         point_dict[point.name] = point
#
# sub_scans = split_scan_by_points(total_scan, point_dict.values(), cube_size=0.6)
# save_sub_scans(sub_scans, total_scan)


from app.base.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.util.batch_cross_points import BatchCrossPointProcessor

processor = BatchCrossPointProcessor(
    input_dir="data/8_floors_wall/total_scan",
    output_dir="data/8_floors_wall/output/total_scan",
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