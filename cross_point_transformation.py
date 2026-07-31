from app.cross_points.CrossPointListRestorer import CrossPointListRestorer
from app.base.scan.PointCloudRegistrator import PointCloudRegistrator

# base_points_path = "data/8_floors_wall/output/scan_2335_filt/cross_points_good_filtered_by_ellipsoid.csv"
# moving_point_path_l = "data/8_floors_wall/output/scan_2334_filt/cross_points_good_filtered_by_ellipsoid.csv"
# moving_point_path_r = "data/8_floors_wall/output/scan_2336_filt/cross_points_good_filtered_by_ellipsoid.csv"

base_points_path = "data/8_floors_wall/output/scan_2335_filt/cross_points_good_filtered_by_ellipsoid.csv"
moving_point_path_l = "data/8_floors_wall/output/scan_2334_filt/cross_points_good.csv"
# moving_point_path_r = "data/8_floors_wall/output/scan_2336_filt/cross_points_good.csv"



base_points_restorer = CrossPointListRestorer(base_points_path)
moving_point_restorer_l = CrossPointListRestorer(moving_point_path_l)
# moving_point_restorer_r = CrossPointListRestorer(moving_point_path_r)

base_points = base_points_restorer.restore_all()
moving_points = moving_point_restorer_l.restore_all()
# moving_points = moving_point_restorer_r.restore_all()

# два списка CrossPoint
reg_lsm = PointCloudRegistrator(base_points, moving_points, method="LSM")
transform_lsm = reg_lsm.compute()
print(transform_lsm)

# reg_l1 = PointCloudRegistrator(base_points, moving_points, method="L1")
# transform_l1 = reg_l1.compute()
# print(transform_l1)
#
# Применить трансформацию ко всем точкам
transformed_points = transform_lsm.transform_points(moving_points)

# print(*moving_points, sep="\n")
# print(*transformed_points, sep="\n")