from app.util.CrossPointExacter import CrossPointExacter

# FILE_PATH = "data/8_floors_wall/test_window/2_1_vp_good.txt"
vl_path = "data/8_floors_wall/test_window/8_6_vl.txt"
vp_path = "data/8_floors_wall/test_window/8_6_vp.txt"
nl_path = "data/8_floors_wall/test_window/8_6_nl.txt"
np_path = "data/8_floors_wall/test_window/8_6_np.txt"

# vl_path = "data/8_floors_wall/test_window/2_3_vl.txt"
# vp_path = "data/8_floors_wall/test_window/2_3_vp.txt"
# nl_path = "data/8_floors_wall/test_window/2_3_nl.txt"
# np_path = "data/8_floors_wall/test_window/2_3_np.txt"

window_pathes = [vl_path, vp_path, nl_path, np_path]

planes = []
cpoints = []
for path in window_pathes:
    cpe = CrossPointExacter(file_path=path,
                            choose_scan_directly_from_dbscan=False,
                            show_scans=False,
                            eps=0.05)
    cpe.calculate_planes()
    planes.append(cpe.planes)
    # for plane in cpe.planes:
    #     print(plane)
    point, status = cpe.calculate_intersect_point()
    cpoints.append((point, status))
    res_str = cpe.get_result_str()

for idx in range(len(cpoints)):
    print(*planes[idx], sep="\n")
    print("\t", *cpoints[idx], "\n============================")