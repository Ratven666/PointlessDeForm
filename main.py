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
