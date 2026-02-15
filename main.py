from app.base.scan.Scan import Scan
from app.base.scan.filters.ScanFilterByDBSCAN import ScanFilterByDBSCAN
from app.base.scan.plotters.ScanPlotterWithLabelsMPL import ScanPlotterWithLabelsMPL
from app.base.scan.utils.ScanNormalsDirectionClassifier import ScanNormalsDirectionClassifier
from app.base.scan.utils.ScanSplitterByLabels import ScanSplitterByLabels

scan = Scan("l1")

scan.import_points_from_file(file_path=r"src/L1.las")
# scan.import_points_from_file(file_path=r"src/L1_raw.txt")

scan.compute_normals(k=8)
s_dbscan_c = ScanNormalsDirectionClassifier(scan)
s_dbscan_c.classify_normals(n_classes=3, unify_hemisphere=True)

# scan.plot(plotter=ScanPlotterWithLabelsMPL)

scans = ScanSplitterByLabels(scan).split()

for scan in scans.values():
    scan.filter_scan(filter_cls=ScanFilterByDBSCAN,
                     eps=0.01,
                     min_samples=5,
                     min_cluster_size=100,
                     )
    print(scan)
    scan.plot(plotter=ScanPlotterWithLabelsMPL)
    scans = ScanSplitterByLabels(scan).split()
    answer = float(input("Number_of_claster? (y/n): "))
    for scan in scans.values():
        point = scan._points[0]
        if point.labels == answer:
            scan.export_points_from_file(file_path=f"src/{scan.name}.txt")
            scan.plot(plotter=ScanPlotterWithLabelsMPL)
            break