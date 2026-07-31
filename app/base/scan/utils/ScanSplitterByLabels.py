from app.base.scan.Scan import Scan


class ScanSplitterByLabels:
    """
    Разделяет скан на подсканы по меткам.
    По умолчанию использует атрибут точки `labels` (как у DBSCAN),
    но можно указать любой другой атрибут, например `normal_label`.
    """

    def __init__(self, scan, label_attr="labels", include_noise=False, noise_label=-1):
        """
        scan          – исходный Scan
        label_attr    – имя атрибута метки в точке (str)
        include_noise – включать ли «шум» (обычно label = -1) в отдельный подскан
        noise_label   – значение метки, считающееся шумом
        """
        self.scan = scan
        self.label_attr = label_attr
        self.include_noise = include_noise
        self.noise_label = noise_label

    def split(self):
        """
        Возвращает словарь: {label_value: Scan}, где каждый Scan содержит
        только точки с соответствующей меткой.
        """
        # собираем уникальные метки
        labels_set = set()
        for p in self.scan:
            if not hasattr(p, self.label_attr):
                continue
            lbl = getattr(p, self.label_attr)
            if (lbl == self.noise_label) and (not self.include_noise):
                continue
            labels_set.add(lbl)

        label_to_scan = {}

        # создаём пустые подсканы
        for lbl in labels_set:
            sub_scan = Scan(scan_name=f"{self.scan.name}_label_{lbl}")
            label_to_scan[lbl] = sub_scan

        # раскладываем точки по подсканам
        for p in self.scan:
            if not hasattr(p, self.label_attr):
                continue
            lbl = getattr(p, self.label_attr)
            if (lbl == self.noise_label) and (not self.include_noise):
                continue
            sub_scan = label_to_scan.get(lbl)
            if sub_scan is not None:
                sub_scan.add_point(p)

        # обновляем границы подсканов
        for sub_scan in label_to_scan.values():
            sub_scan.borders = sub_scan._get_borders_dict(sub_scan._points)

        return label_to_scan


if __name__ == "__main__":
    from app.base.scan.Scan import Scan
    from app.base.scan.plotters.ScanPlotterWithLabelsMPL import ScanPlotterWithLabelsMPL
    from app.base.scan.utils.ScanNormalsDirectionClassifier import ScanNormalsDirectionClassifier
    from app.base.scan.filters.ScanFilterByDBSCAN import ScanFilterByDBSCAN

    scan = Scan("l1")

    scan.import_points_from_file(file_path=r"../../../../src/L1.las")
    scan.compute_normals(k=8)
    s_dbscan_c = ScanNormalsDirectionClassifier(scan)
    s_dbscan_c.classify_normals(n_classes=3, unify_hemisphere=True)

    # scan.plot(plotter=ScanPlotterWithLabelsMPL)

    scans = ScanSplitterByLabels(scan).split()

    for scan in scans.values():
        # scan.filter_scan(filter_cls=ScanFilterByDBSCAN,
        #                  eps=0.01,
        #                  min_samples=5,
        #                  min_cluster_size=100,
        #                  )
        print(scan)
        scan.plot(plotter=ScanPlotterWithLabelsMPL)