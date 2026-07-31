from app.base.scan.parsers.ScanParserABC import ScanParserABC
from app.base.scan.parsers.ScanParserFormTxt import ScanParserFormTxt
from app.base.scan.parsers.ScanParserFromLas import ScanParserFromLas


class ScanParserFactory(ScanParserABC):

    scan_parsers = {"txt": ScanParserFormTxt,
                    "las": ScanParserFromLas,
                    }

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.parser = self.__get_scan_parser()(self.file_path)

    def parse(self, scan):
        self.parser.parse(scan=scan)

    def __get_scan_parser(self):
        file_extension = self.file_path.strip().split(".")[-1]
        return self.scan_parsers[file_extension]
