import os, sys
import traceback
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import traceback

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from co2AnalyzerImpl import CO2LevelAnalyzer
from tempAnalyzerImpl import TempLevelAnalyzer
from sealevelAnalyzerImpl import SeaLevelAnalyzer
from regAnalyzer import RegAnalyzer

matplotlib.use('Qt5Agg')
root = os.path.dirname(os.path.abspath(__file__))
MainUI = uic.loadUiType(os.path.join(root, 'ui/main.ui'))[0]


class Main(QMainWindow, MainUI):
    def __init__(self):
        super().__init__()
        self.nav_toolbar = None
        self.analyzer = None
        self.qv_box = None
        try:
            self.init_ui()
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def init_ui(self):
        self.setupUi(self)
        self.qv_box = self.findChild(QVBoxLayout, 'verticalLayout')
        self.pushButton_1.clicked.connect(self.take_order)
        self.pushButton_2.clicked.connect(self.do_your_job)

    def take_order(self):
        if self.analyzer:
            del self.analyzer
        self.radio_check()

        self.description.setText('데이터 임포트 중.')
        self.analyzer.setup_data()
        self.description.setText('데이터 셋팅 완료.')

    def radio_check(self):
        if self.radioButton_1.isChecked():
            self.analyzer = TempLevelAnalyzer()  # 바뀔 예정
        elif self.radioButton_2.isChecked():
            self.analyzer = CO2LevelAnalyzer()
        elif self.radioButton_3.isChecked():
            self.analyzer = SeaLevelAnalyzer()  # 바뀔 예정
        elif self.radioButton_4.isChecked():
            self.analyzer = RegAnalyzer()  # 바뀔 예정

    def do_your_job(self):
        try:
            if self.analyzer is None:
                self.description.setText('데이터가 선택되지 않았습니다.')
                return

            self.description.setText('데이터 분석 중.')
            description = self.analyzer.analyze()
            self.analyzer.render_result()
            rendered_result = self.analyzer.get_rendered_result()

            # 이전에 그린 그래프 제거
            while self.qv_box.count():
                item = self.qv_box.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

            if not hasattr(self, 'nav_toolbar') or self.nav_toolbar is None:
                self.nav_toolbar = NavigationToolbar(rendered_result, self)
            else:
                self.nav_toolbar.canvas = rendered_result

            self.addToolBar(self.nav_toolbar)
            self.qv_box.addWidget(rendered_result)
            self.description.setText(f'데이터 분석 완료.\n{description}')  # 여기에 분석 결과에 대한 설명을 세팅할 수 있다.
        except Exception as e:
            print(e)
            print(traceback.format_exc())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())

