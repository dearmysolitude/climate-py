import traceback

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pandas as pd
from matplotlib.figure import Figure

from analyzerInterface import AnalyzerInterface

class CO2LevelAnalyzer(AnalyzerInterface):
    def __init__(self):
        super().__init__()

    def setup_data(self):
        ## 데이터 불러오기 및 전처리, df 형태로 객체에 저장
        self.df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': [10, 15, 20, 12, 18, 25, 22, 16, 28, 30]
        })

    def analyze(self):
        ## df으로 그림을 그려 데이터를 반환
        pass

    def render_result(self):
        try:
            # Matplotlib 그래프 생성
            figure = Figure(figsize=(6, 4), dpi=100)
            canvas = FigureCanvasQTAgg(figure)
            ax = figure.add_subplot(111)
            ax.plot(self.df['date'], self.df['value'])
            ax.set_title('CO2 Level')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')

            self.rendered_result = canvas
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def get_rendered_result(self):
        return self.rendered_result
