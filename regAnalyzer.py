import traceback

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from analyzerInterface import AnalyzerInterface
from scipy.stats import pearsonr
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt


class RegAnalyzer(AnalyzerInterface):
    def __init__(self):
        super().__init__()

    def setup_data(self):
        try:

            # 데이터 불러오기
            df1 = pd.read_csv('./data/서울연평균.csv', encoding='cp949')
            df2 = pd.read_csv('./data/co2연평균.csv', encoding='cp949')
            # print(df1)
            # print(df2)
            df1.rename(columns={'일시': 'year', '평균기온(°C)': 'temperature'}, inplace=True)
            df2.rename(columns={'ann inc': 'ann'}, inplace=True)
            # print(df1)

            # 데이터 병합
            merged_df = pd.merge(df1, df2, on=['year'])

            # 필요한 컬럼 선택
            merged_df = merged_df[['temperature', 'ann', 'mean', 'mean_gl']]

            # 결측치 제거
            merged_df.dropna(inplace=True)

            # 회귀분석 수행
            fit = ols('temperature ~ ann + mean + mean_gl', data=merged_df).fit()
            print(fit.summary())

            # 각 독립 변수와 종속 변수 사이의 산점도와 회귀선 시각화
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # ann
            sns.regplot(x='ann', y='temperature', data=merged_df, ax=axs[0, 0])
            axs[0, 0].set_title('Scatter plot with regression line (ann)')

            # mean
            sns.regplot(x='mean', y='temperature', data=merged_df, ax=axs[0, 1])
            axs[0, 1].set_title('Scatter plot with regression line (mean)')

            # mean_gl
            sns.regplot(x='mean_gl', y='temperature', data=merged_df, ax=axs[1, 0])
            axs[1, 0].set_title('Scatter plot with regression line (mean_gl)')

            plt.tight_layout()
            plt.show()


            # # 회귀선을 포함한 산점도 그리기
            # sns.lmplot(x='ann', y='temperature', data=merged_df)
            # plt.title('Scatter plot with regression line')
            # plt.xlabel('CO2 Ann Increase')
            # plt.ylabel('Temperature')
            # plt.show()



            # print(merged_df)
            # print(merged_df.shape)
            # merged_df.drop(columns=['unc', '지점', '지점명', 'year', '평균최저기온(°C)', '평균최고기온(°C)'], inplace=True)
            # merged_df.dropna(inplace=True)
            # print(merged_df)
            # print(merged_df.shape)
            # # merged_df.rename(columns={'평균기온(°C)': 'temperature'}, inplace=True)
            # fit = ols('temperature ~ ann', data=merged_df).fit()
            # print(fit.summary())
            # # 회귀선을 포함한 산점도 그리기
            # sns.lmplot(x='temperature', y='ann', data=merged_df)
            # plt.title('Scatter plot with regression line')
            # plt.xlabel('Temperature')
            # plt.ylabel('Average')
            # plt.show()

            # 상관 계수 계산
            # correlation_coefficient, p_value = pearsonr(merged_df['평균기온(°C)'], merged_df['average'])
            # print("Correlation Coefficient:", correlation_coefficient)
            # print("p-value:", p_value)

        except Exception as e:
            print(e)
            print(traceback.format_exc())




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
