import traceback

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

from analyzerInterface import AnalyzerInterface


class CO2LevelAnalyzer(AnalyzerInterface):
    def __init__(self):
        super().__init__()
        self.predicted_df = None
        self.df_train_vis = None
        self.df_test_vis = None

    def setup_data(self):
        try:
            # 데이터 불러오기 및 전처리, df 형태로 객체에 저장
            filepath = './data/co2_mm_mlo.csv'
            self.df = pd.read_csv(filepath, delimiter=',', header=0, encoding='utf-8')
            self.df = self.df.dropna()
            self.df['date'] = pd.to_datetime(dict(year=self.df.year, month=self.df.month, day=1))

            self.df.set_index('date', inplace=True)

            self.df = self.df[['deseasonalized']]
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def analyze(self):
        proportion_train_test = 0.8
        time_step = 10
        # 테스트 데이터 분할 및 데이터 스케일링
        self.df.reset_index(drop=True)
        dataset = self.df['deseasonalized'].values.reshape(-1, 1)

        # 데이터 분할하기
        df_train = dataset[:int(proportion_train_test * len(dataset)), :]
        df_test = dataset[int(proportion_train_test * len(dataset)):, :]

        # 데이터 스케일링하기
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # 학습 데이터 가공
        x_train_data, y_train_data = [], []

        # 10 을 기준으로 데이터 생성하기
        for i in range(time_step, len(df_train)):
            x_train_data.append(scaled_data[i - time_step:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0],
                                                 x_train_data.shape[1], 1))

        # LSTM 모형 설계 및 파라미터 정의
        lstm = Sequential()

        lstm.add(LSTM(units=time_step, return_sequences=True,
                      input_shape=(x_train_data.shape[1], 1)))
        lstm.add(LSTM(units=time_step))
        lstm.add(Dense(1))

        # 모형의 학습 방법 설정하여 학습 진행하기
        lstm.compile(loss='mean_squared_error', optimizer='adam')

        # 모델 훈련
        # lstm.fit(x_train_data, y_train_data, epochs=100, batch_size=1, verbose=2)

        # 모델 저장
        # lstm.save('./ai_model/lstm_co2_model.h5')

        # 모델 불러오기
        lstm = load_model('./ai_model/lstm_co2_model.h5')

        # 테스트 데이터 준비
        inputs_data = self.df[len(self.df) - len(df_test) - time_step:]
        inputs_data = scaler.transform(inputs_data)

        # 학습이 완료된 LSTM 모형으로 예측
        x_test = []
        for i in range(10, inputs_data.shape[0]):
            x_test.append(inputs_data[i - time_step:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_value = lstm.predict(x_test)
        predicted_value = scaler.inverse_transform(predicted_value)

        data_size = int(0.8 * len(dataset))

        # 데이터 분할
        self.df_train_vis = self.df[:data_size]
        self.df_test_vis = self.df[data_size:]

        # 예측값을 DataFrame으로 변환
        self.predicted_df = pd.DataFrame(predicted_value, columns=['Predicted'])

        # 원본 데이터프레임의 인덱스를 예측값 DataFrame에 할당
        self.predicted_df.index = self.df.index[data_size:]

        return 'MAPE: ' + str(self.mape(df_test, self.predicted_df['Predicted'].values))

    def mape(self, y, predicted_df):
        y = y.reshape(-1)
        return np.mean(np.abs((y - predicted_df) / y) * 100)

    def render_result(self):
        try:
            # Matplotlib 그래프 생성
            figure = Figure(figsize=(6, 4), dpi=100)
            canvas = FigureCanvasQTAgg(figure)
            ax = figure.add_subplot(111)

            ax.plot(self.df_train_vis, label='Train Data')  # 훈련 데이터 그리기
            ax.plot(self.df_test_vis, label='Test Data')  # 테스트 데이터 그리기
            ax.plot(self.predicted_df, label='Predicted Data')
            ax.set_title('CO2 Level')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value[ppm]')
            ax.legend()

            self.rendered_result = canvas
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def get_rendered_result(self):
        return self.rendered_result
