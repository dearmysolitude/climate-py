import traceback
import os, re
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from analyzerInterface import AnalyzerInterface
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, GRU
from matplotlib.dates import YearLocator, DateFormatter

class TempLevelAnalyzer(AnalyzerInterface):

    def __init__(self):
        super().__init__()


    def setup_data(self):
        try:

            # LSTM
            # 데이터 불러오기 및 전처리, df 형태로 객체에 저장
            # path = './data/'
            # file_list = os.listdir(path)
            #
            # file_list_py = [file for file in file_list if re.match(r'서울평균(0[1-9]|1[0-2])\.csv', file)]
            # # print(file_list_py)
            # self.df = pd.DataFrame()
            # df = self.df
            #
            # for i in file_list_py:
            #     data = pd.read_csv(path + i, encoding='cp949')
            #     df = pd.concat([df, data])
            # print(self.df)

            df = pd.read_csv('./data/서울전체연평균.csv', encoding='cp949')
            df.drop(columns=['지점', '지점명'], inplace=True)
            # nul값 드랍
            df.dropna(inplace=True)

            # 평균기온(°C)', '최저기온(°C)', '최고기온(°C)'를 x값으로
            df['date'] = pd.to_datetime(df['일시'])  # 날짜 형식 변환
            data = df[['평균기온(°C)', '평균최저기온(°C)', '평균최고기온(°C)']].values


            # 값 스케일링 (0과 1 사이)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            def create_dataset(dataset, time_step=1):
                data_x, data_y = [], []
                for i in range(len(dataset) - time_step):
                    data_x.append(dataset[i:(i + time_step)])
                    data_y.append(dataset[i + time_step, 0])
                return np.array(data_x), np.array(data_y)

            time_step = 3
            x, y = create_dataset(scaled_data, time_step)
            train_size = int(len(x) * 0.7)
            test_size = len(x) - train_size
            x_train, x_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

            # # LSTM 모델 생성
            # model = Sequential()
            # model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
            # model.add(LSTM(50))
            # model.add(Dense(1))
            # model.compile(loss='mean_squared_error', optimizer='adam')
            #
            # # 모델 훈련
            # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1, verbose=1)
            #
            # # 모델 저장
            # model.save('./temp_ai/lstm_year_temp_model.h5')
            #
            # # 모델 불러오기
            # model = load_model('./temp_ai/lstm_year_temp_model.h5')
            #
            # # 모델 예측
            # train_predict = model.predict(x_train)
            # test_predict = model.predict(x_test)

            # 원래 스케일로 변환
            # def inverse_transform(pred, scaler, n_features):
            #     extended = np.zeros((len(pred), n_features))
            #     extended[:, 0] = pred.flatten()
            #     return scaler.inverse_transform(extended)[:, 0]
            #
            # train_predict = inverse_transform(train_predict, scaler, scaled_data.shape[1])
            # test_predict = inverse_transform(test_predict, scaler, scaled_data.shape[1])
            # y_train = inverse_transform(y_train.reshape(-1, 1), scaler, scaled_data.shape[1])
            # y_test = inverse_transform(y_test.reshape(-1, 1), scaler, scaled_data.shape[1])

            # 성능 평가(MAPE)
            # def mean_absolute_percentage_error(y_true, y_pred):
            #     y_true, y_pred = np.array(y_true), np.array(y_pred)
            #     mask = y_true != 0  # 실제 값이 0이 아닌 경우에만 계산
            #     return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            #
            # train_score = mean_absolute_percentage_error(y_train, train_predict)
            # test_score = mean_absolute_percentage_error(y_test, test_predict)
            # print(f'Train Score: {train_score:.2f} MAPE')
            # print(f'Test Score: {test_score:.2f} MAPE')
            #
            # # 원래 데이터와 예측 데이터 시각화
            # plt.figure(figsize=(12, 6))
            # train_plot_dates = df['date'][:len(train_predict) + time_step]
            # train_data_with_pred = np.concatenate((y_train[:time_step], train_predict))
            # plt.plot(train_plot_dates[:len(train_data_with_pred)], train_data_with_pred, label='Train Data')
            #
            # test_plot_dates = df['date'][
            #                   len(train_predict) + time_step:len(train_predict) + time_step + len(test_predict)]
            # plt.plot(test_plot_dates, y_test, label='Test Data')
            # plt.plot(test_plot_dates, test_predict, label='Test Predict', linestyle='--')
            #
            # plt.xlabel('Date')
            # plt.ylabel('Temperature (°C)')
            # plt.legend()
            # plt.show()

            # #GRU
            # # 데이터 불러오기 및 전처리, df 형태로 객체에 저장
            # path = './data/'
            # file_list = os.listdir(path)
            #
            # file_list_py = [file for file in file_list if re.match(r'서울평균(0[1-9]|1[0-2])\.csv', file)]
            # # print(file_list_py)
            # df = pd.DataFrame()
            #
            # for i in file_list_py:
            #     data = pd.read_csv(path + i, encoding='cp949')
            #     df = pd.concat([df, data])
            # # print(self.df)
            # df.drop(columns=['지점', '지점명'], inplace=True)
            # # nul값 드랍
            # df.dropna(inplace=True)
            #
            # # 평균기온(°C)', '최저기온(°C)', '최고기온(°C)'를 x값으로
            # df['date'] = pd.to_datetime(df['일시'])  # 날짜 형식 변환
            # data = df[['평균기온(°C)', '최저기온(°C)', '최고기온(°C)']].values
            #
            # # 값 스케일링 (0과 1 사이)
            # scaler = MinMaxScaler(feature_range=(0, 1))
            # scaled_data = scaler.fit_transform(data)
            #
            # def create_dataset(dataset, time_step=1):
            #     data_x, data_y = [], []
            #     for i in range(len(dataset) - time_step):
            #         data_x.append(dataset[i:(i + time_step)])
            #         data_y.append(dataset[i + time_step, 0])
            #     return np.array(data_x), np.array(data_y)
            #
            # time_step = 3
            # x, y = create_dataset(scaled_data, time_step)
            # train_size = int(len(x) * 0.7)
            # test_size = len(x) - train_size
            # x_train, x_test = x[:train_size], x[train_size:]
            # y_train, y_test = y[:train_size], y[train_size:]
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
            # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
            #
            # # GRU 모델 생성
            # model = Sequential()
            # model.add(GRU(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
            # model.add(GRU(50))
            # model.add(Dense(1))
            # model.compile(loss='mean_squared_error', optimizer='adam')
            #
            # # 모델 훈련
            # # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1, verbose=1)
            # #
            # # # 모델 저장
            # # model.save('./temp_ai/gru_temp_model.h5')
            #
            # # 모델 불러오기
            # model = load_model('./temp_ai/gru_temp_model.h5')
            #
            # # 모델 예측
            # train_predict = model.predict(x_train)
            # test_predict = model.predict(x_test)
            #
            # # 원래 스케일로 변환
            # def inverse_transform(pred, scaler, n_features):
            #     extended = np.zeros((len(pred), n_features))
            #     extended[:, 0] = pred.flatten()
            #     return scaler.inverse_transform(extended)[:, 0]
            #
            # train_predict = inverse_transform(train_predict, scaler, scaled_data.shape[1])
            # test_predict = inverse_transform(test_predict, scaler, scaled_data.shape[1])
            # y_train = inverse_transform(y_train.reshape(-1, 1), scaler, scaled_data.shape[1])
            # y_test = inverse_transform(y_test.reshape(-1, 1), scaler, scaled_data.shape[1])

            # 성능 평가(MAPE)
            # def mean_absolute_percentage_error(y_true, y_pred):
            #     y_true, y_pred = np.array(y_true), np.array(y_pred)
            #     mask = y_true != 0  # 실제 값이 0이 아닌 경우에만 계산
            #     return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            #
            # train_score = mean_absolute_percentage_error(y_train, train_predict)
            # test_score = mean_absolute_percentage_error(y_test, test_predict)
            # print(f'Train Score: {train_score:.2f} MAPE')
            # print(f'Test Score: {test_score:.2f} MAPE')

            # 원래 데이터와 예측 데이터 시각화
            # plt.figure(figsize=(12, 6))
            # train_plot_dates = df['date'][:len(train_predict) + time_step]
            # train_data_with_pred = np.concatenate((y_train[:time_step], train_predict))
            # plt.plot(train_plot_dates[:len(train_data_with_pred)], train_data_with_pred, label='Train Data')
            #
            # test_plot_dates = df['date'][
            #                   len(train_predict) + time_step:len(train_predict) + time_step + len(test_predict)]
            # plt.plot(test_plot_dates, y_test, label='Test Data')
            # plt.plot(test_plot_dates, test_predict, label='Test Predict', linestyle='--')
            #
            # plt.xlabel('Date')
            # plt.ylabel('Temperature (°C)')
            # plt.legend()
            # plt.show()
            self.df = df

        except Exception as e:
            print(e)
            print(traceback.format_exc())
    def analyze(self):
        ## df으로 그림을 그려 데이터를 반환
        try:
            df = self.df
            df['date'] = pd.to_datetime(df['일시'])  # 날짜 형식 변환
            data = df[['평균기온(°C)', '평균최저기온(°C)', '평균최고기온(°C)']].values

            # 값 스케일링 (0과 1 사이)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            def create_dataset(dataset, time_step=1):
                data_x, data_y = [], []
                for i in range(len(dataset) - time_step):
                    data_x.append(dataset[i:(i + time_step)])
                    data_y.append(dataset[i + time_step, 0])
                return np.array(data_x), np.array(data_y)

            time_step = 3
            x, y = create_dataset(scaled_data, time_step)
            train_size = int(len(x) * 0.7)
            test_size = len(x) - train_size
            x_train, x_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

            model = load_model('./temp_ai/lstm_year_temp_model.h5')

            # 모델 예측
            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)

            # 원래 스케일로 변환
            def inverse_transform(pred, scaler, n_features):
                extended = np.zeros((len(pred), n_features))
                extended[:, 0] = pred.flatten()
                return scaler.inverse_transform(extended)[:, 0]

            train_predict = inverse_transform(train_predict, scaler, scaled_data.shape[1])
            test_predict = inverse_transform(test_predict, scaler, scaled_data.shape[1])
            y_train = inverse_transform(y_train.reshape(-1, 1), scaler, scaled_data.shape[1])
            y_test = inverse_transform(y_test.reshape(-1, 1), scaler, scaled_data.shape[1])

            # 성능 평가(MAPE)
            def mean_absolute_percentage_error(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                mask = y_true != 0  # 실제 값이 0이 아닌 경우에만 계산
                return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

            train_score = mean_absolute_percentage_error(y_train, train_predict)
            test_score = mean_absolute_percentage_error(y_test, test_predict)

            # print(f'Train Score: {train_score:.2f} MAPE')
            # print(f'Test Score: {test_score:.2f} MAPE')
            return round(train_score, 2), round(test_score, 2)
        except Exception as e:
            print(e)
            print(traceback.format_exc())


    def render_result(self):
        try:
            # Matplotlib 그래프 생성
            # print(self.df)
            df = self.df
            df['date'] = pd.to_datetime(df['일시'])  # 날짜 형식 변환
            data = df[['평균기온(°C)', '평균최저기온(°C)', '평균최고기온(°C)']].values

            # 값 스케일링 (0과 1 사이)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            def create_dataset(dataset, time_step=1):
                data_x, data_y = [], []
                for i in range(len(dataset) - time_step):
                    data_x.append(dataset[i:(i + time_step)])
                    data_y.append(dataset[i + time_step, 0])
                return np.array(data_x), np.array(data_y)

            time_step = 3
            x, y = create_dataset(scaled_data, time_step)
            train_size = int(len(x) * 0.7)
            test_size = len(x) - train_size
            x_train, x_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])


            # 모델 불러오기
            model = load_model('./temp_ai/lstm_year_temp_model.h5')
            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)

            # 원래 스케일로 변환
            def inverse_transform(pred, scaler, n_features):
                extended = np.zeros((len(pred), n_features))
                extended[:, 0] = pred.flatten()
                return scaler.inverse_transform(extended)[:, 0]

            train_predict = inverse_transform(train_predict, scaler, scaled_data.shape[1])
            test_predict = inverse_transform(test_predict, scaler, scaled_data.shape[1])
            y_train = inverse_transform(y_train.reshape(-1, 1), scaler, scaled_data.shape[1])
            y_test = inverse_transform(y_test.reshape(-1, 1), scaler, scaled_data.shape[1])

            # Matplotlib 그래프 생성
            figure = Figure(figsize=(6, 4), dpi=100)
            canvas = FigureCanvasQTAgg(figure)
            ax = figure.add_subplot(111)

            # 훈련 데이터 시각화
            train_plot_dates = self.df['date'][time_step:time_step + len(train_predict)]
            ax.plot(train_plot_dates, train_predict, label='Train Data', color='blue', alpha=0.5)

            # 테스트 데이터 시각화
            test_plot_dates = self.df['date'][train_size + time_step:train_size + time_step + len(test_predict)]
            ax.plot(test_plot_dates, y_test, label='Test Data', color='green', alpha=0.5)

            # 테스트 예측 시각화
            ax.plot(test_plot_dates, test_predict, label='Test Predict', linestyle='--', color='red')

            ax.set_title('Temp Level')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')

            # # x축의 날짜 눈금을 5년 단위로 설정
            # ax.xaxis.set_major_locator(YearLocator(5))
            # ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            #
            # # x축 눈금 및 레이블 회전 설정
            # figure.autofmt_xdate()

            # 범례 추가
            ax.legend()

            self.rendered_result = canvas
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def get_rendered_result(self):
        return self.rendered_result
