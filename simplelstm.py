import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import argparse

class Data:
    def __init__(self, train_path, test_path, seq_length):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.seq_length = seq_length

    def data_scaling(self):
        train_df = self.train_df
        test_df = self.test_df

        train_n, _ = train_df.shape
        test_n, _ = test_df.shape

        df_all = pd.concat([train_df, test_df], ignore_index=True)

        scaler_x = MinMaxScaler()
        scaler_x.fit(df_all.iloc[:, 2:7])
        df_all.iloc[:, 2:7] = scaler_x.transform(df_all.iloc[:, 2:7])

        scaler_y = MinMaxScaler()
        scaler_y.fit(df_all.iloc[:, [1]])
        df_all.iloc[:, [1]] = scaler_y.transform(df_all.iloc[:, [1]])

        imputer = SimpleImputer()
        train_imputer = imputer.fit(df_all.iloc[:train_n, 1:7])
        test_imputer = imputer.fit(df_all.iloc[:-test_n, 1:7])

        train_df = train_imputer.transform(df_all.iloc[:train_n, 1:7])
        test_df = test_imputer.transform(df_all.iloc[:-test_n, 1:7])

        return train_df, test_df, scaler_x, scaler_y

    def build_dataset(self, data, seq_length):
        X = []
        Y = []
        for i in range(0, len(data) - seq_length):
            _x = data[i:i+seq_length, :]
            _y = data[i+seq_length, [0]]
            X.append(_x)
            Y.append(_y)
        return np.array(X), np.array(Y)

    def get_dataset(self):
        scaled_train_df, scaled_test_df, scaler_x, scaler_y = self.data_scaling()
        train_X, train_y = self.build_dataset(scaled_train_df, self.seq_length)
        test_X, test_y = self.build_dataset(scaled_test_df, self.seq_length)
        return train_X, train_y, test_X, test_y, scaler_x, scaler_y

def parser():
    parser = argparse.ArgumentParser(description='5년 data input -> 1년 데이터 test')
    parser.add_argument('input', help='데이터 인풋의 경로 지정')
    parser.add_argument('--sl', default=7, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--ep', default=100, type=int)
    args = parser.parse_args()
    return args

def nse_metric(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    mse = np.mean((y_true - y_pred) ** 2)
    nse = 1 - mse / np.var(y_true)
    return nse

def rmse_metric(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def buffer_accuracy(y_true, y_pred, buffer):
    assert y_true.shape == y_pred.shape
    within_buffer = np.abs(y_true - y_pred) <= buffer
    accuracy = np.mean(within_buffer)
    return accuracy

def calculate_buffer(y_true, alpha=1):
    std_dev = np.std(y_true)
    buffer = alpha * std_dev
    return buffer

def main():
    train_path = 'h_test/train/haeng17-21.csv'
    test_path = 'h_test/test/haeng22.csv'
    look_back = 7
    epochs = 10
    learning_rate = 0.001

    Haeng_dataset = Data(train_path, test_path, look_back)
    train_X, train_y, test_X, test_y, scaler_x, scaler_y = Haeng_dataset.get_dataset()

    input_size = train_X.shape[2]
    hidden_size = 10

    # LSTM 모델 생성
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(look_back, input_size)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_X, train_y, epochs=epochs, batch_size=32, verbose=1)

    # 모델 테스트
    testPredict = model.predict(test_X)

    # 실제 값으로 되돌리기 (역정규화)
    testPredict = scaler_y.inverse_transform(testPredict)
    test_y_inv = scaler_y.inverse_transform(test_y)

    mse = mean_squared_error(test_y_inv, testPredict)
    nse = nse_metric(test_y_inv, testPredict)
    rmse = rmse_metric(test_y_inv, testPredict)

    buffer = calculate_buffer(test_y_inv, alpha=1)
    accuracy = buffer_accuracy(test_y_inv, testPredict, buffer)

    print('Test MSE: %.3f' % mse)
    print(f"NSE: {nse}")
    print(f"RMSE: {rmse}")
    print(f"Buffer Accuracy: {accuracy * 100}% - Buffer: {buffer}")

    # 예측 결과와 실제 값 비교를 위한 그래프 생성
    plt.plot(test_y_inv, label='actual')
    plt.plot(testPredict, label='prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()