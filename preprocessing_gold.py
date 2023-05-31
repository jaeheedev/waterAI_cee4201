import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import matplotlib.dates as mdates
from sklearn import linear_model

class Plot_timeseries:
    def __init__(self, dir, out_dir):
        self.dir = dir
        self.out_dir = out_dir
        self.file = pd.read_csv(dir, names=['date', 'stage', 'flux', 'preci', 'temp', 'humid', 'heat'], header=None)
        self.file.drop([0], axis=0, inplace=True)
        self.file.reset_index()
        self.file['date'] = pd.to_datetime(self.file['date'])
        self.file = self.file.set_index('date')

    def plot(self):
        data = self.file
        print(pd.isnull(data).sum())

        y1 = data['stage'].astype('float')
        y2 = data['flux'].astype('float')
        y4 = data['preci'].astype('float')
        y5 = data['temp'].astype('float')
        y6 = data['humid'].astype('float')
        y7 = data['heat'].astype('float')
        
        plt.rc('figure', figsize = (50, 20))
        plt.subplot(3, 1, 1)
        plt.plot(y1)
        # plt.locator_params(axis='x', nbins = xlabel/10)
        plt.title('stage')

        plt.subplot(3, 1, 2)
        plt.plot(y2)
        plt.title('flux')

        plt.subplot(3, 1, 3)
        plt.plot(y5)
        plt.title('temperature')
        plt.show()

    def linear_regression(self):
        data = self.file.copy()

        lin_reg = linear_model.LinearRegression()
        # X and y after excluding missing values
        X = data.dropna(axis=0)[['preci', 'temp', 'humid', 'heat']] 
        y_stage = data.dropna(axis=0)['stage']
        y_flux = data.dropna(axis=0)['flux']

        lin_reg_model_stage = lin_reg.fit(X, y_stage)
        lin_reg_model_flux = lin_reg.fit(X, y_flux)

        y_pred_stage = lin_reg_model_stage.predict(data.loc[:, ['preci', 'temp', 'humid', 'heat']])
        y_pred_flux = lin_reg_model_flux.predict(data.loc[:, ['preci', 'temp', 'humid', 'heat']])
        
        data['stage'] = np.where(data['stage'].isnull(), 
                              pd.Series(y_pred_stage.flatten()), 
                              data['stage'])
        data['flux'] = np.where(data['flux'].isnull(), 
                              pd.Series(y_pred_flux.flatten()), 
                              data['flux'])
        self.file = data

    def seasonality(self, index, show = True):
        data = self.file.copy()

        series = data[index].astype('float')
        result = seasonal_decompose(series, model='addictive', period=365)
        if show:
            result.plot()
            plt.show()

        stl = STL(series, period = 365)
        seasonality = stl.fit()
        if show:
            seasonality.plot()
            plt.show()
        return seasonality

    def MAE(self, true, pred):
        return np.mean(np.abs(true-pred))
    
    def anomaly_detection(self, index, show = True):
        seasonality = self.seasonality(index = index, show = show)

        x = seasonality.resid.index[:]
        y = seasonality.resid.values[:]
        y_p = seasonality.observed.values[:]

        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        IQR = IQR if IQR > 0 else -1*IQR
        lower = Q1 - 1.5 * IQR
        higher = Q3 + 1.5 * IQR
        
        # resid 값에 대한 이상치 탐측 후 나타내기
        if show:
            plt.rc('figure', figsize = (50, 20))
            plt.rc('font', size=15)
            fig, ax = plt.subplots()
            ax.plot_date(x, y, color = 'black', linestyle = "--")
            ax.axhline(y=lower, color = 'blue')
            ax.axhline(y=higher, color = 'blue')
            for i in range(len(y)):
                if y[i] < lower or y[i] > higher:
                    ax.annotate('Anomaly', (mdates.date2num(x[i]), y[i]), xytext = (30, 20), textcoords = 'offset points', color = 'red', arrowprops = dict(facecolor='red', arrowstyle='fancy'))
            
            fig.autofmt_xdate()
            plt.show()

        # 해당 이상치 값이 전체 데이터에 어디에 해당하는가
        if show:
            ax = self.file[index].astype('float').plot()
            plt.rc('figure', figsize = (50, 20))
            plt.rc('font', size=15)
            for i in range(len(y)):
                if y[i] < lower or y[i] > higher:
                    ax.annotate('Anomaly', (mdates.date2num(x[i]), y_p[i]), xytext=(30, 20), textcoords='offset points', color='red',arrowprops=dict(facecolor='red',arrowstyle='fancy'))
            
            fig.autofmt_xdate()
            plt.show()

        # 이상치 값을 가장 가까운 선의 값으로 수정
        for i in range(len(y)):
            if y[i] < lower: y[i] = lower
            elif y[i] > higher: y[i] = higher

        # 수정된 결과 확인치 처리 코드 추가
        new_y = seasonality.resid + seasonality.seasonal + seasonality.trend
        new_data = self.file[index]
        new_data[index] = new_y

        print('MAE SCORE : ', self.MAE(new_y, y_p))

        if show :        
            ax = new_data[index].astype('float').plot()
            plt.rc('figure', figsize = (50, 20))
            plt.rc('font', size=15)
            plt.plot(new_y,
            color='pink',
            markersize=6,
            label = 'revised')

            plt.plot(y_p,
            color='blue',
            markersize=6,
            label = 'previous')
        
            plt.legend()
            plt.show()

        return new_data
   
    
    def output_file_train(self):
        self.linear_regression()
        nd_stage = self.file['stage']
        nd_flux = self.file['flux']
        nd_preci = self.anomaly_detection(index = 'preci', show = False)
        nd_temp = self.anomaly_detection(index = 'temp', show = False)
        nd_humid = self.anomaly_detection(index = 'humid', show = False)
        nd_heat = self.anomaly_detection(index = 'heat', show = False)

        new_data = pd.concat([nd_stage, nd_flux, nd_preci, nd_temp, nd_humid, nd_heat], axis=1)
        new_data = new_data[:-4]
        # csv file export
        new_data.to_csv(self.out_dir)
        print("change completed!")

    def output_file_test(self):
        self.linear_regression()
        self.file.to_csv(self.out_dir)
        print("change completed!")
        

def main():
    # print("train data preprocessing")
    # print("gwang")
    # gwang_plot = Plot_timeseries('./rawdata/train/gwang17-21.csv', './preprocessing/train/revised_gwang17-21.csv')
    # gwang_plot.output_file_train()

    # print("choeng")
    # cheong_plot = Plot_timeseries('./rawdata/train/cheong17-21.csv', './preprocessing/train/revised_cheong17-21.csv')
    # cheong_plot.output_file_train()

    # print('haeng')
    # haeng_plot = Plot_timeseries('./rawdata/train/haeng17-21.csv', './preprocessing/train/revised_haeng17-21.csv')
    # haeng_plot.output_file_train()

    # print('pal')
    # pal_plot = Plot_timeseries('./rawdata/train/pal17-21.csv', './preprocessing/train/revised_pal17-21.csv')
    # pal_plot.output_file_train()

    print("test data preprocessing")
    print("gwang")
    gwang_plot = Plot_timeseries('./goldriver/금강교.csv', './preprocessing_gr/test/revised_gr.csv')
    gwang_plot.output_file_test()

    print("choeng")
    cheong_plot = Plot_timeseries('./goldriver/명학리.csv', './preprocessing_gr/test/revised_mh.csv')
    cheong_plot.output_file_test()

    print('haeng')
    haeng_plot = Plot_timeseries('./goldriver/백제교.csv', './preprocessing_gr/test/revised_beak.csv')
    haeng_plot.output_file_test()

    print('pal')
    pal_plot = Plot_timeseries('./goldriver/신흥리.csv', './preprocessing_gr/test/revised_sh.csv')
    pal_plot.output_file_test()

if __name__ == '__main__':
    main()