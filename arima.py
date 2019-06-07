'''
时间序列模型：ARIMA
'''
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def test_stationarity(timeseries):
    # ADF平稳性检验
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # P值：当原假设为真时，比所得到的样本观察结果更极端的结果出现的概率
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)
    # 滑动均值和方差
    rolmean = timeseries.rolling(window=12).mean()  # 每window个数据取均值和方差
    rolstd = timeseries.rolling(window=12).std()
    # 绘制滑动统计量
    plt.figure()
    plt.plot(timeseries, color='blue', label='原始数据')
    plt.plot(rolmean, color='red', label='滑动均值')
    plt.plot(rolstd, color='black', label='滑动方差')
    plt.legend(loc='best')
    plt.show()
def decompose(timeseries):
    # 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # plt.subplot(411)
    # plt.plot(ts_log, label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(trend, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(seasonal, label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
    return trend, seasonal, residual
if __name__ == '__main__':
    '''setp1：获取时间序列样本集'''
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    df = pd.read_csv('./data/arima_test1.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
    split_idx = 10  # 交叉验证分割比率
    ts_train = df['#Passengers'][0:-split_idx]  # 训练集
    ts_test = df['#Passengers'][-split_idx:]  # 测试集
    '''setp2：取对数和一阶差分，通过滑动均值和方差、以及ADF单根检验差分序列是否满足稳定性'''
    ts_train_log = np.log(ts_train)
    ts_train_log_diff = ts_train_log.diff(1)
    ts_train_log_diff.dropna(inplace=True)
    # trend, seasonal, residual = decompose(ts_train_log)
    # residual.dropna(inplace=True)
    # test_stationarity(ts_train_log_diff)
    '''setp3：模型定阶，画出ACF和PACF的图像'''
    lag_acf = acf(ts_train_log_diff, nlags=20)
    lag_pacf = pacf(ts_train_log_diff, nlags=20, method='ols')
    # plt.subplot(121)
    # plt.plot(lag_acf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts_train_log_diff)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts_train_log_diff)), linestyle='--', color='gray')
    # plt.title('Autocorrelation Function')
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96/np.sqrt(len(ts_train_log_diff)), linestyle='--', color='gray')
    # plt.axhline(y=1.96/np.sqrt(len(ts_train_log_diff)), linestyle='--', color='gray')
    # plt.title('Partial Autocorrelation Function')
    # plt.tight_layout()
    # plt.show()
    # AIC和BIC准则：暴力定阶法
    # order = st.arma_order_select_ic(ts_train_log_diff, max_ar=10, max_ma=10, ic=['aic', 'bic', 'hqic'])
    # print(order.bic_min_order)  # (10, 7)

    '''setp4：训练ARIMA模型'''
    model = ARIMA(ts_train_log, order=(10, 1, 7)).fit(disp=-1)

    '''setp5：检验模型学习效果'''
    # 模型检验：残差的核密度(概率密度)为正态分布
    # pd.DataFrame(model.resid).plot(kind='kde')
    # plt.show()
    # 模型检验：残差序列是否满足白噪声qq图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # fig = qqplot(model.resid, line='q', ax=ax, fit=True)  # 检验拟合的残差序列分布的相似性
    # plt.show()
    # 模型检验：计算DW（检验一阶自相关性）
    # DW = sm.stats.durbin_watson(model.resid.values)
    # print('一阶自相关DW={}'.format(np.round(DW, 2)))
    # 模型检验：观察拟合后d次差分序列和原始差分序列
    # plt.plot(ts_train_log_diff, label='原始差分序列', color='#7B68EE')
    # plt.plot(model.fittedvalues, label='拟合差分序列', color='#FF4040')
    # plt.title('拟合RMSE：{}'.format(np.round(np.sum((model.fittedvalues - ts_train_log_diff) ** 2), 2)))
    # plt.legend(loc='best')
    # plt.show()

    '''setp6：模型测试效果'''
    # 残差序列逆向还原拟合时间序列：log_diff -> log -> Xt
    fit_ARIMA_log_diff = pd.Series(model.fittedvalues, index=ts_train_log.index, copy=True)
    fit_ARIMA_log_diff_cumsum = fit_ARIMA_log_diff.cumsum()
    fit_ARIMA_log = pd.Series(ts_train_log.iloc[0], index=ts_train_log.index)
    fit_ARIMA_log = fit_ARIMA_log.add(fit_ARIMA_log_diff_cumsum, fill_value=0)
    fit_ARIMA_log.dropna(inplace=True)
    fit_ARIMA = np.exp(fit_ARIMA_log)

    # 残差序列交叉验证测试集
    predict_date = pd.date_range(start=fit_ARIMA_log.index[-1], periods=len(ts_test)+1, freq='MS')
    forecast = model.forecast(len(ts_test))[0].tolist()  # 向后预测(测试集)
    predict_ARIMA_log = pd.Series([fit_ARIMA_log[-1]] + forecast, index=predict_date, copy=True)
    predict_ARIMA_log.dropna(inplace=True)
    predict_ARIMA = np.exp(predict_ARIMA_log)

    plt.plot(df['#Passengers'], label='原始序列', color='#7B68EE')
    plt.plot(fit_ARIMA, label='拟合序列', color='#FF4040')
    plt.plot(predict_ARIMA, label='预测序列', color='#3CB371')
    fit_RMSE = np.round(np.sqrt(np.sum((fit_ARIMA - ts_train) ** 2) / len(ts_train)), 2)
    predict_RMSE = np.round(np.sqrt(np.sum((predict_ARIMA - ts_test) ** 2) / len(ts_test)), 2)
    plt.title('拟合RMSE：{}，预测RMSE：{}'.format(fit_RMSE, predict_RMSE))
    plt.legend(loc='best')
    plt.show()
