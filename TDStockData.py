#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 吴韵卫
# @Date:   2019-07-20 02:35:48
# @Last Modified by:   吴韵卫
# @Last Modified time: 2019-07-20 02:35:48

'''
底层数据获取
'''

import sys
reload(sys)
sys.setdefaultencoding('UTF8')
import os

import datetime
import time
import json
import math

import numpy as np
import pandas as pd

from stockstats import StockDataFrame
import talib

from pytdx.hq import TdxHq_API
from pytdx.config.hosts import hq_hosts


class TBStockData:

    __serverList = []
    _bestIP = []
    __bestIPFile = ''
    __tdx = None
    _lastBaseHistList = pd.DataFrame()
    _xdxrData = None


    def __init__(self, autoIP = False):
        self.__serverList = hq_hosts
        self.__bestIPFile = os.path.dirname(os.path.realpath(__file__)) + '/best.ip'

        if autoIP:
            self.getBestIP()
        else:
            if os.path.exists(self.__bestIPFile):
                with open(self.__bestIPFile, 'r') as f:
                    data = f.read()
                    self._bestIP = json.loads(data)

    def ping(self, ip, port):
        api = TdxHq_API()
        time1 = datetime.datetime.now()

        try:
            with api.connect(ip, int(port)):
                if len(api.get_security_list(0, 1)) > 800:
                    return datetime.datetime.now() - time1
                else:
                    return datetime.timedelta(9, 9, 0)
        except:
            return datetime.timedelta(9, 9, 0)

    def getBestIP(self):

        pingTimeList = [self.ping(x[1], x[2]) for x in self.__serverList]
        self._bestIP = self.__serverList[pingTimeList.index(min(pingTimeList))]

        with open(self.__bestIPFile, 'w') as f:
            f.write(json.dumps(self._bestIP))

    def showAllIP(self):
        for item in self.__serverList:
            print item[0],'\t', item[1], '\t', item[2]

    def _connect(self):

        if self.__tdx is None:
            if not self._bestIP:
                self.getBestIP()

            #self.__tdx = TdxHq_API(heartbeat=True, auto_retry=True)
            self.__tdx = TdxHq_API(auto_retry=True)
            self.__tdx.connect(self._bestIP[1], int(self._bestIP[2]))

    #计算量比
    def _setVolRaito(self, row):
        date = row.name
        histList = self._lastBaseHistList[:date]
        if len(histList) < 6:
            return np.nan

        return round((histList['vol'].values[-1] / 240) / (histList[-6:-1]['vol'].sum() / 1200), 3)

    #计算各种指标
    def getData(self, df = pd.DataFrame(), indexs=['turnover', 'vol', 'ma', 'macd', 'kdj', 'cci', 'bbi', 'sar', 'trix']):

        indexs = [x.lower() for x in indexs]
        histList = pd.DataFrame()

        if not df.empty:
            histList = df.copy()
        elif not self._lastBaseHistList.empty:
            histList = self._lastBaseHistList.copy()

        if histList.empty:
            return None

        dayKStatus = False
        try:
            if int(time.mktime(time.strptime(str(histList.index[-1]), "%Y-%m-%d %X"))) - int(time.mktime(time.strptime(str(histList.index[-2]), "%Y-%m-%d %X"))) > 43200:
                #日线以上行情
                dayKStatus = True
        except:
            dayKStatus = True

        #计算涨幅
        histList['p_change'] = histList['close'].pct_change().round(5) * 100

        #量比
        histList['vol_ratio'] = histList.apply(self._setVolRaito, axis=1)

        #振幅
        histList['amp'] = ((histList['high'] - histList['low']) / histList.shift()['close'] * 100).round(3)

        #计算换手率
        if self._xdxrData is None:
            xdxrData = self.getXdxr(str(histList['code'].values[0]))
        else:
            xdxrData = self._xdxrData
        info = xdxrData[xdxrData['liquidity_after'] > 0][['liquidity_after', 'shares_after']]

        if dayKStatus:
            startDate = str(histList.index[0])[0:10]
            endDate = str(histList.index[-1])[0:10]
            info1 = info[info.index <= startDate][-1:]
            info = info1.append(info[info.index >= startDate]).drop_duplicates()
            info = info.reindex(pd.date_range(info1.index[-1], endDate))
            info = info.resample('1D').last().fillna(method='pad')[startDate:endDate]
            #info['date'] = info.index
            #info['date'] = info['date'].dt.strftime('%Y-%m-%d')
            #info = info.set_index('date')

            circulate = info['liquidity_after'] * 10000
            capital = info['shares_after'] * 10000
        else:
            circulate = info['liquidity_after'].values[-1] * 10000
            capital = info['shares_after'].values[-1] * 10000

        #histList['circulate'] = (circulate / 10000 / 10000).round(4)

        if 'turnover' in indexs and dayKStatus:
            histList['turnover'] = (histList['vol'] * 100 / circulate).round(5) * 100
            histList['turnover5'] = talib.MA(histList['turnover'].values, timeperiod=5).round(3)

        #stockstats转换,主要是用来计算KDJ等相关指标
        #用talib计算KDJ时会与现有软件偏差大
        ss = StockDataFrame.retype(histList[['high','low','open','close']])

        #MACD计算
        if 'macd' in indexs:
            difList, deaList, macdList = talib.MACD(histList['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            macdList = macdList * 2
            histList['macd_dif'] = difList.round(3)
            histList['macd_dea'] = deaList.round(3)
            histList['macd_value'] = macdList.round(3)
            histList['macd_value_ma'] = 0
            try:
                histList['macd_value_ma'] = talib.MA(histList['macd_value'].values, timeperiod=5).round(3)
            except:
                pass
            histList['macd_cross_status'] = 0
            macdPosList = histList['macd_dif'] > histList['macd_dea']
            histList.loc[macdPosList[(macdPosList == True) & (macdPosList.shift() == False)].index, 'macd_cross_status'] = 1
            histList.loc[macdPosList[(macdPosList == False) & (macdPosList.shift() == True)].index, 'macd_cross_status'] = -1
            #histList[['macd_cross_status']] = histList[['macd_cross_status']].fillna(method='pad')

        #KDJ计算
        if 'kdj' in indexs:
            histList['kdj_k'] = ss['kdjk'].round(3)
            histList['kdj_d'] = ss['kdjd'].round(3)
            histList['kdj_j'] = ss['kdjj'].round(3)
            histList['kdj_cross_status'] = 0
            kdjPosList = histList['kdj_k'] >= histList['kdj_d']
            histList.loc[kdjPosList[(kdjPosList == True) & (kdjPosList.shift() == False)].index, 'kdj_cross_status'] = 1
            histList.loc[kdjPosList[(kdjPosList == False) & (kdjPosList.shift() == True)].index, 'kdj_cross_status'] = -1
            #histList[['kdj_cross_status']] = histList[['kdj_cross_status']].fillna(method='pad')

        #CCI计算
        if 'cci' in indexs:
            histList['cci'] = ss['cci'].round(3)

        #ma相关计算
        if 'ma' in indexs:
            histList['ma5'] = talib.MA(histList['close'].values, timeperiod=5).round(3)
            histList['ma10'] = talib.MA(histList['close'].values, timeperiod=10).round(3)
            histList['ma20'] = talib.MA(histList['close'].values, timeperiod=20).round(3)
            histList['ma30'] = talib.MA(histList['close'].values, timeperiod=30).round(3)
            histList['ma60'] = talib.MA(histList['close'].values, timeperiod=60).round(3)
            histList['ma240'] = talib.MA(histList['close'].values, timeperiod=240).round(3)
            histList[['ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma240']] = histList[['ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma240']].fillna(0)

        #成交量计算
        if 'vol' in indexs:
            histList['vol5'] = talib.MA(histList['vol'].values, timeperiod=5).round(3)
            histList['vol10'] = talib.MA(histList['vol'].values, timeperiod=10).round(3)
            histList['vol20'] = talib.MA(histList['vol'].values, timeperiod=20).round(3)
            histList['vol_zoom'] = (histList['vol'] / histList['vol5'] * 1.0).round(3)
            histList['vol5_vol10_cross_status'] = 0
            volumePosList = histList['vol5'] >= histList['vol10']
            histList.loc[volumePosList[(volumePosList == True) & (volumePosList.shift() == False)].index, 'vol5_vol10_cross_status'] = 1
            histList.loc[volumePosList[(volumePosList == False) & (volumePosList.shift() == True)].index, 'vol5_vol10_cross_status'] = -1
            del volumePosList
            histList['vol5_vol20_cross_status'] = 0
            volumePosList = histList['vol5'] >= histList['vol20']
            histList.loc[volumePosList[(volumePosList == True) & (volumePosList.shift() == False)].index, 'vol5_vol20_cross_status'] = 1
            histList.loc[volumePosList[(volumePosList == False) & (volumePosList.shift() == True)].index, 'vol5_vol20_cross_status'] = -1
            del volumePosList
            histList['vol10_vol20_cross_status'] = 0
            volumePosList = histList['vol10'] >= histList['vol20']
            histList.loc[volumePosList[(volumePosList == True) & (volumePosList.shift() == False)].index, 'vol10_vol20_cross_status'] = 1
            histList.loc[volumePosList[(volumePosList == False) & (volumePosList.shift() == True)].index, 'vol10_vol20_cross_status'] = -1
            #histList[['vol5_vol10_cross_status', 'vol5_vol20_cross_status', 'vol10_vol20_cross_status']] = histList[['vol5_vol10_cross_status', 'vol5_vol20_cross_status', 'vol10_vol20_cross_status']].fillna(method='pad')

        #bbi计算
        if 'bbi' in indexs:
            ma3 = talib.MA(histList['close'].values, timeperiod=3)
            ma6 = talib.MA(histList['close'].values, timeperiod=6)
            ma12 = talib.MA(histList['close'].values, timeperiod=12)
            ma24 = talib.MA(histList['close'].values, timeperiod=24)
            histList['bbi'] = (ma3 + ma6 + ma12 + ma24) / 4
            histList['bbi'] = histList['bbi'].round(3)

        #SAR计算
        if 'sar' in indexs:
            sarList = talib.SAR(histList['high'].values, histList['low'].values, acceleration=0.04, maximum=0.2)
            histList['sar'] = sarList.round(3)
            histList['sar_cross_status'] = 0
            sarPosList = histList['close'] >= histList['sar']
            histList.loc[sarPosList[(sarPosList == True) & (sarPosList.shift() == False)].index, 'sar_cross_status'] = 1
            histList.loc[sarPosList[(sarPosList == False) & (sarPosList.shift() == True)].index, 'sar_cross_status'] = -1

        #计算TRIX
        if 'trix' in indexs:
            histList['trix'] = np.nan
            histList['trma'] = np.nan
            histList['trix_diff'] = np.nan
            try:
                trix = talib.TRIX(histList['close'].values, 12)
                trma = talib.MA(trix, timeperiod=20)
                histList['trix'] = trix.round(3)
                histList['trma'] = trma.round(3)
                histList['trix_diff'] = histList['trix'] - histList['trma']
                histList['trix_cross_status'] = 0
                trixPosList = histList['trix'] >= histList['trma']
                histList.loc[trixPosList[(trixPosList == True) & (trixPosList.shift() == False)].index, 'trix_cross_status'] = 1
                histList.loc[trixPosList[(trixPosList == False) & (trixPosList.shift() == True)].index, 'trix_cross_status'] = -1
                #histList[['trix_cross_status']] = histList[['trix_cross_status']].fillna(method='pad')
            except:
                pass

        if 'cyc' in indexs:
            avePrice = histList['amount'] / (histList['vol'] * 100)
            histList['cyc5'] = talib.MA(avePrice.values, timeperiod=5).round(3)
            histList['cyc13'] = talib.MA(avePrice.values, timeperiod=13).round(3)
            histList['cyc34'] = talib.MA(avePrice.values, timeperiod=34).round(3)
            #histList['cycx'] = talib.EMA(histList['close'].values, timeperiod=histList['vol'].values * 100 / circulate).round(3)
            histList['cyc5_cyc13_cross_status'] = 0
            volumePosList = histList['cyc5'] >= histList['cyc13']
            histList.loc[volumePosList[(volumePosList == True) & (volumePosList.shift() == False)].index, 'cyc5_cyc13_cross_status'] = 1
            histList.loc[volumePosList[(volumePosList == False) & (volumePosList.shift() == True)].index, 'cyc5_cyc13_cross_status'] = -1
            del volumePosList
            histList['cyc13_cyc34_cross_status'] = 0
            volumePosList = histList['cyc13'] >= histList['cyc34']
            histList.loc[volumePosList[(volumePosList == True) & (volumePosList.shift() == False)].index, 'cyc13_cyc34_cross_status'] = 1
            histList.loc[volumePosList[(volumePosList == False) & (volumePosList.shift() == True)].index, 'cyc13_cyc34_cross_status'] = -1
            del volumePosList

        if 'boll' in indexs:
            up, mid, low = talib.BBANDS(
                histList['close'].values,
                timeperiod=20,
                # number of non-biased standard deviations from the mean
                nbdevup=2,
                nbdevdn=2,
                # Moving average type: simple moving average here
                matype=0)
            histList['boll_up'] = up.round(3)
            histList['boll_mid'] = mid.round(3)
            histList['boll_low'] = low.round(3)


        return histList

    #整理开始,结束时间,并计算相差天数
    def _getDate(self, start, end):
        if not end:
            end = time.strftime('%Y-%m-%d',time.localtime())

        if not start:
            t = int(time.mktime(time.strptime(str(end), '%Y-%m-%d'))) - 86400 * 800
            start = str(time.strftime('%Y-%m-%d',time.localtime(t)))

        startTimestamp = int(time.mktime(time.strptime(str(start), '%Y-%m-%d')))
        endTimestamp = int(time.mktime(time.strptime(str(end), '%Y-%m-%d')))
        diffDayNum = int((time.time() - startTimestamp) / 86400)
        if diffDayNum <= 0:
            diffDayNum = 1

        return start, end, diffDayNum

    #得到市场代码
    def getMarketCode(self, code):
        code = str(code)
        if code[0] in ['5', '6', '9'] or code[:3] in ["009", "126", "110", "201", "202", "203", "204"]:
            return 1
        return 0

    #时间整理
    def _dateStamp(self, date):
        datestr = str(date)[0:10]
        date = time.mktime(time.strptime(datestr, '%Y-%m-%d'))
        return date

    #整理时间
    def _timeStamp(self, _time):
        if len(str(_time)) == 10:
            # yyyy-mm-dd格式
            return time.mktime(time.strptime(_time, '%Y-%m-%d'))
        elif len(str(_time)) == 16:
                # yyyy-mm-dd hh:mm格式
            return time.mktime(time.strptime(_time, '%Y-%m-%d %H:%M'))
        else:
            timestr = str(_time)[0:19]
            return time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S'))


    #得到除权信息
    def getXdxr(self, code):

        self._connect()

        category = {
            '1': '除权除息', '2': '送配股上市', '3': '非流通股上市', '4': '未知股本变动', '5': '股本变化',
            '6': '增发新股', '7': '股份回购', '8': '增发新股上市', '9': '转配股上市', '10': '可转债上市',
            '11': '扩缩股', '12': '非流通股缩股', '13':  '送认购权证', '14': '送认沽权证'}

        data = self.__tdx.to_df(self.__tdx.get_xdxr_info(self.getMarketCode(code), code))

        if len(data) >= 1:
            data = data\
                .assign(date=pd.to_datetime(data[['year', 'month', 'day']], format='%Y-%m-%d'))\
                .drop(['year', 'month', 'day'], axis=1)\
                .assign(category_meaning=data['category'].apply(lambda x: category[str(x)]))\
                .assign(code=str(code))\
                .rename(index=str, columns={'panhouliutong': 'liquidity_after',
                                            'panqianliutong': 'liquidity_before', 'houzongguben': 'shares_after',
                                            'qianzongguben': 'shares_before'})\
                .set_index('date', drop=False, inplace=False)

            xdxrData = data.assign(date=data['date'].apply(lambda x: str(x)[0:10]))
            #xdxrData = xdxrData.set_index('date')
            self._xdxrData = xdxrData
            return xdxrData
        else:
            return None


    #得到股本
    def getGuben(self, code):
        self._connect()

        if self._xdxrData is None:
            xdxrData = self.getXdxr(code)
        else:
            xdxrData = self._xdxrData
        info = xdxrData[xdxrData['liquidity_after'] > 0][['liquidity_after', 'shares_after']]

        circulate = info['liquidity_after'].values[-1] * 10000
        capital = info['shares_after'].values[-1] * 10000

        return capital,circulate


    #按天得到标准数据
    '''
    ktype = D(天)/W(周)/M(月)/Q(季)/Y(年)
    autype = bfq(不复权)/hfq(后复权)/qfq(前复权)
    '''
    def getDays(self, code, ktype = 'D', start = '', end = '', autype = 'qfq', indexs = ['turnover', 'vol', 'ma', 'macd', 'kdj', 'cci', 'bbi', 'sar', 'trix']):
        startDate, endDate, diffDayNum = self._getDate(start, end)

        self._connect()

        ktypeCode = 9
        if ktype.lower() == 'd':
            ktypeCode = 9
        elif ktype.lower() == 'w':
            ktypeCode = 5
        elif ktype.lower() == 'm':
            ktypeCode = 6
        elif ktype.lower() == 'q':
            ktypeCode = 10
        elif ktype.lower() == 'y':
            ktypeCode = 11

        histList = pd.concat([self.__tdx.to_df(self.__tdx.get_security_bars(ktypeCode, self.getMarketCode(code), code, (int(diffDayNum / 800) - i) * 800, 800)) for i in range(int(diffDayNum / 800) + 1)], axis=0)

        if histList.empty:
            return None

        histList = histList[histList['open'] != 0]
        histList = histList[histList['vol'] > 1]

        if not autype or autype == 'bfq':
            histList = histList.assign(date=histList['datetime'].apply(lambda x: str(x[0:10]))).assign(code=str(code))\
                    .assign(date_stamp=histList['datetime'].apply(lambda x: self._dateStamp(str(x)[0:10])))

            histList = histList.drop(['year', 'month', 'day', 'hour', 'minute', 'datetime', 'date_stamp'], axis=1)
            histList = histList.set_index('date')
            histList = histList[startDate:endDate]
            self._lastBaseHistList = histList

            histList['p_change'] = histList['close'].pct_change().round(5) * 100

            if indexs:
                return self.getData(indexs=indexs)
            else:
                return histList

        elif autype == 'qfq':

            bfqData = histList.assign(date=pd.to_datetime(histList['datetime'].apply(lambda x: str(x[0:10])))).assign(code=str(code))\
                .assign(date_stamp=histList['datetime'].apply(lambda x: self._dateStamp(str(x)[0:10])))
            bfqData = bfqData.set_index('date')

            bfqData = bfqData.drop(
                ['year', 'month', 'day', 'hour', 'minute', 'datetime'], axis=1)

            xdxrData = self.getXdxr(code)
            if xdxrData is not None:
                info = xdxrData[xdxrData['category'] == 1]
                bfqData['if_trade'] = True
                data = pd.concat([bfqData, info[['category']]
                                  [bfqData.index[0]:]], axis=1)

                #data['date'] = data.index
                data['if_trade'].fillna(value=False, inplace=True)
                data = data.fillna(method='ffill')
                data = pd.concat([data, info[['fenhong', 'peigu', 'peigujia',
                                              'songzhuangu']][bfqData.index[0]:]], axis=1)
                data = data.fillna(0)

                data['preclose'] = (data['close'].shift(1) * 10 - data['fenhong'] + data['peigu']
                                    * data['peigujia']) / (10 + data['peigu'] + data['songzhuangu'])
                data['adj'] = (data['preclose'].shift(-1) /
                               data['close']).fillna(1)[::-1].cumprod()
                data['open'] = data['open'] * data['adj']
                data['high'] = data['high'] * data['adj']
                data['low'] = data['low'] * data['adj']
                data['close'] = data['close'] * data['adj']
                data['preclose'] = data['preclose'] * data['adj']
                data = data[data['if_trade']]

                histList = data.drop(['fenhong', 'peigu', 'peigujia', 'songzhuangu', 'if_trade', 'category', 'preclose', 'date_stamp', 'adj'], axis=1)
                histList = histList[startDate:endDate]
                self._lastBaseHistList = histList

                histList['p_change'] = histList['close'].pct_change().round(5) * 100

                if indexs:
                    return self.getData(indexs=indexs)
                else:
                    return histList
            else:
                bfqData['preclose'] = bfqData['close'].shift(1)
                bfqData['adj'] = 1

                histList = bfqData.drop(['preclose', 'date_stamp', 'adj'], axis=1)
                histList = histList[startDate:endDate]
                self._lastBaseHistList = histList

                if indexs:
                    return self.getData(indexs=indexs)
                else:
                    return histList

        elif autype == 'hfq':
            xdxrData = self.getXdxr(code)

            info = xdxrData[xdxrData['category'] == 1]

            bfqData = histList.assign(date=histList['datetime'].apply(lambda x: x[0:10])).assign(code=str(code))\
                .assign(date_stamp=histList['datetime'].apply(lambda x: self._dateStamp(str(x)[0:10])))
            bfqData = bfqData.set_index('date')

            bfqData = bfqData.drop(
                ['year', 'month', 'day', 'hour', 'minute', 'datetime'], axis=1)

            bfqData['if_trade'] = True
            data = pd.concat([bfqData, info[['category']]
                              [bfqData.index[0]:]], axis=1)

            data['if_trade'].fillna(value=False, inplace=True)
            data = data.fillna(method='ffill')
            data = pd.concat([data, info[['fenhong', 'peigu', 'peigujia',
                                          'songzhuangu']][bfqData.index[0]:]], axis=1)
            data = data.fillna(0)

            data['preclose'] = (data['close'].shift(1) * 10 - data['fenhong'] + data['peigu']
                                * data['peigujia']) / (10 + data['peigu'] + data['songzhuangu'])
            data['adj'] = (data['preclose'].shift(-1) /
                           data['close']).fillna(1).cumprod()
            data['open'] = data['open'] / data['adj']
            data['high'] = data['high'] / data['adj']
            data['low'] = data['low'] / data['adj']
            data['close'] = data['close'] / data['adj']
            data['preclose'] = data['preclose'] / data['adj']
            data = data[data['if_trade']]

            histList = data.drop(['fenhong', 'peigu', 'peigujia', 'songzhuangu', 'if_trade', 'category', 'preclose', 'date_stamp', 'adj'], axis=1)
            histList = histList[startDate:endDate]
            self._lastBaseHistList = histList

            histList['p_change'] = histList['close'].pct_change().round(5) * 100

            if indexs:
                return self.getData(indexs=indexs)
            else:
                return histList

    #按分钟得到标准数据
    '''
    ktype = 1/5/15/30/60  分钟
    '''
    def getMins(self, code, ktype = 1, start = '', end = '', indexs=['vol', 'ma', 'macd', 'kdj', 'cci', 'bbi', 'sar', 'trix']):
        startDate, endDate, diffDayNum = self._getDate(start, end)

        self._connect()

        ktypeCode = 8
        if int(ktype) == 1:
            ktypeCode = 8
            diffDayNum = 240 * diffDayNum
        elif int(ktype) == 5:
            ktypeCode = 0
            diffDayNum = 48 * diffDayNum
        elif int(ktype) == 15:
            ktypeCode = 1
            diffDayNum = 16 * diffDayNum
        elif int(ktype) == 30:
            ktypeCode = 2
            diffDayNum = 8 * diffDayNum
        elif int(ktype) == 60:
            ktypeCode = 3
            diffDayNum = 4 * diffDayNum

        if diffDayNum > 20800:
            diffDayNum = 20800

        histList = pd.concat([self.__tdx.to_df(self.__tdx.get_security_bars(ktypeCode, self.getMarketCode(
            str(code)), str(code), (int(diffDayNum / 800) - i) * 800, 800)) for i in range(int(diffDayNum / 800) + 1)], axis=0)

        if histList.empty:
            return None

        histList = histList\
            .assign(datetime=pd.to_datetime(histList['datetime']), code=str(code))\
            .assign(date=histList['datetime'].apply(lambda x: str(x)[0:10]))\
            .assign(date_stamp=histList['datetime'].apply(lambda x: self._dateStamp(x)))\
            .assign(time_stamp=histList['datetime'].apply(lambda x: self._timeStamp(x)))

        histList['date'] = histList['datetime']
        histList = histList.drop(['year', 'month', 'day', 'hour', 'minute', 'datetime', 'date_stamp', 'time_stamp'], axis=1)
        histList = histList.set_index('date')
        histList = histList[startDate:endDate]
        self._lastBaseHistList = histList

        histList['p_change'] = histList['close'].pct_change().round(5) * 100
        histList['vol'] = histList['vol'] / 100.0

        if indexs:
            return self.getData(indexs=indexs)
        else:
            return histList


    #按天得到指数日k线
    '''
    ktype = D(天)/W(周)/M(月)/Q(季)/Y(年)
    '''
    def getIndexDays(self, code, ktype = 'D', start = '', end = '', indexs=['turnover', 'vol', 'ma', 'macd', 'kdj', 'cci', 'bbi', 'sar', 'trix']):
        startDate, endDate, diffDayNum = self._getDate(start, end)

        self._connect()

        ktypeCode = 9
        if ktype.lower() == 'd':
            ktypeCode = 9
        elif ktype.lower() == 'w':
            ktypeCode = 5
        elif ktype.lower() == 'm':
            ktypeCode = 6
        elif ktype.lower() == 'q':
            ktypeCode = 10
        elif ktype.lower() == 'y':
            ktypeCode = 11

        if str(code)[0] in ['5', '1']:  # ETF
            data = pd.concat([self.__tdx.to_df(self.__tdx.get_security_bars(
                ktypeCode, 1 if str(code)[0] in ['0', '8', '9', '5'] else 0, code, (int(diffDayNum / 800) - i) * 800, 800)) for i in range(int(diffDayNum / 800) + 1)], axis=0)
        else:
            data = pd.concat([self.__tdx.to_df(self.__tdx.get_index_bars(
                ktypeCode, 1 if str(code)[0] in ['0', '8', '9', '5'] else 0, code, (int(diffDayNum / 800) - i) * 800, 800)) for i in range(int(diffDayNum / 800) + 1)], axis=0)

        histList = data.assign(date=data['datetime'].apply(lambda x: str(x[0:10]))).assign(code=str(code))\
            .assign(date_stamp=data['datetime'].apply(lambda x: self._dateStamp(str(x)[0:10])))\
            .assign(code=code)

        if histList.empty:
            return None

        histList = histList.drop(['year', 'month', 'day', 'hour', 'minute', 'datetime', 'date_stamp', 'up_count', 'down_count'], axis=1)
        histList = histList.set_index('date')
        histList = histList[startDate:endDate]
        self._lastBaseHistList = histList

        histList['p_change'] = histList['close'].pct_change().round(5) * 100

        if indexs:
            return self.getData(indexs=indexs)
        else:
            return histList

    #按分钟得到标准数据
    '''
    ktype = 1/5/15/30/60  分钟
    '''
    def getIndexMins(self, code, ktype = 1, start = '', end = '', indexs=['vol', 'ma', 'macd', 'kdj', 'cci', 'bbi', 'sar', 'trix']):
        startDate, endDate, diffDayNum = self._getDate(start, end)

        self._connect()

        ktypeCode = 8
        if int(ktype) == 1:
            ktypeCode = 8
            diffDayNum = 240 * diffDayNum
        elif int(ktype) == 5:
            ktypeCode = 0
            diffDayNum = 48 * diffDayNum
        elif int(ktype) == 15:
            ktypeCode = 1
            diffDayNum = 16 * diffDayNum
        elif int(ktype) == 30:
            ktypeCode = 2
            diffDayNum = 8 * diffDayNum
        elif int(ktype) == 60:
            ktypeCode = 3
            diffDayNum = 4 * diffDayNum

        if diffDayNum > 20800:
            diffDayNum = 20800

        if str(code)[0] in ['5', '1']:  # ETF
            data = pd.concat([self.__tdx.to_df(self.__tdx.get_security_bars(
                ktypeCode, 1 if str(code)[0] in ['0', '8', '9', '5'] else 0, code, (int(diffDayNum / 800) - i) * 800, 800)) for i in range(int(diffDayNum / 800) + 1)], axis=0)
        else:
            data = pd.concat([self.__tdx.to_df(self.__tdx.get_index_bars(
                ktypeCode, 1 if str(code)[0] in ['0', '8', '9', '5'] else 0, code, (int(diffDayNum / 800) - i) * 800, 800)) for i in range(int(diffDayNum / 800) + 1)], axis=0)

        histList = data.assign(datetime=pd.to_datetime(data['datetime']), code=str(code))\
            .assign(date=data['datetime'].apply(lambda x: str(x)[0:10]))\
            .assign(date_stamp=data['datetime'].apply(lambda x: self._dateStamp(x)))\
            .assign(time_stamp=data['datetime'].apply(lambda x: self._timeStamp(x)))

        if histList.empty:
            return None

        histList['date'] = histList['datetime']
        histList = histList.drop(['year', 'month', 'day', 'hour', 'minute', 'datetime', 'date_stamp', 'time_stamp', 'up_count', 'down_count'], axis=1)
        histList = histList.set_index('date')
        histList = histList[startDate:endDate]
        self._lastBaseHistList = histList

        histList['p_change'] = histList['close'].pct_change().round(5) * 100

        if indexs:
            return self.getData(indexs=indexs)
        else:
            return histList

    #实时逐笔
    '''
    0买 1卖 2中性
    '''
    def getRealtimeTransaction(self, code):
        self._connect()

        try:
            data = pd.concat([self.__tdx.to_df(self.__tdx.get_transaction_data(
                self.getMarketCode(str(code)), code, (2 - i) * 2000, 2000)) for i in range(3)], axis=0)
            if 'value' in data.columns:
                data = data.drop(['value'], axis=1)
            data = data.dropna()
            day = datetime.date.today()
            histList = data.assign(date=str(day)).assign(datetime=pd.to_datetime(data['time'].apply(lambda x: str(day) + ' ' + str(x))))\
                .assign(code=str(code)).assign(order=range(len(data.index)))

            histList['money'] = histList['price'] * histList['vol'] * 100
            histList['type'] = histList['buyorsell']
            histList['type'].replace([0,1,2], ['B','S','N'], inplace = True)

            histList = histList.drop(['order', 'buyorsell'], axis=1).reset_index()
            return histList
        except:
            return None

    #历史逐笔
    '''
    0买 1卖 2中性
    '''
    def getHistoryTransaction(self, code, date):
        self._connect()

        try:
            data = pd.concat([self.__tdx.to_df(self.__tdx.get_history_transaction_data(
                self.getMarketCode(str(code)), code, (2 - i) * 2000, 2000, int(str(date).replace('-', '')))) for i in range(3)], axis=0)
            if 'value' in data.columns:
                data = data.drop(['value'], axis=1)
            data = data.dropna()
            #day = datetime.date.today()
            day = date
            histList = data.assign(date=str(day)).assign(datetime=pd.to_datetime(data['time'].apply(lambda x: str(day) + ' ' + str(x))))\
                .assign(code=str(code)).assign(order=range(len(data.index)))

            histList['money'] = histList['price'] * histList['vol'] * 100
            histList['type'] = histList['buyorsell']
            histList['type'].replace([0,1,2], ['B','S','N'], inplace = True)

            histList = histList.drop(['order', 'buyorsell'], axis=1).reset_index()
            return histList
        except:
            return None

    #实时分时数据
    def getRealtimeMinuteTime(self, code):
        self._connect()

        date = str(time.strftime('%Y-%m-%d',time.localtime()))

        morningData = pd.date_range(start=str(date) + ' 09:31', end=str(date) + ' 11:30', freq = 'min')
        morningDF = pd.DataFrame(index=morningData)


        afternoonData = pd.date_range(start=str(date) + ' 13:01',end=str(date) + ' 15:00', freq = 'min')
        afternoonDF = pd.DataFrame(index=afternoonData)
        timeData = morningDF.append(afternoonDF)

        histList = self.__tdx.to_df(self.__tdx.get_minute_time_data(
                self.getMarketCode(str(code)), code))

        #非标准均价计算
        money = histList['price'] * histList['vol'] * 100
        histList['money'] = money.round(2)
        totalMoney = money.cumsum()
        totalVol = histList['vol'].cumsum()
        histList['ave'] = totalMoney / (totalVol * 100)
        histList['ave'] = histList['ave'].round(3)

        histList['datetime'] = timeData.index[0:len(histList)]
        histList['date'] = histList['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
        histList['time'] = histList['datetime'].apply(lambda x: x.strftime('%H:%M'))

        histList = histList.reset_index()

        return histList

    #历史分时数据
    def getHistoryMinuteTime(self, code, date):
        self._connect()

        morningData = pd.date_range(start=str(date) + ' 09:31', end=str(date) + ' 11:30', freq = 'min')
        morningDF = pd.DataFrame(index=morningData)


        afternoonData = pd.date_range(start=str(date) + ' 13:01',end=str(date) + ' 15:00', freq = 'min')
        afternoonDF = pd.DataFrame(index=afternoonData)
        timeData = morningDF.append(afternoonDF)

        histList = self.__tdx.to_df(self.__tdx.get_history_minute_time_data(
                self.getMarketCode(str(code)), code, int(str(date).replace('-', ''))))

        #非标准均价计算
        money = histList['price'] * histList['vol'] * 100
        histList['money'] = money.round(2)
        totalMoney = money.cumsum()
        totalVol = histList['vol'].cumsum()
        histList['ave'] = totalMoney / (totalVol * 100)
        histList['ave'] = histList['ave'].round(3)

        histList['datetime'] = timeData.index[0:len(histList)]
        histList['date'] = histList['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
        histList['time'] = histList['datetime'].apply(lambda x: x.strftime('%H:%M'))

        histList = histList.reset_index()

        return histList


    #实时报价(五档行情)
    '''
    market => 市场
    active1 => 活跃度
    price => 现价
    last_close => 昨收
    open => 开盘
    high => 最高
    low => 最低
    reversed_bytes0 => 保留
    reversed_bytes1 => 保留
    vol => 总量
    cur_vol => 现量
    amount => 总金额
    s_vol => 内盘
    b_vol => 外盘
    reversed_bytes2 => 保留
    reversed_bytes3 => 保留
    bid1 => 买一价
    ask1 => 卖一价
    bid_vol1 => 买一量
    ask_vol1 => 卖一量
    bid2 => 买二价
    ask2 => 卖二价
    bid_vol2 => 买二量
    ask_vol2 => 卖二量
    bid3 => 买三价
    ask3 => 卖三价
    bid_vol3 => 买三量
    ask_vol3 => 卖三量
    bid4 => 买四价
    ask4 => 卖四价
    bid_vol4 => 买四量
    ask_vol4 => 卖四量
    bid5 => 买五价
    ask5 => 卖五价
    bid_vol5 => 买五量
    ask_vol5 => 卖五量
    reversed_bytes4 => 保留
    reversed_bytes5 => 保留
    reversed_bytes6 => 保留
    reversed_bytes7 => 保留
    reversed_bytes8 => 保留
    reversed_bytes9 => 涨速
    active2 => 活跃度
    '''
    def getRealtimeQuotes(self, codeList):
        self._connect()

        itemList = []
        for item in codeList:
            itemList.append((self.getMarketCode(item), item))

        histList = self.__tdx.to_df(self.__tdx.get_security_quotes(itemList))
        histList = histList.set_index('code')

        return histList

    #计算指定日期成交量细节
    def getVolAnalysis(self, code, date):
        self._connect()

        if str(time.strftime('%Y-%m-%d',time.localtime())) == str(date):
            if int(time.strftime('%H%M',time.localtime())) > 1600:
                volList = self.getHistoryTransaction(code, date)
            else:
                volList = self.getRealtimeTransaction(code)
        else:
            volList = self.getHistoryTransaction(code, date)

        if volList is None:
            return None

        guben,circulate = self.getGuben(code)

        if not self._lastBaseHistList.empty:
            histList = self._lastBaseHistList.copy()
        else:
            histList = self.getDays(code, end=date, indexs=[])

        #涨停单数量
        limitVol = round(histList[-5:]['vol'].mean() * 0.0618)
        #超大单,先转成市值,再转回成手数
        superVol = float(circulate) * float(histList['close'].values[-1]) * 0.000618 / float(histList['close'].values[-1]) / 100
        #大单
        bigVol = round(superVol * 0.518)
        #中单
        middleVol = round(superVol * 0.382)
        #小单
        smallVol = round(superVol * 0.191)

        #买单统计
        buyVolList = volList[volList['type'] == 'B']
        totalBuyVolNum = buyVolList['vol'].sum()
        mainBuyVolNum = buyVolList[buyVolList['vol'] >= bigVol]['vol'].sum()
        limitBuyVolNum = math.ceil(buyVolList[(buyVolList['vol'] >= limitVol)]['vol'].sum() / limitVol)
        superBuyVolNum = math.ceil(buyVolList[(buyVolList['vol'] < limitVol) & (buyVolList['vol'] >= superVol)]['vol'].sum() / superVol)
        bigBuyVolNum = math.ceil(buyVolList[(buyVolList['vol'] < superVol) & (buyVolList['vol'] >= bigVol)]['vol'].sum() / bigVol)
        middleBuyVolNum = math.ceil(buyVolList[(buyVolList['vol'] < bigVol) & (buyVolList['vol'] >= middleVol)]['vol'].sum() / middleVol)
        smallBuyVolNum = math.ceil(buyVolList[(buyVolList['vol'] < middleVol) & (buyVolList['vol'] >= smallVol)]['vol'].sum() / smallVol)
        microBuyVolNum = len(buyVolList[(buyVolList['vol'] < smallVol)])
        #print limitBuyVolNum,superBuyVolNum,bigBuyVolNum,middleBuyVolNum,smallBuyVolNum,microBuyVolNum

        #卖单统计
        sellVolList = volList[volList['type'] == 'S']
        totalSellVolNum = sellVolList['vol'].sum()
        mainSellVolNum = sellVolList[sellVolList['vol'] >= bigVol]['vol'].sum()
        limitSellVolNum = math.ceil(sellVolList[(sellVolList['vol'] >= limitVol)]['vol'].sum() / limitVol)
        superSellVolNum = math.ceil(sellVolList[(sellVolList['vol'] < limitVol) & (sellVolList['vol'] >= superVol)]['vol'].sum() / superVol)
        bigSellVolNum = math.ceil(sellVolList[(sellVolList['vol'] < superVol) & (sellVolList['vol'] >= bigVol)]['vol'].sum() / bigVol)
        middleSellVolNum = math.ceil(sellVolList[(sellVolList['vol'] < bigVol) & (sellVolList['vol'] >= middleVol)]['vol'].sum() / middleVol)
        smallSellVolNum = math.ceil(sellVolList[(sellVolList['vol'] < middleVol) & (sellVolList['vol'] >= smallVol)]['vol'].sum() / smallVol)
        microSellVolNum = len(sellVolList[(sellVolList['vol'] < smallVol)])
        #print limitSellVolNum,superSellVolNum,bigSellVolNum,middleSellVolNum,smallSellVolNum,microSellVolNum

        #计算吸筹线
        #主力标准吸筹金额
        mainBaseMoney = round(histList['close'].values[-1] * circulate * 0.001 / 10000 / 10000, 4)
        #主力强力吸筹金额
        mainBigMoney = round(histList['close'].values[-1] * circulate * 0.003 / 10000 / 10000, 4)

        #资金统计
        totalMoney = round(volList['money'].sum() / 10000 / 10000, 4)
        totalBuyMoney = round(buyVolList['money'].sum() / 10000 / 10000, 4)
        totalSellMoney = round(sellVolList['money'].sum() / 10000 / 10000, 4)
        totalAbsMoney = round(totalBuyMoney - totalSellMoney, 3)
        mainMoney = round(volList[volList['vol'] >= bigVol]['money'].sum() / 10000 / 10000, 4)
        mainBuyMoney = round(buyVolList[buyVolList['vol'] >= bigVol]['money'].sum() / 10000 / 10000, 4)
        mainSellMoney = round(sellVolList[sellVolList['vol'] >= bigVol]['money'].sum() / 10000 / 10000, 4)
        mainAbsMoney = round(mainBuyMoney - mainSellMoney, 3)

        mainRate = 0
        try:
            mainRate = round((mainBuyMoney + mainSellMoney) / totalMoney * 100, 2)
        except:
            pass

        mainBuyRate = 0
        try:
            mainBuyRate = round(mainBuyMoney / (mainBuyMoney + mainSellMoney) * 100, 2)
        except:
            pass
        #print totalAbsMoney,mainAbsMoney,totalMoney,totalBuyMoney,totalSellMoney,mainBuyMoney,mainSellMoney,mainRate,mainBuyRate

        #成交笔数
        volNum = len(volList)

        #平均每笔交易价格
        aveTradePrice = round(totalMoney / volNum * 10000 * 10000, 2)

        #平均每股买价格
        avePerShareBuyPrice = 0
        try:
            avePerShareBuyPrice = round(totalBuyMoney * 10000 * 10000 / (totalBuyVolNum * 100), 3)
        except:
            pass

        #主力平均每股买价格
        mainAvePerShareBuyPrice = 0
        try:
            mainAvePerShareBuyPrice = round(mainBuyMoney * 10000 * 10000 / (mainBuyVolNum * 100), 3)
        except:
            pass

        #平均每股卖价格
        avePerShareSellPrice = 0
        try:
            avePerShareSellPrice = round(totalSellMoney * 10000 * 10000 / (totalSellVolNum * 100), 3)
        except:
            pass

        #主力平均每股卖价格
        mainAvePerShareSellPrice = 0
        try:
            mainAvePerShareSellPrice = round(mainSellMoney * 10000 * 10000 / (mainSellVolNum * 100), 3)
        except:
            pass

        #print totalMoney,volNum,aveVolPrice * 10000 * 10000
        statData = {}
        statData['limit_buy_vol_num'] = limitBuyVolNum
        statData['super_buy_vol_num'] = superBuyVolNum
        statData['big_buy_vol_num'] = bigBuyVolNum
        statData['middle_buy_vol_num'] = middleBuyVolNum
        statData['small_buy_vol_num'] = smallBuyVolNum
        statData['micro_buy_vol_num'] = microBuyVolNum
        statData['limit_sell_vol_num'] = limitSellVolNum
        statData['super_sell_vol_num'] = superSellVolNum
        statData['big_sell_vol_num'] = bigSellVolNum
        statData['middle_sell_vol_num'] = middleSellVolNum
        statData['small_sell_vol_num'] = smallSellVolNum
        statData['micro_sell_vol_num'] = microSellVolNum
        statData['total_abs_money'] = totalAbsMoney
        statData['main_abs_money'] = mainAbsMoney
        statData['total_money'] = totalMoney
        statData['total_buy_money'] = totalBuyMoney
        statData['total_sell_money'] = totalSellMoney
        statData['main_money'] = mainMoney
        statData['main_buy_money'] = mainBuyMoney
        statData['main_sell_money'] = mainSellMoney
        statData['main_rate'] = mainRate
        statData['main_buy_rate'] = mainBuyRate
        statData['trade_num'] = volNum
        statData['vol_num'] = volList['vol'].sum()
        statData['ave_trade_price'] = aveTradePrice
        statData['main_base_money'] = mainBaseMoney
        statData['main_big_money'] = mainBigMoney
        statData['ave_per_share_buy_price'] = avePerShareBuyPrice
        statData['ave_per_share_sell_price'] = avePerShareSellPrice
        statData['main_ave_per_share_buy_price'] = mainAvePerShareBuyPrice
        statData['main_ave_per_share_sell_price'] = mainAvePerShareSellPrice
        statData['circulate_money'] = round(circulate * histList['close'].values[-1] / 10000 / 10000, 4)

        return statData

    #输出ebk文件
    def outputEbk(self, stockList, ebkPath = ''):

        if len(ebkPath) <= 0:
            ebkPath = os.getcwd() + '/' + sys.argv[0][0:-3] + '.' + str(time.strftime('%Y%m%d',time.localtime())) + '.ebk'

        if not isinstance(stockList,list):
            return False

        fp = open(ebkPath, "a")
        fp.write('\r\n')    #ebk第一行为空行
        for code in stockList:
            if self.getMarketCode(code) == 1:
                fp.write('1' + code)
            else:
                fp.write('0' + code)

            fp.write('\r\n')

        fp.close()

        return True

    #输出sel文件
    def outputSel(self, stockList, selPath = ''):
        import struct

        if len(selPath) <= 0:
            selPath = os.getcwd() + '/' + sys.argv[0][0:-3] + '.' + str(time.strftime('%Y%m%d',time.localtime())) + '.sel'

        if not isinstance(stockList,list):
            return False

        stocks = []
        for code in stockList:
            if self.getMarketCode(code) == 1:
                stocks.append('\x07\x11' + code)
            else:
                stocks.append('\x07\x21' + code)

        with open(selPath, 'ab') as fp:
            data = struct.pack('H', len(stocks)).decode() + ''.join(stocks)
            fp.write(data.encode())

        return True

    #ebk to sel
    def ebk2sel(self, ebkPath):
        import struct

        if not os.path.exists(ebkPath):
            return False

        selPath = ebkPath.replace('.ebk', '.sel')

        stocks = []
        with open(ebkPath, 'r') as ebkfp:
            for code in ebkfp:
                code = code.strip()
                if len(code) > 0:
                    if self.getMarketCode(code[1:]) == 1:
                        stocks.append('\x07\x11' + code[1:])
                    else:
                        stocks.append('\x07\x21' + code[1:])

        with open(selPath, 'wb') as selfp:
            data = struct.pack('H', len(stocks)).decode() + ''.join(stocks)
            selfp.write(data.encode())


        return True

    #sel to ebk
    def sel2ebk(self, selPath):
        import struct

        if not os.path.exists(selPath):
            return False

        ebkPath = selPath.replace('.sel', '.ebk')

        with open(selPath, 'rb') as selfp:
            ebkfp = open(ebkPath, "a")
            cnt = struct.unpack('<H', selfp.read(2))[0]
            for _ in range(cnt):
                data = selfp.readline(8).decode()
                exch = '1' if data[1] == '\x11' else '0'
                code = exch + data[2:]

                ebkfp.write(code + '\r\n')

            ebkfp.close()

        return True

