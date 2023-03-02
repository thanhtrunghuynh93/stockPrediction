from collections import deque
import matplotlib.pyplot as plt

def plot_psar(df):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    psar_bull = df.loc[df['PSAR_trend']==1]['PSAR']
    psar_bear = df.loc[df['PSAR_trend']==0]['PSAR']

    plt.figure(figsize=(12, 8))
    plt.plot(df['close'], label='Close', linewidth=1)
    plt.scatter(psar_bull.index, psar_bull, color=colors[1], label='Up Trend')
    plt.scatter(psar_bear.index, psar_bear, color=colors[3], label='Down Trend')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} Price and Parabolic SAR')
    plt.legend()
    plt.show()

class PSAR:

    def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
        self.max_af = max_af
        self.init_af = init_af
        self.af = init_af
        self.af_step = af_step
        self.extreme_point = None
        self.high_price_trend = []
        self.low_price_trend = []
        self.high_price_window = deque(maxlen=2)
        self.low_price_window = deque(maxlen=2)

        # Lists to track results
        self.psar_list = []
        self.af_list = []
        self.ep_list = []
        self.high_list = []
        self.low_list = []
        self.trend_list = []        

    def calcPSAR(self, highs, lows):
                
        for num_days in range(len(highs)):
            
            if num_days >= 3:
                psar = self._calcPSAR()
            else:
                psar = self._initPSARVals(highs[num_days], lows[num_days])

            psar = self._updateCurrentVals(psar, highs[num_days], lows[num_days])      

        #Fill NA
        self.psar_list[0] = self.psar_list[2]      
        self.psar_list[1] = self.psar_list[2]      

        return self.psar_list

    def _initPSARVals(self, high, low):
        
        if len(self.low_price_window) <= 1:
            self.trend = None
            self.extreme_point = high
            return None
        
        if self.high_price_window[0] < self.high_price_window[1]:
            self.trend = 1
            psar = min(self.low_price_window)
            self.extreme_point = max(self.high_price_window)
        else:
            self.trend = 0
            psar = max(self.high_price_window)
            self.extreme_point = min(self.low_price_window)

        return psar

    def _calcPSAR(self):
        prev_psar = self.psar_list[-1]
        if self.trend == 1:  # Up
            psar = prev_psar + self.af * (self.extreme_point - prev_psar)
            psar = min(psar, min(self.low_price_window))
        else:
            psar = prev_psar - self.af * (prev_psar - self.extreme_point)
            psar = max(psar, max(self.high_price_window))

        return psar

    def _updateCurrentVals(self, psar, high, low):
        if self.trend == 1:
            self.high_price_trend.append(high)
        elif self.trend == 0:
            self.low_price_trend.append(low)

        psar = self._trendReversal(psar, high, low)

        self.psar_list.append(psar)
        self.af_list.append(self.af)
        self.ep_list.append(self.extreme_point)
        self.high_list.append(high)
        self.low_list.append(low)
        self.high_price_window.append(high)
        self.low_price_window.append(low)
        self.trend_list.append(self.trend)

        return psar

    def _trendReversal(self, psar, high, low):
                
        # Checks for reversals
        reversal = False
        if self.trend == 1 and psar > low:
            self.trend = 0
            psar = max(self.high_price_trend)
            self.extreme_point = low
            reversal = True
        elif self.trend == 0 and psar < high:
            self.trend = 1
            psar = min(self.low_price_trend)
            self.extreme_point = high
            reversal = True

        if reversal:
            self.af = self.init_af
            self.high_price_trend.clear()
            self.low_price_trend.clear()
        else:
            
            if high > self.extreme_point and self.trend == 1:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = high
            elif low < self.extreme_point and self.trend == 0:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = low

        return psar
