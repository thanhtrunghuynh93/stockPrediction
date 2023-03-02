import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal

def get_dist_plot(c, v, kx, ky):
    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=c, y=v, nbinsx=150, 
                               histfunc='sum', histnorm='probability density',
                               marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='#D2691E'))
    return fig


class VolumeProfileAnalyzer:

    def __init__(self, volume_col = "volume", close_col = "close"):
        
        self.volume_col = volume_col
        self.close_col = close_col
        
    def get_resistance_support_ranges(self, df, kde_factor = 0.05, num_samples = 500, min_prom_coef = 0.3, verbose = 0, bullish_trend = False): 

        ranges = []
        volume = df[self.volume_col]
        close = df[self.close_col]

        kde = stats.gaussian_kde(close, weights=volume, bw_method=kde_factor)
        xr = np.linspace(close.min(),close.max(), num_samples)
        ticks_per_sample = (xr.max() - xr.min()) / num_samples

        kdy = kde(xr)
        min_prom = kdy.max() * min_prom_coef
        peaks, peak_props = signal.find_peaks(kdy, prominence = min_prom, width = 1)
        pkx = xr[peaks]
        pky = kdy[peaks]
        nx = [close[-1]]
        ny = [volume[-1] / volume.max() * kdy.max()]
        safety_margin = 1.02
        
        left_ips = peak_props['left_ips']
        right_ips = peak_props['right_ips']
        width_x0 = xr.min() + (left_ips * ticks_per_sample)
        width_x1 = xr.min() + (right_ips * ticks_per_sample)
        width_y = peak_props['width_heights']

        current_support = min(np.min(close), close[-1])
        current_support_index = -1        
        current_resistance = max(np.max(close), close[-1])
        if current_support == close[-1]:
            current_support = current_support * 0.8
        if bullish_trend and current_resistance == close[-1]: #If current price is the max price ever
            current_resistance = current_resistance * 1.2 #Only in bullish trend
        current_resistance_index = -1
        
        idx = 0
        for pkx, x0, x1, y in zip(pkx, width_x0, width_x1, width_y):
            ranges.append([pkx, x0, x1, y / kdy.mean()])
            if pkx > current_support and close[-1] > pkx * safety_margin: # Current price is greater than the level, then level is support
                current_support = pkx
                current_support_index = idx
            if pkx < current_resistance and pkx > close[-1] * safety_margin: # Current price is greater than the level, then level is support
                current_resistance = pkx
                current_resistance_index = idx
            idx += 1
            if verbose == 1:        
                print("Resistance/Support: {:.2f} from {:.2f} to {:.2f}, average vol compare to average vol: {:.2f}".format(pkx, x0, x1, y / kdy.mean()))
                
        return ranges, current_support, current_support_index, current_resistance, current_resistance_index

    def visualize_profile(self, df, kde_factor = 0.05, num_samples = 500, min_prom_coef = 0.3, verbose = 0):

        volume = df[self.volume_col]
        close = df[self.close_col]

        kde = stats.gaussian_kde(close, weights=volume, bw_method=kde_factor)
        xr = np.linspace(close.min(),close.max(), num_samples)
        kdy = kde(xr)
        ticks_per_sample = (xr.max() - xr.min()) / num_samples
        fig = get_dist_plot(close, volume, xr, kdy)

        min_prom = kdy.max() * 0.3
        width_range=1
        peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom, width=width_range)
        pkx = xr[peaks]
        pky = kdy[peaks]
        nx = [close[-1]]
        ny = [volume[-1] / volume.max() * kdy.max()]
        pk_marker_args=dict(size=10)

        left_ips = peak_props['left_ips']
        right_ips = peak_props['right_ips']
        width_x0 = xr.min() + (left_ips * ticks_per_sample)
        width_x1 = xr.min() + (right_ips * ticks_per_sample)
        width_y = peak_props['width_heights']

        fig = get_dist_plot(close, volume, xr, kdy)
        fig.add_trace(go.Scatter(name='Peaks', x=pkx, y=pky, mode='markers', marker=pk_marker_args))
        fig.add_trace(go.Scatter(name='Current_price', x=nx, y=ny, mode='markers', marker=pk_marker_args))

        for x0, x1, y in zip(width_x0, width_x1, width_y):
            fig.add_shape(type='line',
                xref='x', yref='y',
                x0=x0, y0=y, x1=x1, y1=y,
                line=dict(
                    color='red',
                    width=2,
                )
            )
        return fig


    
