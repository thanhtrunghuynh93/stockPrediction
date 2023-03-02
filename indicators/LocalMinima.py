from scipy.signal import argrelextrema
import numpy as np

def FindLocalMinima(df):
    n = 5

    df['min'] = df.iloc[argrelextrema(df.low.values, np.less_equal, order=n)[0]]['close']
    df['max'] = df.iloc[argrelextrema(df.high.values, np.greater_equal, order=n)[0]]['close']

    # Extract only rows where local peaks are not null
    dfMax = df[df['max'].notnull()]
    dfMin = df[df['min'].notnull()]

    # Remove all local maximas which have other maximas close to them 
    prevIndex = -1
    currentIndex = 0
    dropRows = []
    # find indices
    for i1, p1 in dfMax.iterrows():
        currentIndex = i1
        if currentIndex <= prevIndex + n * 0.64:
            dropRows.append(currentIndex)
        prevIndex = i1
    # drop them from the max df
    dfMax = dfMax.drop(dropRows)
    # replace with nan in initial df
    for ind in dropRows:
        df.iloc[ind, :]['max'] = np.nan

    # Remove all local minimas which have other minimas close to them 
    prevIndex = -1
    currentIndex = 0
    dropRows = []
    # find indices
    for i1, p1 in dfMin.iterrows():
        currentIndex = i1
        if currentIndex <= prevIndex + n * 0.64:
            dropRows.append(currentIndex)
        prevIndex = i1
    # drop them from the min df
    dfMin = dfMin.drop(dropRows)
    # replace with nan in initial df
    for ind in dropRows:
        df.iloc[ind, :]['min'] = np.nan

    return dfMax, dfMin