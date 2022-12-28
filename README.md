## Timeseries Stationarity Utility in [C#](https://learn.microsoft.com/en-us/dotnet/csharp/) Dotnet

This is a small library written in [C#](https://learn.microsoft.com/en-us/dotnet/csharp/) that does [Timeseries Stationarity Tests](https://machinelearningmastery.com/time-series-data-stationary-python/) and has some utility to deal with Stationarity. This is a direct port of python's [statsmodels](https://www.statsmodels.org/stable/index.html) library.

## Algorithms Checklist

| Alogorithm                                                                                                                                                  | Implemented |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [KPSS (Kwiatkowski-Phillips-Schmidt-Shin)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html#statsmodels.tsa.stattools.kpss) | ✓           |
| [ACF (Auto Correlation Function)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf)            | TODO        |
| [STL (Season-Trend decomposition using LOESS)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html#statsmodels.tsa.seasonal.STL) | TODO        |

## Note

**_Use at your own risks. I haven't tested the library thoroughly._**

## Usage

### KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

```C#
TimeSeriesStationaryUtils.Algorithm.KPSS(
    timeSeries: yourTimeSeries, // double[] array
    isStationaryAroundTrend: false, // by default, Is Stationary Around Constant would be used
    int nlags = -1 // <0: Auto (using Hobijn et al (1998)), 0: Legacy, >0: number of lags used
)
```

## Copyrights

Licensed under [@MIT](./LICENSE)
