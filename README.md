## Timeseries Stationarity Utility in [C#](https://learn.microsoft.com/en-us/dotnet/csharp/) Dotnet

This is a small library written in [C#](https://learn.microsoft.com/en-us/dotnet/csharp/) that does [Timeseries Stationarity Tests](https://machinelearningmastery.com/time-series-data-stationary-python/) and has some utility to deal with Stationarity. This is a direct port of python's [statsmodels](https://www.statsmodels.org/stable/index.html) library.

## Algorithms Checklist

| Alogorithm                                                                                                                                                  | Implemented |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [KPSS (Kwiatkowski-Phillips-Schmidt-Shin)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html#statsmodels.tsa.stattools.kpss) | ✓           |
| [ACF (Auto Correlation Function)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf)            | ✓         |
| [STL (Season-Trend decomposition using LOESS)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html#statsmodels.tsa.seasonal.STL) | TODO        |

## Note

**_Use at your own risks. I haven't tested the library thoroughly._**

## Usage

### KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

```C#
TimeSeriesStationaryUtils.Algorithm.KPSS(
    timeSeries: yourTimeSeries, // double[] array
    isStationaryAroundTrend: false, // by default, Is Stationary Around Constant would be used
    nlags = -1 // <0: Auto (using Hobijn et al (1998)), 0: Legacy, >0: number of lags used
)
```

### ACF (Auto Correlation Function)

```C#
TimeSeriesStationaryUtils.Algorithm.ACF(
    timeSeries: yourTimeSeries,  // double[] array
    adjusted: false, // If True, then denominators for autocovariance are n-k, otherwise n.
    nlags: null,
    // Number of lags to return autocorrelation for. If not provided,
    // uses min(10 * np.log10(nobs), nobs - 1). The returned value
    // includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
    qstat: false, // If True, returns the Ljung-Box q statistic for each autocorrelation coefficient.
    alpha: null,
    // If a number is given, the confidence intervals for the given level are
    // returned. For instance if alpha=.05, 95 % confidence intervals are
    // returned where the standard deviation is computed according to
    // Bartlett"s formula.
    bartlettConfInt: true
    // Confidence intervals for ACF values are generally placed at 2
    // standard errors around r_k. The formula used for standard error
    // depends upon the situation. If the autocorrelations are being used
    // to test for randomness of residuals as part of the ARIMA routine,
    // the standard errors are determined assuming the residuals are white
    // noise. The approximate formula for any lag is that standard error
    // of each r_k = 1/sqrt(N).
    // For the ACF of raw data, the standard error at a lag k is
    // found as if the right model was an MA(k-1). This allows the possible
    // interpretation that if all autocorrelations past a certain lag are
    // within the limits, the model might be an MA of order defined by the
    // last significant autocorrelation. In this case, a moving average
    // model is assumed for the data and the standard errors for the
    // confidence intervals should be generated using Bartlett's formula.
)
```

## Copyrights

Licensed under [@MIT](./LICENSE)
