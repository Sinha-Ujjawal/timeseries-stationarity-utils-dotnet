## Timeseries Stationarity Utility in [C#](https://learn.microsoft.com/en-us/dotnet/csharp/) Dotnet

This is a small library written in [C#](https://learn.microsoft.com/en-us/dotnet/csharp/) that does [Timeseries Stationarity Tests](https://machinelearningmastery.com/time-series-data-stationary-python/) and has some utility to deal with Stationarity. This is a direct port of python's [statsmodels](https://www.statsmodels.org/stable/index.html) library.

## Algorithms Checklist

| Alogorithm                                                                                                                                                  | Implemented |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [KPSS (Kwiatkowski-Phillips-Schmidt-Shin)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html#statsmodels.tsa.stattools.kpss) | ✓           |
| [ACF (Auto Correlation Function)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf)            | ✓         |
| [STL (Season-Trend decomposition using LOESS)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html#statsmodels.tsa.seasonal.STL) | In Progress...        |
| [ADF (Augmented Dickey-Fuller unit root test.))](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller) | ✓        |

## Note

**_Use at your own risks. I haven't tested the library thoroughly._**

## Usage

### KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

```C#
struct KPSSTestStatistic
{
    readonly public double kpssStat;
    readonly public double pValue;
    readonly public uint nlags;
    readonly public double crit10Percent;
    readonly public double crit5Percent;
    readonly public double crit2Point5Percent;
    readonly public double crit1Percent;
}

KPSSTestStatistic TimeSeriesStationaryUtils.KPSS.Run(
    double[] timeSeries, // time series array
    bool isStationaryAroundTrend = false, // by default, Is Stationary Around Constant would be used
    int nlags = -1 // <0: Auto, 0: Legacy, >0: number of lags used
)
```

### ACF (Auto Correlation Function)

```C#
struct QStatistics
{
    readonly public double[] qStats; // The Ljung-Box Q-Statistic for lags 1, 2, ..., nlags (excludes lag zero). Returned if qstat is True.
    readonly public double[] pvalues; // The p-values associated with the Q-statistics for lags 1, 2, ..., nlags (excludes lag zero). Returned if qstat is True.
}

struct ACFStatistic
{
    readonly public double[] acf; // The autocorrelation function for lags 0, 1, ..., nlags. Shape (nlags+1,).
    readonly public (double First, double Second)[]? confInt; // Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape (nlags + 1, 2). Returned if alpha is not null.
    readonly public QStat.QStatistics? qStatistics;
}

ACFStatistic TimeSeriesStationaryUtils.ACF.Run(
    double[] timeSeries,
    bool adjusted = false, // If True, then denominators for autocovariance are n-k, otherwise n.
    uint? nlags = null,
    // Number of lags to return autocorrelation for. If not provided,
    // uses min(10 * np.log10(nobs), nobs - 1). The returned value
    // includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
    bool qstat = false, // If True, returns the Ljung-Box q statistic for each autocorrelation coefficient.
    double? alpha = null,
    // If a number is given, the confidence intervals for the given level are
    // returned. For instance if alpha=.05, 95 % confidence intervals are
    // returned where the standard deviation is computed according to
    // Bartlett"s formula.
    bool bartlettConfInt = true
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

### ADF (Augmented Dick-Fuller Test)

```C#
struct MackinnonCritValues
{
    double crit1Percent;
    double crit5Percent;
    double crit10Percent;
}

struct ADFStatistic
{
    double adfStat; // The test statistic.
    double pvalue; // MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    uint usedlag; // The number of lags used.
    uint nobs; // The number of observations used for the ADF regression and calculation of the critical values.
    MackinnonCritValues? mackinnonCritValues;
    // Critical values for the test statistic at the 1 %, 5 %, and 10 %
    // levels. Based on MacKinnon (2010).
    double? icbest; // The maximized information criterion if autolag is not None.
}

ADFStatistic TimeSeriesStationaryUtils.ADF.Run(
    double[] timeSeries,
    uint? maxlag=null, // Maximum lag which is included in test, default value of 12*(nobs/100)^{1/4} is used when ``null``.
    Regression regression=Regression.C,
    // Constant and trend order to include in regression.
    // * "c" : constant only (default).
    // * "ct" : constant and trend.
    // * "ctt" : constant, and linear and quadratic trend.
    // * "n" : no constant, no trend.
    OLS.AutoLagCriterion? autoLagCriterion=OLS.AutoLagCriterion.AIC,
    //  Method to use when automatically determining the lag length among the
    // values 0, 1, ..., maxlag.
    // * If "AIC" (default) or "BIC", then the number of lags is chosen
    // to minimize the corresponding information criterion.
    // * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
    // lag until the t-statistic on the last lag length is significant
    // using a 5%-sized test.
    // * If None, then the number of included lags is set to maxlag.
    QRMethod qRMethod = QRMethod.Thin
    // qRMethod to use, QRMethod.Thin is the fastest
)
```

## Copyrights

Licensed under [@MIT](./LICENSE)
