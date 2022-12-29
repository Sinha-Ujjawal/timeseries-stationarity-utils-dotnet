namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics.Distributions;

    public sealed class QStat
    {
        public struct QStatistics
        {
            readonly public double[] qStats; // The Ljung-Box Q-Statistic for lags 1, 2, ..., nlags (excludes lag zero). Returned if qstat is True.
            readonly public double[] pvalues; // The p-values associated with the Q-statistics for lags 1, 2, ..., nlags (excludes lag zero). Returned if qstat is True.

            public QStatistics(double[] qStats, double[] pvalues)
            {
                this.qStats = qStats;
                this.pvalues = pvalues;
            }
        }

        /// <summary>
        /// Compute Ljung-Box Q Statistic.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L532
        /// </summary>
        public static QStatistics Run(double[] timeSeries, uint nobs)
        {
            var range = Enumerable.Range(1, timeSeries.Length).AsVector();
            var ret = (
                nobs
                * (nobs + 2)
                * (
                    (1.0 / (nobs - range))
                    .PointwiseMultiply(timeSeries.AsVector().PointwisePower(2))
                    .RunningSum()
                )
            );
            var chi2 = ret.Zip(range).Select(values => 1 - ChiSquared.CDF(freedom: values.Second, x: values.First));
            return new QStatistics(qStats: ret.ToArray(), pvalues: chi2.ToArray());
        }
    }
}