namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.Distributions;

    public sealed class ACF
    {
        public struct ACFStatistic
        {
            readonly public double[] acf; // The autocorrelation function for lags 0, 1, ..., nlags. Shape (nlags+1,).
            readonly public (double First, double Second)[]? confInt; // Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape (nlags + 1, 2). Returned if alpha is not null.
            readonly public QStat.QStatistics? qStatistics;

            public ACFStatistic(
                double[] acf,
                (double First, double Second)[]? confInt = null,
                QStat.QStatistics? qStatistics = null
            )
            {
                this.acf = acf;
                this.confInt = confInt;
                this.qStatistics = qStatistics;
            }
        }

        /// <summary>
        /// Calculate the autocorrelation function.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L577
        /// </summary>
        public static ACFStatistic Run(
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
        {
            uint nobs = (uint) timeSeries.Length;
            if (nlags is null)
                nlags = Math.Min((uint)(10 * Math.Log10(nobs)), (uint)nobs - 1);
            var avf = ACOVF.Run(timeSeries, adjusted: adjusted, demean: true);      
            var acf = avf.Take((int) nlags + 1).AsVector() / avf[0];
            if (!(qstat || (alpha is not null)))
                return new ACFStatistic(acf: acf.ToArray());
            if (alpha is null)
                alpha = 0.05;
            Vector<double> varacf;
            if (bartlettConfInt)
            {
                varacf = (
                    (new double[] {0.0, 1.0})
                    .Concat(1 + 2 * acf.Skip(1).SkipLast(1).AsVector().PointwisePower(2).RunningSum())
                    .AsVector() 
                    / nobs
                );
            }
            else
                varacf = Enumerable.Repeat(1.0 / nobs, acf.Count()).AsVector();
            var interval = Normal.InvCDF(mean: 0, stddev: 1, p: 1 - (double) alpha / 2.0) * varacf.PointwiseSqrt();
            var confInt = (acf - interval).Zip(acf + interval).ToArray();
            if (!qstat)
                return new ACFStatistic(acf: acf.ToArray(), confInt: confInt);
            var qStatistics = QStat.Run(acf.Skip(1).ToArray(), nobs: nobs);  // drop lag 0
            if (alpha is not null)
                return new ACFStatistic(acf: acf.ToArray(), confInt: confInt, qStatistics: qStatistics);
            else
                return new ACFStatistic(acf: acf.ToArray(), qStatistics: qStatistics);
        }
    }
}