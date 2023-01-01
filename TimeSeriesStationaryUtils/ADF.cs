namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics.LinearAlgebra;

    public static class ADF
    {
        public struct ADFStatistic
        {
            double adf; // The test statistic.
            double pvalue; // MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
            uint usedlag; // The number of lags used.
            uint nobs; // The number of observations used for the ADF regression and calculation of the critical values.
            Dictionary<String, double> criticalValues;
            // Critical values for the test statistic at the 1 %, 5 %, and 10 %
            // levels. Based on MacKinnon (2010).
            double icbest; // The maximized information criterion if autolag is not None.
        }

        /// <summary>
        /// Augmented Dickey-Fuller unit root test.
        /// The Augmented Dickey-Fuller test can be used to test for a unit root in a
        /// univariate process in the presence of serial correlation.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L166
        ///</summary>
        public static ADFStatistic Run(
            double[] timeSeries,
            uint? maxlag=null, // Maximum lag which is included in test, default value of 12*(nobs/100)^{1/4} is used when ``null``.
            Extensions.Trend regression=Extensions.Trend.C,
            // Constant and trend order to include in regression.
            // * "c" : constant only (default).
            // * "ct" : constant and trend.
            // * "ctt" : constant, and linear and quadratic trend.
            // * "n" : no constant, no trend.
            OLS.AutoLagCriterion? autoLagCriterion=OLS.AutoLagCriterion.AIC
            //  Method to use when automatically determining the lag length among the
            // values 0, 1, ..., maxlag.
            // * If "AIC" (default) or "BIC", then the number of lags is chosen
            // to minimize the corresponding information criterion.
            // * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
            // lag until the t-statistic on the last lag length is significant
            // using a 5%-sized test.
            // * If None, then the number of included lags is set to maxlag.
        )
        {
            if (timeSeries.Min() == timeSeries.Max())
                throw new ArgumentException("Invalid input, timeSeries is constant");
            
            var nobs = (uint) timeSeries.Length;
            var ntrend = regression.ntrend();
            
            if (maxlag is null)
            {
                // from Greene referencing Schwert 1989
                var maxlagCandidate = (int)(Math.Ceiling(12.0 * Math.Pow(nobs / 100.0, 1 / 4.0)));
                // -1 for the diff
                maxlagCandidate = Math.Min((int) nobs / 2 - (int) ntrend - 1, (int) maxlagCandidate);
                if (maxlagCandidate < 0)
                    throw new NotSupportedException(@"sample size is too short to use selected regression component");
                maxlag = (uint) maxlagCandidate;
            }
            else if (maxlag > (nobs / 2 - ntrend - 1))
                throw new ArgumentException(@"maxlag must be less than (nobs/2 - 1 - ntrend) where ntrend is the number of included deterministic regressors");
            
            var timeSeriesDiff = timeSeries.Diff().AsVector();
            var timeSeriesDAll = (
                timeSeriesDiff.ToColumnMatrix()
                .LagMat((uint) maxlag)
                .OriginalWithLaggedValues(trim: Extensions.Trim.Both) // original="in" // with lagged values; with trim="both"
            );
            nobs = (uint) timeSeriesDAll.RowCount;
            timeSeriesDAll.SetSubMatrix(
                rowIndex: 0,
                rowCount: timeSeriesDAll.RowCount,
                columnIndex: 0,
                columnCount: 1,
                timeSeries
                    .TakeLast((int) nobs + 1)
                    .SkipLast(1)
                    .AsMatrix(nRows: (uint) timeSeriesDAll.RowCount, nCols: 1)
            ); // replace 0 timeSeriesDiff with level of timeSeries
            var timeSeriesDShort = timeSeriesDiff.TakeLast((int) nobs).AsVector();
            
            if (autoLagCriterion is not null)
            {
                Matrix<double> fullRHS;
                if (regression != Extensions.Trend.N)
                {
                    fullRHS = timeSeriesDAll.AddTrend(trend: regression, prepend: true);
                }
                else
                    fullRHS = timeSeriesDAll;
                var startlag = fullRHS.ColumnCount - timeSeriesDAll.ColumnCount + 1;
                // 1 for level
                // search for lag length with smallest information criteria
                // Note: use the same number of observations to have comparable IC
                // aic and bic: smaller is better

                OLS.AutoLagResult autoLagResult = OLS.AutoLag(
                    xs: fullRHS,
                    ys: timeSeriesDShort,
                    (uint) startlag,
                    (uint) maxlag,
                    criterion: (OLS.AutoLagCriterion) autoLagCriterion
                );
                var bestlag = autoLagResult.bestlag - startlag;
            }

            throw new NotImplementedException();
        }
    }
}