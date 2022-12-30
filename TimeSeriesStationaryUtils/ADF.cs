namespace TimeSeriesStationaryUtils
{
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

        public enum Regression
        {
            N,   // "n" : no constant, no trend.
            C,   // "c" : constant only (default).
            CT,  // "ct" : constant and trend.
            CTT  // "ctt" : constant, and linear and quadratic trend.
        }

        public enum Autolag
        {
            AIC,
            BIC,
            TStat
        }

        /// <summary>
        /// Augmented Dickey-Fuller unit root test.
        /// The Augmented Dickey-Fuller test can be used to test for a unit root in a
        /// univariate process in the presence of serial correlation.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L166
        ///</summary>
        public static void Run(
            double[] timeSeries,
            int? maxlag=null, // Maximum lag which is included in test, default value of 12*(nobs/100)^{1/4} is used when ``null``.
            Regression regression=Regression.C,
            // Constant and trend order to include in regression.
            // * "c" : constant only (default).
            // * "ct" : constant and trend.
            // * "ctt" : constant, and linear and quadratic trend.
            // * "n" : no constant, no trend.
            Autolag? autolag=Autolag.AIC,
            //  Method to use when automatically determining the lag length among the
            // values 0, 1, ..., maxlag.
            // * If "AIC" (default) or "BIC", then the number of lags is chosen
            // to minimize the corresponding information criterion.
            // * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
            // lag until the t-statistic on the last lag length is significant
            // using a 5%-sized test.
            // * If None, then the number of included lags is set to maxlag.
            bool regresults=false
            // If True, the full regression results are returned. Default is False.
        )
        {
            if (timeSeries.Min() == timeSeries.Max())
                throw new ArgumentException("Invalid input, timeSeries is constant");
            var nobs = (uint) timeSeries.Length;
            var ntrend = regression.ntrend();
            if (maxlag is null)
            {
                // from Greene referencing Schwert 1989
                maxlag = (int)(Math.Ceiling(12.0 * Math.Pow(nobs / 100.0, 1 / 4.0)));
                // -1 for the diff
                maxlag = Math.Min((int) nobs / 2 - (int) ntrend - 1, (int) maxlag);
                if (maxlag < 0)
                    throw new NotSupportedException(@"sample size is too short to use selected regression component");
            }
            else if (maxlag > (nobs / 2 - ntrend - 1))
                throw new ArgumentException(@"maxlag must be less than (nobs/2 - 1 - ntrend) where ntrend is the number of included deterministic regressors");
            var timeSeriesDiff = timeSeries.Diff().AsVector();
            var timeSeriesDAll = (
                timeSeriesDiff.ToColumnMatrix()
                .LagMat((uint) maxlag)
                .LaggedValues(trim: Extensions.Trim.Both) // original="ex" // only the lagged matrix with trim="both"
            );
            nobs = (uint) timeSeriesDAll.RowCount;
            throw new NotImplementedException();
        }
    }
}