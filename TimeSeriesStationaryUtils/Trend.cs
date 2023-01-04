namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics.LinearAlgebra;

    public static class Trend
    {
        public enum Regression
        {
            N,  // "n" : no constant, no trend.
            C,  // "c" : constant only.
            T,  // "t": trend only.
            CT, // "ct" : constant and trend.
            CTT // "ctt" : constant, and linear and quadratic trend.
        }

        public static uint ntrend(this Regression that)
        {
            if (that == Regression.N)
                return 0;
            else if (that == Regression.C || that == Regression.T)
                return 1;
            else if (that == Regression.CT)
                return 2;
            else // Regression.CTT
                return 3;
        }

        /// <summary>
        /// Add a trend and/or constant to a matrix.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/tsatools.py#L38
        /// </summary>
        public static Matrix<double> AddTrend(
            this Matrix<double> that,
            Regression trend=Regression.C,
            bool prepend=false
        )
        {
            uint trendOrder;
            if (trend == Regression.N)
                return that;
            else if (trend == Regression.C)
                trendOrder = 0;
            else if (trend == Regression.CT || trend == Regression.T)
                trendOrder = 1;
            else //if (trend == Trend.CTT)
                trendOrder = 2;
            
            uint nobs = (uint) that.RowCount;
            var trendArr = (
                Enumerable.Range(1, (int) nobs)
                    .Select(value => value * 1.0)
                    .Vandermonde(n: trendOrder + 1, increasing: false)
                    // put in order ctt
                    .Select(values => values.Reverse())
                    .AsMatrix()
            );
            if (trend == Regression.T)
                trendArr = trendArr.SubMatrix(
                    rowIndex: 0,
                    rowCount: trendArr.RowCount,
                    columnIndex: 1,
                    columnCount: 1
                );
            return prepend ? trendArr.Append(that) : that.Append(trendArr);
        }
    }
}