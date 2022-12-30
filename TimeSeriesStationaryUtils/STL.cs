namespace TimeSeriesStationaryUtils
{
    ///<summary>
    /// Season-Trend decomposition using LOESS.
    /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stl/_stl.pyx
    ///</summary>
    public sealed class STL
    {
        readonly public double[] timeSeries;
        readonly public uint nobs;
        readonly public uint period;
        readonly public uint seasonal;
        readonly public uint trend;
        readonly public uint lowPass;
        readonly public uint seasonalDeg;
        readonly public uint trendDeg;
        readonly public uint lowPassDeg;
        readonly public uint lowPassJump;
        readonly public uint trendJump;
        readonly public uint seasonalJump;
        readonly public bool robust;
        readonly private bool useRW;
        readonly private double[] trendArr;
        readonly private double[] seasonArr;
        readonly private double[] rw;
        readonly private double[][] work;

        public STL(
            double[] timeSeries,
            uint period,
            // Periodicity of the sequence.
            uint seasonal=7,
            // Length of the seasonal smoother. Must be an odd integer, and should
            // normally be >= 7 (default).
            uint? trend=null,
            // Length of the trend smoother. Must be an odd integer. If not provided
            // uses the smallest odd integer greater than
            // 1.5 * period / (1 - 1.5 / seasonal), following the suggestion in
            // the original implementation.
            uint? lowPass=null,
            // Length of the low-pass filter. Must be an odd integer >=3. If not
            // provided, uses the smallest odd integer > period.
            uint seasonalDeg=1,
            // Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
            uint trendDeg=1,
            // Degree of trend LOESS. 0 (constant) or 1 (constant and trend).
            uint lowPassDeg=1,
            // Degree of low pass LOESS. 0 (constant) or 1 (constant and trend).
            bool robust=false,
            // Flag indicating whether to use a weighted version that is robust to
            // some forms of outliers.
            uint seasonalJump=1,
            // Positive integer determining the linear interpolation step. If larger
            // than 1, the LOESS is used every seasonalJump points and linear
            // interpolation is between fitted points. Higher values reduce
            // estimation time.
            uint trendJump=1,
            // Positive integer determining the linear interpolation step. If larger
            // than 1, the LOESS is used every trendJump points and values between
            // the two are linearly interpolated. Higher values reduce estimation
            // time.
            uint lowPassJump=1
            // Positive integer determining the linear interpolation step. If larger
            // than 1, the LOESS is used every lowPassJump points and values between
            // the two are linearly interpolated. Higher values reduce estimation
            // time.
        )
        {
            this.timeSeries = timeSeries;
            this.nobs = (uint) timeSeries.Length;

            if (period < 2)
                throw new ArgumentOutOfRangeException("period must be a positive integer >= 2");
            this.period = period;
            
            
            if (((seasonal & 1) == 0) || (seasonal < 3))
                throw new ArgumentException("seasonal must be an odd positive integer >= 3");
            this.seasonal = seasonal;
            
            if (trend is null)
            {
                trend = (uint) (Math.Ceiling(1.5 * period / (1 - 1.5 / seasonal)));
                // ensure odd
                trend += ~(trend & 1);
            }
            if (((trend & 1) == 0) || (trend < 3) || (trend <= period))
                throw new ArgumentException("trend must be an odd positive integer >= 3 where trend > period");
            this.trend = (uint) trend;

            if (lowPass is null)
            {
                lowPass = period + 1;
                lowPass += ~(lowPass & 1);
            }
            if (((lowPass & 1) == 0) || (lowPass < 3) || (lowPass <= period))
                throw new ArgumentException("lowPass must be an odd positive integer >= 3 where trend > period");
            this.lowPass = (uint) lowPass;

            this.seasonalDeg = seasonalDeg;
            this.trendDeg = trendDeg;
            this.lowPassDeg = 1;
            this.robust = robust;
            this.seasonalJump = seasonalJump;
            this.trendJump = trendJump;
            this.lowPassJump = lowPassJump;
            
            // default internal parameters
            this.useRW = false;
            this.trendArr = Enumerable.Repeat(0.0, timeSeries.Length).ToArray();
            this.seasonArr = Enumerable.Repeat(0.0, timeSeries.Length).ToArray();
            this.rw = Enumerable.Repeat(1.0, timeSeries.Length).ToArray();
            this.work = (
                Enumerable.Repeat(0, 7)
                .Select(_ => Enumerable.Repeat(0.0, timeSeries.Length + (int) period + (int) period).ToArray())
                .ToArray()
            );

            throw new NotImplementedException();
        }
    }
}