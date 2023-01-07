namespace TimeSeriesStationaryUtils
{
    ///<summary>
    /// Season-Trend decomposition using LOESS.
    /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stl/_stl.pyx
    ///</summary>
    public sealed class STL
    {
        private readonly double[] timeSeries;
        private readonly uint nobs;
        private readonly uint period;
        private readonly uint seasonal;
        private readonly uint trend;
        private readonly uint lowPass;
        private readonly uint seasonalDeg;
        private readonly uint trendDeg;
        private readonly uint lowPassDeg;
        private readonly uint lowPassJump;
        private readonly uint trendJump;
        private readonly uint seasonalJump;
        private readonly bool robust;
        private          bool useRW;
        private readonly double[] trendArr;
        private readonly double[] seasonArr;
        private readonly double[] rw;
        private readonly double[,] work;

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
            this.work = new double[7, timeSeries.Length + (int) period + (int) period];
        }

        public struct DecomposeResult
        {
            double[] observed;
            double[] seasonal;
            double[] trend;
            double[] resid;
            double[] weights;

            public DecomposeResult(
                double[] observed,
                double[] seasonal,
                double[] trend,
                double[] resid,
                double[] weights
            )
            {
                this.observed = observed;
                this.seasonal = seasonal;
                this.trend = trend;
                this.resid = resid;
                this.weights = weights;
            }
        }

        public DecomposeResult Fit(
            uint? innerIter=null,
            // Number of iterations to perform in the inner loop. If not provided
            // uses 2 if ``robust`` is True, or 5 if not.
            uint? outerIter=null
            // Number of iterations to perform in the outer loop. If not provided
            // uses 15 if ``robust`` is True, or 0 if not.
        )
        {
            if (innerIter is null)
                innerIter = this.robust ? 2u : 5u;
            if (outerIter is null)
                outerIter = this.robust ? 15u : 0u;

            this.useRW = false;
            for (int i = 0; i < this.nobs; i += 1)
            {
                this.seasonArr[i] = this.trendArr[i] = 0.0;
                this.rw[i] = 1.0;
            }

            var k = 0;
            while (true)
            {
                this.oneStp((uint) innerIter);
                k += 1;
                if (k > outerIter)
                    break;
                for (int i = 0; i < this.nobs; i += 1)
                    this.work[0, i] = this.trendArr[i] + this.seasonArr[i];
                // this.rwts();
                this.useRW = true;
            }

            double[] seasonal = this.seasonArr.ToArray();
            double[] trend    = this.trendArr.ToArray();
            double[] rw       = this.rw.ToArray();
            double[] resid    = (
                this.timeSeries
                .Zip(seasonal.Zip(trend))
                .Select(values => values.First - values.Second.First - values.Second.Second)
                .ToArray()
            );
            return new DecomposeResult(
                observed: this.timeSeries,
                seasonal: seasonal,
                trend: trend,
                resid: resid,
                weights: rw
            );
        }

        private void oneStp(uint innerIter)
        {
            for (int j = 0; j < innerIter; j += 1)
            {
                for (int i = 0; i < this.nobs; i += 1)
                    this.work[0, i] = this.timeSeries[i] - this.trendArr[i];
                this.ss();
                // this.fts();
                // this.ess(work[2, :], n, nl, ildeg, nljump, False, work[3, :],
                //         work[0, :], work[4, :])
                // for i in range(self.nobs):
                //     season[i] = work[1, np+i] - work[0, i]
                //     work[0, i] = y[i] - season[i]
                // self._ess(work[0, :], n, nt, itdeg, ntjump, self._use_rw, rw,
                //         trend, work[2, :])
            }
            throw new NotImplementedException();
        }

        private double est<X, Y, Z>(
            SliceView<X, uint, double> y,
            uint n,
            uint len_,
            uint ideg,
            uint xs,
            uint nleft,
            uint nright,
            SliceView<Y, uint, double> w,
            bool useRW,
            SliceView<Z, uint, double> rw
        )
        {
            throw new NotImplementedException();
        }

        private void ess<X, Y, Z, W>(
            SliceView<X, uint, double> y,
            uint n,
            uint len_,
            uint ideg,
            uint njump,
            bool useRW,
            SliceView<Y, uint, double> rw,
            SliceView<Z, uint, double> ys,
            SliceView<W, uint, double> res
        )
        {
            if (n < 2)
            {
                ys[0] = y[0];
                return;
            }
            uint newnj = Math.Min(njump, n - 1);
            uint i, nleft = 0, nright = 0;
            if (len_ >= n)
            {
                nleft = 1;
                nright = n;
                i = 0;
                while (i < n)
                {
                    // formerly: for i in range(0, n, newnj):
                    ys[i] = this.est(
                        y: y,
                        n: n,
                        len_: len_,
                        ideg: ideg,
                        xs: i + 1,
                        nleft: nleft,
                        nright: nright,
                        w: res,
                        useRW: useRW,
                        rw: rw
                    );
                    if (double.IsNaN(ys[i]))
                        ys[i] = y[i];
                    i += newnj;
                }
            }
            else if (newnj == 1)
            {
                uint nsh = (len_ + 2) >> 1;
                nleft = 1;
                nright = len_;
                for (i = 0; i < n; i += 1)
                {
                    if ((i + 1) > nsh && nright != n)
                    {
                        nleft += 1;
                        nright += 1;
                    }                        
                    ys[i] = this.est(
                        y: y,
                        n: n,
                        len_: len_,
                        ideg: ideg,
                        xs: i + 1,
                        nleft: nleft,
                        nright: nright,
                        w: res,
                        useRW: useRW,
                        rw: rw
                    );
                    if (double.IsNaN(ys[i]))
                        ys[i] = y[i];
                }
            }
            else
            {
                uint nsh = (len_ + 1) >> 1;
                i = 0;
                while (i < n)
                {
                    // formerly: for i in range(0, n, newnj):
                    if ((i + 1) < nsh)
                    {
                        nleft = 1;
                        nright = len_;
                    }
                    else if ((i + 1) >= (n - nsh + 1))
                    {
                        nleft = n - len_ + 1;
                        nright = n;
                    }
                    else
                    {
                        nleft = i + 1 - nsh + 1;
                        nright = len_ + i + 1 - nsh;
                    }
                    ys[i] = this.est(
                        y: y,
                        n: n,
                        len_: len_,
                        ideg: ideg,
                        xs: i + 1,
                        nleft: nleft,
                        nright: nright,
                        w: res,
                        useRW: useRW,
                        rw: rw
                    );
                    if (double.IsNaN(ys[i]))
                        ys[i] = y[i];
                    i += newnj;
                }
            }
            // newnj > 1
            i = 0;
            while (i < (n - newnj))
            {
                // Formerly: for i in range(0, n - newnj, newnj):
                double delta = (ys[i + newnj] - ys[i]) / newnj;
                for (uint j = i; j < i + newnj; j += 1)
                    ys[j] = ys[i] + delta * ((j + 1) - (i + 1));
                i += newnj;
            }
            uint k = ((n - 1) / newnj) * newnj + 1;
            if (k != n)
            {
                ys[n - 1] = this.est(
                    y: y,
                    n: n,
                    len_: len_,
                    ideg: ideg,
                    xs: n,
                    nleft: nleft,
                    nright: nright,
                    w: res,
                    useRW: useRW,
                    rw: rw
                );
                if (double.IsNaN(ys[n - 1]))
                    ys[n - 1] = y[n - 1];
                if (k != (n - 1))
                {
                    double delta = (ys[n - 1] - ys[k - 1]) / (n - k);
                    for (uint j = k; j < n; j += 1)
                        ys[j] = ys[k - 1] + delta * ((j + 1) - k);
                }
            }
        }

        private void ss()
        {
            // Original variable names
            SliceView<double[,], uint, double> y         = this.work.AsSlice1D(rows: new uint[]{0});
            uint n                                       = this.nobs;
            uint np                                      = this.period;
            uint ns                                      = this.seasonal;
            uint isdeg                                   = this.seasonalDeg;
            uint nsjump                                  = this.seasonalJump;
            SliceView<double[], uint, double> rw         = this.rw.AsSlice1D();
            SliceView<double[,], uint, double> seasonArr = this.work.AsSlice1D(rows: new uint[]{1});
            SliceView<double[,], uint, double> work1     = this.work.AsSlice1D(rows: new uint[]{2});
            SliceView<double[,], uint, double> work2     = this.work.AsSlice1D(rows: new uint[]{3});
            SliceView<double[,], uint, double> work3     = this.work.AsSlice1D(rows: new uint[]{4});
            SliceView<double[], uint, double> work4      = this.seasonArr.AsSlice1D();
            bool useRW                   = this.useRW;
            for(int j = 0; j < np; j += 1)
            {
                uint k = (uint) (n - (j + 1)); // np + 1
                for (uint i = 0; i < k; i += 1)
                    work1[i] = y[(uint) (i * np + j)];
                if (useRW)
                {
                    for (uint i = 0; i < k; i += 1)
                        work3[i] = rw[(uint) (i * np + j)];
                }

                this.ess(
                    y: work1,
                    n: k,
                    len_: ns,
                    ideg: isdeg,
                    njump: nsjump,
                    useRW: useRW,
                    rw: work3,
                    ys: work2.AsSlice1D(indexes: new Slice(start: 1)),
                    res: work4
                );
                uint xs = 0;
                uint nright = Math.Min(ns, k);
                work2[0] = this.est(
                    y: work1,
                    n: k,
                    len_: ns,
                    ideg: isdeg,
                    xs: xs,
                    nleft: 1,
                    nright: nright,
                    w: work4,
                    useRW: useRW,
                    rw: work3
                );
                if (double.IsNaN(work2[0]))
                    work2[0] = work2[1];
                xs = k + 1;
                uint nleft = Math.Max(1, k - ns + 1);
                work2[k + 1] = this.est(
                    y: work1,
                    n: k,
                    len_: ns,
                    ideg: isdeg,
                    xs: xs,
                    nleft: nleft,
                    nright: k,
                    w: work4,
                    useRW: useRW,
                    rw: work3
                );
                if (double.IsNaN(work2[k + 1]))
                    work2[k + 1] = work2[k];
                for(uint m = 0; m < k + 2; m += 1)
                    seasonArr[(uint) (m * np + j)] = work2[m];
            }
        }
    }
}