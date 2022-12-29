namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.IntegralTransforms;

    public sealed class ACOVF
    {
        /// <summary>
        /// Find the next regular number greater than or equal to target.
        /// Regular numbers are composites of the prime factors 2, 3, and 5.
        /// Also known as 5-smooth numbers or Hamming numbers, these are the optimal
        /// size for inputs to FFTPACK.
        /// Target must be a positive integer.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/compat/scipy.py#L14
        /// </summary>
        private static int NextRegular(int target)
        {
            if (target <= 6)
                return target;

            // Quickly check if it's already a power of 2
            if ((target & (target - 1)) == 0)
                return target;

            var match = int.MaxValue;  // Anything found will be smaller
            var p5 = 1;
            while (p5 < target)
            {
                var p35 = p5;
                while (p35 < target)
                {
                    // Ceiling integer division, avoiding conversion to float
                    // (quotient = ceil(target / p35))
                    var quotient = (int) Math.Ceiling((double) target / p35);
                    // Quickly find next power of 2 >= quotient
                    var p2 = 1 << ((quotient - 1).Size());

                    var n_ = p2 * p35;
                    if (n_ == target)
                        return n_;
                    else if (n_ < match)
                        match = n_;
                    p35 *= 3;
                    if (p35 == target)
                        return p35;
                }
                if (p35 < match)
                    match = p35;
                p5 *= 5;
                if (p5 == target)
                    return p5;
            }
            if (p5 < match)
                match = p5;
            return match;
        }
        
        /// <summary>
        /// Estimate autocovariances.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L394
        /// </summary>
        public static double[] Run(
            double[] timeSeries,
            bool adjusted=false, // If True, then denominators is n-k, otherwise n.
            bool demean=true, // If True, then subtract the mean x from each element of x.
            uint? nlag=null
            // Limit the number of autocovariances returned.  Size of returned
            // array is nlag + 1.  Setting nlag when fft is False uses a simple,
            // direct estimator of the autocovariances that only computes the first
            // nlag + 1 values. This can be much faster when the time series is long
            // and only a small number of autocovariances are needed.
        )
        {
            var xo = timeSeries.AsVector();
            if (demean)
                xo -= xo.Average();
            var n = timeSeries.Length;
            var lagLen = nlag;
            if (nlag is null)
                lagLen = (uint)n - 1;
            else if (nlag > n - 1)
                throw new ArgumentException($"nlag must be smaller than nobs - 1");
            Vector<double> d;
            if (adjusted && lagLen is not null)
                d = Enumerable.Range(1, (int)lagLen + 1).Concat(Enumerable.Range(1, (int)lagLen).Reverse()).Select(value => value * 1.0).AsVector();
            else
                d = Enumerable.Repeat(n*1.0, n + n - 1).AsVector();
            var nobs = xo.Count();
            var m = NextRegular(2 * nobs + 1);
            var fourierOptions = FourierOptions.Matlab;
            var frf = xo.FFT(n: (uint) m, options: fourierOptions).AsVector();
            var ret = (
                (frf.PointwiseMultiply(frf.Conjugate())).ToArray().IFFT(options: fourierOptions)
                .Take(nobs).AsVector()
                .PointwiseDivide(d.Skip(nobs - 1).AsComplex32Vector())
                .Select(value => (double) value.Real)
            );
            if (nlag is not null && lagLen is not null)
                ret = ret.Take((int) lagLen + 1);
            return ret.ToArray();
        }
    }
}