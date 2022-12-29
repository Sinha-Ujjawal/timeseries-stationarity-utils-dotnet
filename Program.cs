using System.Diagnostics;
using TimeSeriesStationaryUtils;

namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics;
    using MathNet.Numerics.LinearRegression;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.IntegralTransforms;
    using MathNet.Numerics.Distributions;

    public static class Extensions
    {
        public static IEnumerable<U> Scan<T, U>(this IEnumerable<T> input, Func<U, T, U> next, U state)
        {
            yield return state;
            foreach (var item in input)
            {
                state = next(state, item);
                yield return state;
            }
        }

        public static double DotProduct(this IEnumerable<double> sequence, IEnumerable<double> other)
        {
            return sequence.Zip(other).Select((values, _) => values.First * values.Second).Sum();
        }

        public static int Size(this int bits)
        {
            int size = 0;

            for (; bits != 0; bits >>= 1)
                size++;

            return size;
        }

        public static Complex32[] IFFT(
            this Complex32[] spectrum,
            FourierOptions options = FourierOptions.Default
        )
        {
            var spectrumCopy = spectrum.ToArray();
            Fourier.Inverse(spectrumCopy, options: options);
            return spectrumCopy;
        }

        public static Complex32[] FFT(
            this Vector<double> vec,
            FourierOptions options = FourierOptions.Default,
            uint? n=null
        )
        {
            if (n is not null)
                vec = vec.EnsureLength(n: (uint)n, padWith: 0.0).AsVector();
            var samples = vec.Select(value => new Complex32(real: (float) value, imaginary: 0.0f)).ToArray();
            Fourier.Forward(samples, options: options);
            return samples;
        }

        public static Complex32[] FFT(
            this Vector<int> vec,
            FourierOptions options = FourierOptions.Default,
            uint? n=null
        )
        {
            return vec.Select(value => value * 1.0).AsVector().FFT();
        }

        public static IEnumerable<T> EnsureLength<T>(this IEnumerable<T> sequence, uint n, T padWith)
        {
            return sequence.Concat(Enumerable.Repeat(padWith, (int) n)).Take((int) n);
        }

        public static Vector<double> EnsureLength<T>(this Vector<double> vec, uint n, double padWith)
        {
            return vec.AsEnumerable().EnsureLength(n: n, padWith: padWith).AsVector();
        }

        public static Vector<Complex32> EnsureLength<T>(this Vector<Complex32> vec, uint n, Complex32 padWith)
        {
            return vec.AsEnumerable().EnsureLength(n: n, padWith: padWith).AsVector();
        }

        public static IEnumerable<double> RunningSum(this IEnumerable<double> sequence)
        {
            return sequence.Scan((prev, next) => prev + next, 0.0).Skip(1);
        }

        public static double SquaredSum(this IEnumerable<double> sequence)
        {
            return sequence.Select(value => value * value).Sum();
        }

        public static Vector<double> RunningSum(this Vector<double> vec)
        {
            return vec.AsEnumerable().RunningSum().AsVector();
        }

        public static double SquaredSum(this Vector<double> vec)
        {
            return vec.AsEnumerable().SquaredSum();
        }

        public static double RunningSquaredSum(this Vector<double> vec)
        {
            return vec.AsEnumerable().RunningSum().SquaredSum();
        }

        public static IEnumerable<double> DeMean(this IEnumerable<double> sequence)
        {
            double mean = sequence.Average();
            return sequence.Select(value => value - mean);
        }

        public static Vector<Complex32> AsComplex32Vector(this IEnumerable<double> sequence)
        {
            return sequence.Select(value => new Complex32(real: (float) value, imaginary: 0.0f)).AsVector();
        }

        public static Vector<double> AsVector(this IEnumerable<double> sequence)
        {
            return Vector<double>.Build.DenseOfEnumerable(sequence);
        }

        public static Vector<double> AsVector(this IEnumerable<int> sequence)
        {
            return Vector<double>.Build.DenseOfEnumerable(sequence.Select(value => value * 1.0));
        }

        public static Vector<Complex32> AsVector(this IEnumerable<Complex32> sequence)
        {
            return Vector<Complex32>.Build.DenseOfEnumerable(sequence);
        }

        public static double Truncate(this double val, uint ndigits)
        {
            var _TenPows = Math.Pow(10, ndigits);
            return Math.Truncate(val * _TenPows) / _TenPows;
        }
    }


    public sealed class Algorithm
    {
        /// <summary>
        /// Computes the number of lags for covariance matrix estimation in KPSS test
        /// using method of Hobijn et al (1998). See also Andrews (1991), Newey & West
        /// (1994), and Schwert (1989). Assumes Bartlett / Newey-West kernel.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L2094
        /// </summary>
        private static int KPSSAutoLag(Vector<double> resids, int nobs)
        {
            var covlags = (int)Math.Pow(nobs, 2.0 / 9.0);
            var s0 = resids.SquaredSum() / nobs;
            var s1 = 0.0;
            var residsLength = resids.Count;
            foreach (int i in Enumerable.Range(start: 1, count: covlags))
            {
                var residsProd = resids.SubVector(i, residsLength - i).DotProduct(resids.SubVector(0, nobs - i));
                residsProd = residsProd / (nobs / 2.0);
                s0 += residsProd;
                s1 += ((double)i) * residsProd;
            }
            var sHat = s1 / s0;
            var pwr = 1.0 / 3.0;
            var gammaHat = 1.1447 * Math.Pow(sHat * sHat, pwr);
            var autolags = (int)(gammaHat * Math.Pow(nobs, pwr));
            return autolags;
        }

        /// <summary>
        /// Computes equation 10, p. 164 of Kwiatkowski et al. (1992). This is the
        /// consistent estimator for the variance.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L2082
        /// </summary>
        private static double SigmaEstKPSS(Vector<double> resids, int nobs, int nlags)
        {
            var sHat = resids.SquaredSum();
            var residsLength = resids.Count;
            foreach (int i in Enumerable.Range(start: 1, count: nlags))
            {
                var residsProd = resids.SubVector(i, residsLength - i).DotProduct(resids.SubVector(0, nobs - i));
                sHat += 2 * residsProd * (1.0 - (i / (nlags + 1.0)));
            }
            return sHat / nobs;
        }

        public struct KPSSTestStatistic
        {
            readonly public double kpssStat;
            readonly public double pValue;
            readonly public uint nlags;
            readonly public double crit10Percent;
            readonly public double crit5Percent;
            readonly public double crit2Point5Percent;
            readonly public double crit1Percent;

            public KPSSTestStatistic(
                 double kpssStat,
                 double pValue,
                 uint nlags,
                 double crit10Percent,
                 double crit5Percent,
                 double crit2Point5Percent,
                 double crit1Percent
             )
            {
                this.kpssStat = kpssStat;
                this.pValue = pValue;
                this.nlags = nlags;
                this.crit10Percent = crit10Percent;
                this.crit5Percent = crit5Percent;
                this.crit2Point5Percent = crit2Point5Percent;
                this.crit1Percent = crit1Percent;
            }

            public override string ToString()
            {
                var sb = new StringWriter();
                sb.WriteLine($"Test Statistic:\t{kpssStat}");
                sb.WriteLine($"p-value:\t{pValue}");
                sb.WriteLine($"Lags User:\t{nlags}");
                sb.WriteLine($"Critical Value (10%):\t{crit10Percent}");
                sb.WriteLine($"Critical Value (5%):\t{crit5Percent}");
                sb.WriteLine($"Critical Value (2.5%):\t{crit2Point5Percent}");
                sb.WriteLine($"Critical Value (1%):\t{crit1Percent}");
                return sb.ToString();
            }
        }

        /// <summary>
        /// Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
        /// Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null
        /// hypothesis that x is level or trend stationary.
        /// Taken from python Statsmodels: https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L1912
        /// </summary>
        public static KPSSTestStatistic KPSS(
            double[] timeSeries, // time series array
            bool isStationaryAroundTrend = false, // by default, Is Stationary Around Constant would be used
            int nlags = -1 // <0: Auto, 0: Legacy, >0: number of lags used
        )
        {
            int nobs = timeSeries.Length;
            Vector<double> resids;
            double[] crit;
            if (!isStationaryAroundTrend)
            { // Stationary around constant
                resids = timeSeries.DeMean().AsVector();
                crit = new double[4] { 0.347, 0.463, 0.574, 0.739 };
            }
            else
            {
                double[] xs = Enumerable.Range(1, nobs).Select(value => value * 1.0).ToArray<double>();
                var (intercept, slope) = SimpleRegression.Fit(
                    x: xs,
                    y: timeSeries
                );
                IEnumerable<double> timeSeriesHat = xs.Select((x, _) => slope * x + intercept);
                resids = timeSeriesHat.Zip(timeSeries).Select((ys, _) => ys.Second - ys.First).AsVector();
                crit = new double[4] { 0.119, 0.146, 0.176, 0.216 };
            }
            if (nlags == 0)
            { // Legacy
                nlags = (int)Math.Ceiling(12.0 * Math.Pow(nobs / 100.0, 1 / 4.0));
            }
            else if (nlags < 0)
            { // Auto
                nlags = KPSSAutoLag(resids, nobs);
                nlags = Math.Min(nlags, nobs - 1);
            }
            else if (nlags >= nobs)
            {
                throw new ArgumentException($"lags ({nlags}) must be < number of observations ({nobs})");
            }
            double[] pvals = { 0.10, 0.05, 0.025, 0.01 };
            var eta = resids.RunningSquaredSum() / (nobs * nobs);
            var sHat = SigmaEstKPSS(resids, nobs, nlags);
            var kpssStat = eta / sHat;
            var pValue = Interpolate.Linear(points: crit, values: pvals).Interpolate(kpssStat);
            if (pValue.AlmostEqual(pvals.Last(), 1e-3))
            {
                Console.Error.WriteLine(@"The test statistic is outside of the range of p-values available in the
look-up table. The actual p-value is smaller than the p-value returned.");
            }
            else if (pValue.AlmostEqual(pvals[0], 1e-3))
            {
                Console.Error.WriteLine(@"The test statistic is outside of the range of p-values available in the
look-up table. The actual p-value is greater than the p-value returned.");
            }
            return new KPSSTestStatistic(
                kpssStat,
                pValue,
                (uint)nlags,
                crit10Percent: crit[0],
                crit5Percent: crit[1],
                crit2Point5Percent: crit[2],
                crit1Percent: crit[3]
            );
        }

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
        public static double[] ACOVF(
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
        public static QStatistics QStat(double[] timeSeries, uint nobs)
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

        public struct ACFStatistic
        {
            readonly public double[] acf; // The autocorrelation function for lags 0, 1, ..., nlags. Shape (nlags+1,).
            readonly public (double First, double Second)[]? confInt; // Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape (nlags + 1, 2). Returned if alpha is not null.
            readonly public QStatistics? qStatistics;

            public ACFStatistic(
                double[] acf,
                (double First, double Second)[]? confInt = null,
                QStatistics? qStatistics = null
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
        public static ACFStatistic ACF(
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
            var avf = ACOVF(timeSeries, adjusted: adjusted, demean: true);      
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
            var qStatistics = QStat(acf.Skip(1).ToArray(), nobs: nobs);  // drop lag 0
            if (alpha is not null)
                return new ACFStatistic(acf: acf.ToArray(), confInt: confInt, qStatistics: qStatistics);
            else
                return new ACFStatistic(acf: acf.ToArray(), qStatistics: qStatistics);
        }
    }
}

class Tests
{
    double[] sunActivities;

    private void AlmostEqual(double x, double y, double err = 1e-3)
    {
        Debug.Assert(
            Math.Abs(x - y) <= err,
            message: $"ABS([Left Value] {x} - [Right Value] {y}) > {err}"
        );
    }

    private void AlmostEqual(IEnumerable<double> xs, IEnumerable<double> ys, double err = 1e-3)
    {
        Debug.Assert(
            xs.Count() == ys.Count(),
            message: $"Length of both arrays must be same. Length(Left = {xs.Count()}) != Length(Right = {ys.Count()})"
        );
        Debug.Assert(
            xs.Zip(ys).Select(values => Math.Abs(values.First - values.Second) <= err).All(b => b),
            message: $"ABS([Left Values] {xs} - [Right Values] {ys}) > {err}"
        );
    }

    private void AlmostEqual(IEnumerable<(double First, double Second)> xs, IEnumerable<(double First, double Second)> ys, double err = 1e-3)
    {
        Debug.Assert(
            xs.Count() == ys.Count(),
            message: $"Length of both arrays must be same. Length(Left = {xs.Count()}) != Length(Right = {ys.Count()})"
        );
        Debug.Assert(
            (
                xs.Zip(ys)
                .Select(values => (
                    (Math.Abs(values.First.First - values.Second.First) <= err)
                    && (Math.Abs(values.First.Second - values.Second.Second) <= err)
                ))
                .All(b => b)
            ),
            message: $"ABS([Left Values] {xs} - [Right Values] {ys}) > {err}"
        );
    }

    private void Equal<T>(T x, T y) where T : notnull
    {
        Debug.Assert(
            x.Equals(y),
            message: $"[Left Value] {x} != [Right Value] {y}"
        );
    }

    public Tests()
    {
        this.sunActivities = LoadSunActivitiesTimeSeries();
    }

    private static double[] LoadSunActivitiesTimeSeries()
    {
        List<double> sunActivities = new List<double>();
        using (var reader = new StreamReader(@"./testing_datasets/SunActivities.csv"))
        {
            if (!reader.EndOfStream)
            {
                // Skip the first line
                reader.ReadLine();
            }
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (line is not null)
                {
                    var values = line.Split(",");
                    sunActivities.Add(Convert.ToDouble(values.Last()));
                }
                else
                {
                    break;
                }
            }
        }
        return sunActivities.ToArray();
    }

    public void testKPSSDataStationaryAroundConstant()
    {
        var kpssTestStatistic = Algorithm.KPSS(
            timeSeries: sunActivities,
            isStationaryAroundTrend: false
        );
        var kpssTestStatisticExpected = new Algorithm.KPSSTestStatistic(
            kpssStat: 0.6698,
            pValue: 0.016,
            nlags: 7,
            crit10Percent: 0.347,
            crit5Percent: 0.463,
            crit2Point5Percent: 0.574,
            crit1Percent: 0.739
        );
        AlmostEqual(kpssTestStatistic.kpssStat, kpssTestStatisticExpected.kpssStat);
        AlmostEqual(kpssTestStatistic.pValue, kpssTestStatisticExpected.pValue);
        Equal(kpssTestStatistic.nlags, kpssTestStatisticExpected.nlags);
        AlmostEqual(kpssTestStatistic.crit10Percent, kpssTestStatisticExpected.crit10Percent);
        AlmostEqual(kpssTestStatistic.crit5Percent, kpssTestStatisticExpected.crit5Percent);
        AlmostEqual(kpssTestStatistic.crit2Point5Percent, kpssTestStatisticExpected.crit2Point5Percent);
        AlmostEqual(kpssTestStatistic.crit1Percent, kpssTestStatisticExpected.crit1Percent);
    }

    public void testKPSSDataStationaryAroundTrend()
    {
        var kpssTestStatistic = Algorithm.KPSS(
            timeSeries: sunActivities,
            isStationaryAroundTrend: true
        );
        var kpssTestStatisticExpected = new Algorithm.KPSSTestStatistic(
            kpssStat: 0.1158,
            pValue: 0.105,
            nlags: 7,
            crit10Percent: 0.119,
            crit5Percent: 0.146,
            crit2Point5Percent: 0.176,
            crit1Percent: 0.216
        );
        AlmostEqual(kpssTestStatistic.kpssStat, kpssTestStatisticExpected.kpssStat);
        AlmostEqual(kpssTestStatistic.pValue, kpssTestStatisticExpected.pValue);
        Equal(kpssTestStatistic.nlags, kpssTestStatisticExpected.nlags);
        AlmostEqual(kpssTestStatistic.crit10Percent, kpssTestStatisticExpected.crit10Percent);
        AlmostEqual(kpssTestStatistic.crit5Percent, kpssTestStatisticExpected.crit5Percent);
        AlmostEqual(kpssTestStatistic.crit2Point5Percent, kpssTestStatisticExpected.crit2Point5Percent);
        AlmostEqual(kpssTestStatistic.crit1Percent, kpssTestStatisticExpected.crit1Percent);
    }

    public void testingKPSS()
    {
        Console.WriteLine("Testing KPSS");
        Console.WriteLine("  Testing KPSS Stationary Around Constant");
        this.testKPSSDataStationaryAroundConstant();
        Console.WriteLine("  Testing KPSS Stationary Around Trend");
        this.testKPSSDataStationaryAroundTrend();
    }

    public void testingACFDefault()
    {
        var acfTestStatistic = Algorithm.ACF(timeSeries: sunActivities);
        var acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518,
                0.1582025356911707,
                0.4730975308980598,
                0.6589800155363379,
                0.6502908198407041,
                0.4566625437895419,
                0.16179329478317217,
                -0.12205104904067786,
                -0.3161807966260729,
                -0.3747112537274217,
                -0.3060575265823909,
                -0.13480689540510293,
                0.09158727406274836,
                0.29756319807027737,
                0.42070739821101444,
                0.411839537144707,
                0.270207575926828,
                0.04496207925856511
            }
        );
        AlmostEqual(acfTestStatistic.acf.AsVector(), acfTestStatisticExpected.acf.AsVector());
    }

    public void testingACFWithNLags()
    {
        var acfTestStatistic = Algorithm.ACF(
            timeSeries: sunActivities,
            nlags: 10
        );
        var acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518,
                0.1582025356911707,
                0.4730975308980598,
                0.6589800155363379
            }
        );
        AlmostEqual(acfTestStatistic.acf.AsVector(), acfTestStatisticExpected.acf.AsVector());
        acfTestStatistic = Algorithm.ACF(
            timeSeries: sunActivities,
            nlags: 7
        );
        acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518
            }
        );
    }

    public void testingACFWithAlphaPoint05Bartlett()
    {
        var acfTestStatistic = Algorithm.ACF(
            timeSeries: sunActivities,
            alpha: 0.05
        );
        var acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518,
                0.1582025356911707,
                0.4730975308980598,
                0.6589800155363379,
                0.6502908198407041,
                0.4566625437895419,
                0.16179329478317217,
                -0.12205104904067786,
                -0.3161807966260729,
                -0.3747112537274217,
                -0.3060575265823909,
                -0.13480689540510293,
                0.09158727406274836,
                0.29756319807027737,
                0.42070739821101444,
                0.411839537144707,
                0.270207575926828,
                0.04496207925856511
            },
            confInt: new (double First, double Second)[] {
                (1.0, 1.0),
                (0.7087028389661938, 0.9316997498738511),
                (0.2805097695475459, 0.6220272144715895),
                (-0.14541503918715815, 0.22456814232779493),
                (-0.46088878152182444, -0.09069514071337892),
                (-0.6153762352043454, -0.23510262644320415),
                (-0.578208808998814, -0.17498137004930842),
                (-0.36755091865962286, 0.05280309208071923),
                (-0.05343433343348378, 0.3698394048158252),
                (0.2599955438361258, 0.6861995179599938),
                (0.4331980260674155, 0.8847620050052603),
                (0.4017455872531289, 0.8988360524282794),
                (0.18779622922628442, 0.7255288583527995),
                (-0.11654860138308148, 0.4401351909494258),
                (-0.4015596766211478, 0.15745757853979203),
                (-0.5963512012452317, -0.036010392006914194),
                (-0.6592830341052225, -0.09013947334962086),
                (-0.5966985236938991, -0.015416529470882678),
                (-0.4294273503397586, 0.15981355952955273),
                (-0.20379901538831424, 0.38697356351381096),
                (0.0018240844387089061, 0.5933023117018459),
                (0.12126932270030488, 0.720145473721724),
                (0.1051411116036789, 0.7185379626857351),
                (-0.043290621993795575, 0.5837057738474516),
                (-0.2714181982749296, 0.3613423567920598)
            }
        );
        AlmostEqual(acfTestStatistic.acf.AsVector(), acfTestStatisticExpected.acf.AsVector());
        Debug.Assert(
            acfTestStatistic.confInt is not null,
            message: $"Confint is null, expected some value"
        );
        if (acfTestStatisticExpected.confInt is null)
            throw new Exception("Unreachable!");
        AlmostEqual(acfTestStatistic.confInt, acfTestStatisticExpected.confInt);
    }

    public void testingACFWithAlphaPoint05NonBartlett()
    {
        var acfTestStatistic = Algorithm.ACF(
            timeSeries: sunActivities,
            alpha: 0.05,
            bartlettConfInt: false
        );
        var acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518,
                0.1582025356911707,
                0.4730975308980598,
                0.6589800155363379,
                0.6502908198407041,
                0.4566625437895419,
                0.16179329478317217,
                -0.12205104904067786,
                -0.3161807966260729,
                -0.3747112537274217,
                -0.3060575265823909,
                -0.13480689540510293,
                0.09158727406274836,
                0.29756319807027737,
                0.42070739821101444,
                0.411839537144707,
                0.270207575926828,
                0.04496207925856511
            },
            confInt: new (double First, double Second)[] {
                (0.8885015445461714, 1.1114984554538285),
                (0.7087028389661938, 0.9316997498738511),
                (0.3397700365557391, 0.5627669474633963),
                (-0.07192190388351023, 0.151075007024147),
                (-0.38729041657143026, -0.16429350566377307),
                (-0.5367378862776033, -0.31374097536994616),
                (-0.4880935449778898, -0.26509663407023254),
                (-0.2688723687432804, -0.04587545783562319),
                (0.0467040802373421, 0.26970099114499935),
                (0.3615990754442312, 0.5845959863518884),
                (0.5474815600825093, 0.7704784709901665),
                (0.5387923643868755, 0.7617892752945328),
                (0.3451640883357133, 0.5681609992433705),
                (0.05029483932934356, 0.2732917502370008),
                (-0.23354950449450645, -0.010552593586849249),
                (-0.42767925207990154, -0.2046823411722443),
                (-0.4862097091812503, -0.2632127982735931),
                (-0.41755598203621946, -0.19455907112856227),
                (-0.24630535085893154, -0.023308439951274323),
                (-0.01991118139108025, 0.20308572951657697),
                (0.18606474261644876, 0.40906165352410595),
                (0.3092089427571858, 0.5322058536648431),
                (0.30034108169087836, 0.5233379925985356),
                (0.15870912047299937, 0.3817060313806566),
                (-0.06653637619526351, 0.1564605347123937)
            }
        );
        AlmostEqual(acfTestStatistic.acf.AsVector(), acfTestStatisticExpected.acf.AsVector());
        Debug.Assert(
            acfTestStatistic.confInt is not null,
            message: $"confInt is null, expected some value"
        );
        if (acfTestStatisticExpected.confInt is null)
            throw new Exception("Unreachable!");
        AlmostEqual(acfTestStatistic.confInt, acfTestStatisticExpected.confInt);
    }

    public void testingACFWithAlphaPointQStat()
    {
        var acfTestStatistic = Algorithm.ACF(timeSeries: sunActivities, qstat: true);
        var acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518,
                0.1582025356911707,
                0.4730975308980598,
                0.6589800155363379,
                0.6502908198407041,
                0.4566625437895419,
                0.16179329478317217,
                -0.12205104904067786,
                -0.3161807966260729,
                -0.3747112537274217,
                -0.3060575265823909,
                -0.13480689540510293,
                0.09158727406274836,
                0.29756319807027737,
                0.42070739821101444,
                0.411839537144707,
                0.270207575926828,
                0.04496207925856511
            },
            qStatistics: new Algorithm.QStatistics(
                qStats: new double[] {
                    209.89836353742976,
                    273.64400804059835,
                    274.1359040985167,
                    298.1011690749651,
                    355.26381738878973,
                    400.2444486159611,
                    408.12537759734283,
                    416.1159750621361,
                    487.8126436798599,
                    627.3826726281839,
                    763.7523617970419,
                    831.2289634860766,
                    839.7275792721771,
                    844.580239517275,
                    877.2572649941754,
                    923.3088647080358,
                    954.1366372053316,
                    960.1380029293418,
                    962.9176594693135,
                    992.3604833413744,
                    1051.4195635777849,
                    1108.2123328696512,
                    1132.7451757848503,
                    1133.4268341711972
                },
                pvalues: new double[] {
                    1.445572992738202e-47,
                    3.7927887230279043e-60,
                    3.9322711091079876e-59,
                    2.7822189384881116e-63,
                    1.2875072901819138e-74,
                    2.4769836759153463e-83,
                    4.313800919310703e-84,
                    6.671800474590689e-85,
                    2.3373183200233053e-99,
                    2.3819790998963243e-128,
                    1.1434340014732639e-156,
                    3.314665335174312e-170,
                    4.25659815419771e-171,
                    3.193479306947175e-171,
                    2.5940105209507806e-177,
                    2.8869346431222513e-186,
                    5.7661916029977535e-192,
                    2.2954782237649135e-192,
                    4.344563928164502e-192,
                    1.6618222792482172e-197,
                    3.074666640781376e-209,
                    1.7330498275991565e-220,
                    7.373439199353681e-225,
                    3.748263065537049e-224
                }
            )
        );
        AlmostEqual(acfTestStatistic.acf.AsVector(), acfTestStatisticExpected.acf.AsVector());
        Debug.Assert(
            acfTestStatistic.qStatistics is not null,
            message: $"qStatistics is null, expected some value"
        );
        if (acfTestStatisticExpected.qStatistics is null)
            throw new Exception("Unreachable!");
        AlmostEqual(
            ((Algorithm.QStatistics) acfTestStatistic.qStatistics).qStats,
            ((Algorithm.QStatistics) acfTestStatisticExpected.qStatistics).qStats
        );
        AlmostEqual(
            ((Algorithm.QStatistics) acfTestStatistic.qStatistics).pvalues,
            ((Algorithm.QStatistics) acfTestStatisticExpected.qStatistics).pvalues
        );
    }

    public void testingACFWithAlphaPointQStatAlphaPoint05Bartlett()
    {
        var acfTestStatistic = Algorithm.ACF(
            timeSeries: sunActivities,
            qstat: true,
            alpha: 0.05
        );
        var acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518,
                0.1582025356911707,
                0.4730975308980598,
                0.6589800155363379,
                0.6502908198407041,
                0.4566625437895419,
                0.16179329478317217,
                -0.12205104904067786,
                -0.3161807966260729,
                -0.3747112537274217,
                -0.3060575265823909,
                -0.13480689540510293,
                0.09158727406274836,
                0.29756319807027737,
                0.42070739821101444,
                0.411839537144707,
                0.270207575926828,
                0.04496207925856511
            },
            qStatistics: new Algorithm.QStatistics(
                qStats: new double[] {
                    209.89836353742976,
                    273.64400804059835,
                    274.1359040985167,
                    298.1011690749651,
                    355.26381738878973,
                    400.2444486159611,
                    408.12537759734283,
                    416.1159750621361,
                    487.8126436798599,
                    627.3826726281839,
                    763.7523617970419,
                    831.2289634860766,
                    839.7275792721771,
                    844.580239517275,
                    877.2572649941754,
                    923.3088647080358,
                    954.1366372053316,
                    960.1380029293418,
                    962.9176594693135,
                    992.3604833413744,
                    1051.4195635777849,
                    1108.2123328696512,
                    1132.7451757848503,
                    1133.4268341711972
                },
                pvalues: new double[] {
                    1.445572992738202e-47,
                    3.7927887230279043e-60,
                    3.9322711091079876e-59,
                    2.7822189384881116e-63,
                    1.2875072901819138e-74,
                    2.4769836759153463e-83,
                    4.313800919310703e-84,
                    6.671800474590689e-85,
                    2.3373183200233053e-99,
                    2.3819790998963243e-128,
                    1.1434340014732639e-156,
                    3.314665335174312e-170,
                    4.25659815419771e-171,
                    3.193479306947175e-171,
                    2.5940105209507806e-177,
                    2.8869346431222513e-186,
                    5.7661916029977535e-192,
                    2.2954782237649135e-192,
                    4.344563928164502e-192,
                    1.6618222792482172e-197,
                    3.074666640781376e-209,
                    1.7330498275991565e-220,
                    7.373439199353681e-225,
                    3.748263065537049e-224
                }
            ),
            confInt: new (double First, double Second)[] {
                (1.0, 1.0),
                (0.7087028389661938, 0.9316997498738511),
                (0.2805097695475459, 0.6220272144715895),
                (-0.14541503918715815, 0.22456814232779493),
                (-0.46088878152182444, -0.09069514071337892),
                (-0.6153762352043454, -0.23510262644320415),
                (-0.578208808998814, -0.17498137004930842),
                (-0.36755091865962286, 0.05280309208071923),
                (-0.05343433343348378, 0.3698394048158252),
                (0.2599955438361258, 0.6861995179599938),
                (0.4331980260674155, 0.8847620050052603),
                (0.4017455872531289, 0.8988360524282794),
                (0.18779622922628442, 0.7255288583527995),
                (-0.11654860138308148, 0.4401351909494258),
                (-0.4015596766211478, 0.15745757853979203),
                (-0.5963512012452317, -0.036010392006914194),
                (-0.6592830341052225, -0.09013947334962086),
                (-0.5966985236938991, -0.015416529470882678),
                (-0.4294273503397586, 0.15981355952955273),
                (-0.20379901538831424, 0.38697356351381096),
                (0.0018240844387089061, 0.5933023117018459),
                (0.12126932270030488, 0.720145473721724),
                (0.1051411116036789, 0.7185379626857351),
                (-0.043290621993795575, 0.5837057738474516),
                (-0.2714181982749296, 0.3613423567920598)
            }
        );
        AlmostEqual(acfTestStatistic.acf.AsVector(), acfTestStatisticExpected.acf.AsVector());
        Debug.Assert(
            acfTestStatistic.qStatistics is not null,
            message: $"qStatistics is null, expected some value"
        );
        if (acfTestStatisticExpected.qStatistics is null)
            throw new Exception("Unreachable!");
        AlmostEqual(
            ((Algorithm.QStatistics) acfTestStatistic.qStatistics).qStats,
            ((Algorithm.QStatistics) acfTestStatisticExpected.qStatistics).qStats
        );
        AlmostEqual(
            ((Algorithm.QStatistics) acfTestStatistic.qStatistics).pvalues,
            ((Algorithm.QStatistics) acfTestStatisticExpected.qStatistics).pvalues
        );
        Debug.Assert(
            acfTestStatistic.confInt is not null,
            message: $"Confint is null, expected some value"
        );
        if (acfTestStatisticExpected.confInt is null)
            throw new Exception("Unreachable!");
        AlmostEqual(acfTestStatistic.confInt, acfTestStatisticExpected.confInt);
    }

    public void testingACFWithAlphaPointQStatAlphaPoint05NonBartlett()
    {
        var acfTestStatistic = Algorithm.ACF(
            timeSeries: sunActivities,
            qstat: true,
            alpha: 0.05,
            bartlettConfInt: false
        );
        var acfTestStatisticExpected = new Algorithm.ACFStatistic(
            acf: new double[] {
                1.0,
                0.8202012944200224,
                0.4512684920095677,
                0.03957655157031838,
                -0.2757919611176017,
                -0.42523943082377474,
                -0.3765950895240612,
                -0.1573739132894518,
                0.1582025356911707,
                0.4730975308980598,
                0.6589800155363379,
                0.6502908198407041,
                0.4566625437895419,
                0.16179329478317217,
                -0.12205104904067786,
                -0.3161807966260729,
                -0.3747112537274217,
                -0.3060575265823909,
                -0.13480689540510293,
                0.09158727406274836,
                0.29756319807027737,
                0.42070739821101444,
                0.411839537144707,
                0.270207575926828,
                0.04496207925856511
            },
            qStatistics: new Algorithm.QStatistics(
                qStats: new double[] {
                    209.89836353742976,
                    273.64400804059835,
                    274.1359040985167,
                    298.1011690749651,
                    355.26381738878973,
                    400.2444486159611,
                    408.12537759734283,
                    416.1159750621361,
                    487.8126436798599,
                    627.3826726281839,
                    763.7523617970419,
                    831.2289634860766,
                    839.7275792721771,
                    844.580239517275,
                    877.2572649941754,
                    923.3088647080358,
                    954.1366372053316,
                    960.1380029293418,
                    962.9176594693135,
                    992.3604833413744,
                    1051.4195635777849,
                    1108.2123328696512,
                    1132.7451757848503,
                    1133.4268341711972
                },
                pvalues: new double[] {
                    1.445572992738202e-47,
                    3.7927887230279043e-60,
                    3.9322711091079876e-59,
                    2.7822189384881116e-63,
                    1.2875072901819138e-74,
                    2.4769836759153463e-83,
                    4.313800919310703e-84,
                    6.671800474590689e-85,
                    2.3373183200233053e-99,
                    2.3819790998963243e-128,
                    1.1434340014732639e-156,
                    3.314665335174312e-170,
                    4.25659815419771e-171,
                    3.193479306947175e-171,
                    2.5940105209507806e-177,
                    2.8869346431222513e-186,
                    5.7661916029977535e-192,
                    2.2954782237649135e-192,
                    4.344563928164502e-192,
                    1.6618222792482172e-197,
                    3.074666640781376e-209,
                    1.7330498275991565e-220,
                    7.373439199353681e-225,
                    3.748263065537049e-224
                }
            ),
            confInt: new (double First, double Second)[] {
                (0.8885015445461714, 1.1114984554538285),
                (0.7087028389661938, 0.9316997498738511),
                (0.3397700365557391, 0.5627669474633963),
                (-0.07192190388351023, 0.151075007024147),
                (-0.38729041657143026, -0.16429350566377307),
                (-0.5367378862776033, -0.31374097536994616),
                (-0.4880935449778898, -0.26509663407023254),
                (-0.2688723687432804, -0.04587545783562319),
                (0.0467040802373421, 0.26970099114499935),
                (0.3615990754442312, 0.5845959863518884),
                (0.5474815600825093, 0.7704784709901665),
                (0.5387923643868755, 0.7617892752945328),
                (0.3451640883357133, 0.5681609992433705),
                (0.05029483932934356, 0.2732917502370008),
                (-0.23354950449450645, -0.010552593586849249),
                (-0.42767925207990154, -0.2046823411722443),
                (-0.4862097091812503, -0.2632127982735931),
                (-0.41755598203621946, -0.19455907112856227),
                (-0.24630535085893154, -0.023308439951274323),
                (-0.01991118139108025, 0.20308572951657697),
                (0.18606474261644876, 0.40906165352410595),
                (0.3092089427571858, 0.5322058536648431),
                (0.30034108169087836, 0.5233379925985356),
                (0.15870912047299937, 0.3817060313806566),
                (-0.06653637619526351, 0.1564605347123937)
            }
        );
        AlmostEqual(acfTestStatistic.acf.AsVector(), acfTestStatisticExpected.acf.AsVector());
        Debug.Assert(
            acfTestStatistic.qStatistics is not null,
            message: $"qStatistics is null, expected some value"
        );
        if (acfTestStatisticExpected.qStatistics is null)
            throw new Exception("Unreachable!");
        AlmostEqual(
            ((Algorithm.QStatistics) acfTestStatistic.qStatistics).qStats,
            ((Algorithm.QStatistics) acfTestStatisticExpected.qStatistics).qStats
        );
        AlmostEqual(
            ((Algorithm.QStatistics) acfTestStatistic.qStatistics).pvalues,
            ((Algorithm.QStatistics) acfTestStatisticExpected.qStatistics).pvalues
        );
        Debug.Assert(
            acfTestStatistic.confInt is not null,
            message: $"Confint is null, expected some value"
        );
        if (acfTestStatisticExpected.confInt is null)
            throw new Exception("Unreachable!");
        AlmostEqual(acfTestStatistic.confInt, acfTestStatisticExpected.confInt);
    }

    public void testingACF()
    {
        Console.WriteLine("Testing ACF");
        Console.WriteLine(" Testing ACF with default settings");
        this.testingACFDefault();
        Console.WriteLine(" Testing ACF explicit number of lags");
        this.testingACFWithNLags();
        Console.WriteLine(" Testing ACF with Alpha 0.05, using Bartlett's formula");
        this.testingACFWithAlphaPoint05Bartlett();
        Console.WriteLine(" Testing ACF with Alpha 0.05, not using Bartlett's formula");
        this.testingACFWithAlphaPoint05NonBartlett();
        Console.WriteLine(" Testing ACF with QStat");
        this.testingACFWithAlphaPointQStat();
        Console.WriteLine(" Testing ACF with QStat, and Alpha 05, using Bartlett's formula");
        this.testingACFWithAlphaPointQStatAlphaPoint05Bartlett();
        Console.WriteLine(" Testing ACF with QStat, and Alpha 05, not using Bartlett's formula");
        this.testingACFWithAlphaPointQStatAlphaPoint05NonBartlett();
    }

    public void run()
    {
        Console.WriteLine("Running Tests");
        this.testingKPSS();
        this.testingACF();
        Console.WriteLine("All tests passed 👍️");
    }
}

class MainClass
{
    static void Main()
    {
        new Tests().run();
    }
}
