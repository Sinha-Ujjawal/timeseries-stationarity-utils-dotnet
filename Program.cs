using System.Diagnostics;

namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics;
    using MathNet.Numerics.LinearRegression;
    using MathNet.Numerics.LinearAlgebra;

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

        public static IEnumerable<double> RunningSum(this IEnumerable<double> sequence)
        {
            return sequence.Scan((prev, next) => prev + next, 0.0);
        }

        public static double SquaredSum(this IEnumerable<double> sequence)
        {
            return sequence.Select((value, _) => value * value).Sum();
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

        public static Vector<double> AsVector(this IEnumerable<double> sequence)
        {
            return Vector<double>.Build.DenseOfEnumerable(sequence);
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
                double m = timeSeries.Average();
                resids = timeSeries.Select((value, _) => value - m).AsVector();
                crit = new double[4] { 0.347, 0.463, 0.574, 0.739 };
            }
            else
            {
                double[] xs = Enumerable.Range(1, nobs).Select((value, _) => value * 1.0).ToArray<double>();
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
                if (line != null)
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
        var kpssTestStatistic = TimeSeriesStationaryUtils.Algorithm.KPSS(
            timeSeries: sunActivities,
            isStationaryAroundTrend: false
        );
        var kpssTestStatisticExpected = new TimeSeriesStationaryUtils.Algorithm.KPSSTestStatistic(
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
        var kpssTestStatistic = TimeSeriesStationaryUtils.Algorithm.KPSS(
            timeSeries: sunActivities,
            isStationaryAroundTrend: true
        );
        var kpssTestStatisticExpected = new TimeSeriesStationaryUtils.Algorithm.KPSSTestStatistic(
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

    public void run()
    {
        Console.WriteLine("Running Tests");
        this.testingKPSS();
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
