namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.IntegralTransforms;

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
}