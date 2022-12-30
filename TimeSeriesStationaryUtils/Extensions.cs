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

        public static double[][] As2DArray(this IEnumerable<double> sequence, uint nRows, uint nCols)
        {
            return sequence.Chunk((int) nCols).ToArray();
        }

        public static Matrix<double> AsMatrix(this IEnumerable<double> sequence, uint nRows, uint nCols)
        {
            return Matrix<double>.Build.DenseOfRowArrays(sequence.As2DArray(nRows, nCols));
        }

        public static double Truncate(this double val, uint ndigits)
        {
            var _TenPows = Math.Pow(10, ndigits);
            return Math.Truncate(val * _TenPows) / _TenPows;
        }

        public static IEnumerable<double> Diff(this IEnumerable<double> seq)
        {
            return seq.Zip(seq.Skip(1)).Select(values => values.Second - values.First);
        }

        public static uint ntrend(this ADF.Regression that)
        {
            if (that == ADF.Regression.N)
                return 0;
            else if (that == ADF.Regression.C)
                return 1;
            else if (that == ADF.Regression.CT)
                return 2;
            else // ADF.Regression.CTT
                return 3;
        }

        public enum Trim
        {
            Forward, // trim invalid observations in front.
            Backward, // trim invalid initial observations.
            Both, // trim invalid observations on both sides.
            NoTrim, // no trimming of observations.
        }

        public struct LagMatResult
        {
            private Matrix<double> lm;
            private uint nobs;
            private uint nvar;
            private uint maxlag;

            public LagMatResult(
                Matrix<double> lm,
                uint nobs,
                uint nvar,
                uint maxlag
            )
            {
                this.lm = lm;
                this.nobs = nobs;
                this.nvar = nvar;
                this.maxlag = maxlag;
            }

            private (int, int) TrimExtremes(Trim trim)
            {
                int startobs;
                if (trim == Trim.NoTrim || trim == Trim.Forward)
                    startobs = 0;
                else // Trim.Backward OR Trim.Both
                    startobs = (int) maxlag;
                
                int endobs;
                if (trim == Trim.NoTrim || trim == Trim.Backward)
                    endobs = lm.RowCount;
                else // Trim.Forward OR Trim.Both
                    endobs = (int) nobs;
                return (startobs, endobs);
            }

            /// <summary>
            /// original = "ex"
            /// drops the original array returning only the lagged values.
            /// </summary>
            public Matrix<double> LaggedValues(Trim trim=Trim.Forward)
            {
                var dropIdx = (int) nvar;
                var (startobs, endobs) = this.TrimExtremes(trim);
                return lm.SubMatrix(
                    rowIndex: startobs,
                    rowCount: endobs - startobs,
                    columnIndex: dropIdx,
                    columnCount: lm.ColumnCount - (int) dropIdx
                );
            }

            /// <summary>
            /// original = "in"
            /// returns the original array and the lagged values as a single array
            /// </summary>
            public Matrix<double> OriginalWithLaggedValues(Trim trim=Trim.Forward)
            {
                var dropIdx = 0;
                var (startobs, endobs) = this.TrimExtremes(trim);
                return lm.SubMatrix(
                    rowIndex: startobs,
                    rowCount: endobs - startobs,
                    columnIndex: dropIdx,
                    columnCount: lm.ColumnCount - (int) dropIdx
                );
            }

            /// <summary>
            /// original = "sep"
            /// returns a tuple (original array, lagged values). The original
            /// array is truncated to have the same number of rows as
            /// the returned lagmat.
            /// </summary>
            public (Matrix<double>, Matrix<double>) OriginalAndLaggedValues(Trim trim=Trim.Forward)
            {
                var dropIdx = (int) nvar;
                var (startobs, endobs) = this.TrimExtremes(trim);
                var lags = lm.SubMatrix(
                    rowIndex: startobs,
                    rowCount: endobs - startobs,
                    columnIndex: dropIdx,
                    columnCount: lm.ColumnCount - (int) dropIdx
                );
                var leads = lm.SubMatrix(
                    rowIndex: startobs,
                    rowCount: endobs - startobs,
                    columnIndex: 0,
                    columnCount: dropIdx
                );
                return (lags, leads);
            }
        }

        /// <summary>
        /// Create 2d array of lags.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/tsatools.py#L296
        /// </summary>
        public static LagMatResult LagMat(
            this Matrix<double> mat,
            uint maxlag // All lags from zero to maxlag are included.
        )
        {
            var orig = mat;
            var (nobs, nvar) = (mat.RowCount, mat.ColumnCount);
            if (maxlag >= nobs)
                throw new ArgumentOutOfRangeException("maxlag should be < nobs");
            
            var (lmRows, lmCols) = (nobs + (int) maxlag, nvar * ((int) maxlag + 1));
            var lm = (
                Enumerable.Repeat(0.0, lmRows * lmCols)
                .AsMatrix(nRows: (uint) lmRows, nCols: (uint) lmCols)
            );
            foreach(var k in Enumerable.Range(0, (int) maxlag + 1))
            {
                lm.SetSubMatrix(
                    rowIndex: (int) maxlag - k,
                    rowCount: nobs,
                    columnIndex: nvar * ((int) maxlag - k),
                    columnCount: nvar,
                    mat
                );
            }
            return new LagMatResult(lm: lm, nobs: (uint) nobs, nvar: (uint) nvar, maxlag: maxlag);
        }
    }
}