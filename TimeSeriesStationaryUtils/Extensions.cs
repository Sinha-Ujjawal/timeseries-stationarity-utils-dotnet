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

        public static IEnumerable<T> iterate<T>(
            Func<T, T> next,
            T state
        )
        {
            while (true)
            {
                yield return state;
                state = next(state);   
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

        public static Vector<T> EnsureLength<T>(this Vector<T> vec, uint n, T padWith)
        where T : struct, IEquatable<T>, IFormattable
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

        public static double MeanSquaredError(this IEnumerable<double> that, IEnumerable<double> other)
        {
            return that.Zip(other).Select(values => Math.Pow(values.Second - values.First, 2)).Average();
        }

        public static double MeanSquaredError(this Vector<double> that, Vector<double> other)
        {
            return (that - other).Select(value => value * value).Average();
        }

        public static double SumSquaredError(this IEnumerable<double> that, IEnumerable<double> other)
        {
            return that.Zip(other).Select(values => values.Second - values.First).Select(value => value * value).Sum();
        }

        public static double SumSquaredError(this Vector<double> that, Vector<double> other)
        {
            return (that - other).Select(value => value * value).Sum();
        }

        public static Vector<double> MeanSquaredError(this Matrix<double> that, Matrix<double> other)
        {
            var squaredErrs = (that - other).PointwisePower(2.0);
            return squaredErrs.ColumnSums() / squaredErrs.RowCount;
        }
        
        public static bool HasConstantColumn(this Matrix<double> that)
        {
            return that.EnumerateColumns().Any(col => col.Minimum() == col.Maximum());
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

        public static Vector<T> AsVector<T>(this IEnumerable<T> sequence)
        where T : struct, IEquatable<T>, IFormattable
        {
            return Vector<T>.Build.DenseOfEnumerable(sequence);
        }

        public static Vector<double> AsVector(this IEnumerable<int> sequence)
        {
            return Vector<double>.Build.DenseOfEnumerable(sequence.Select(value => value * 1.0));
        }

        public static T[][] As2DArray<T>(this IEnumerable<T> sequence, uint nRows, uint nCols)
        where T : struct, IEquatable<T>, IFormattable
        {
            return sequence.Chunk((int) nCols).ToArray();
        }

        public static Matrix<T> AsMatrix<T>(this IEnumerable<T> sequence, uint nRows, uint nCols)
        where T : struct, IEquatable<T>, IFormattable
        {
            return Matrix<T>.Build.DenseOfRowArrays(sequence.As2DArray(nRows, nCols));
        }

        public static Matrix<double> AsMatrix(this IEnumerable<int> sequence, uint nRows, uint nCols)
        {
            return Matrix<double>.Build.DenseOfRowArrays(sequence.Select(value => value * 1.0).As2DArray(nRows, nCols));
        }

        public static Matrix<T> AsMatrix<T>(this IEnumerable<IEnumerable<T>> sequence)
        where T : struct, IEquatable<T>, IFormattable
        {
            return Matrix<T>.Build.DenseOfRowArrays(
                sequence
                    .Select(values => values.ToArray())
                    .ToArray()
            );
        }

        public static Matrix<double> AsMatrix(this IEnumerable<IEnumerable<int>> sequence)
        {
            return Matrix<double>.Build.DenseOfRowArrays(
                sequence
                    .Select(values => values.Select(value => value * 1.0).ToArray())
                    .ToArray()
            );
        }

        public static Matrix<T> WithConstantColumn<T>(this Matrix<T> that, T value)
        where T : struct, IEquatable<T>, IFormattable
        {
            var newColumn = Enumerable.Repeat(value, that.RowCount);
            return that.Append(newColumn.AsMatrix(nRows: (uint) that.RowCount, nCols: 1));
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

        public static IEnumerable<IEnumerable<double>> Vandermonde(
            this IEnumerable<double> sequence,
            uint? n = null,
            bool increasing = false
        )
        {
            if (n == null)
                n = (uint) sequence.LongCount();
            return (
                sequence
                .Select(value => {
                    return (
                        Enumerable.Repeat(value, (int) n)
                            .Scan((prev, next) => prev * next, 1.0)
                            .Take((int) n)
                    );
                })
                .Select(values => increasing ? values : values.Reverse())
            );
        }

        public enum Trend
        {
            N,  // "n" : no constant, no trend.
            C,  // "c" : constant only.
            T,  // "t": trend only.
            CT, // "ct" : constant and trend.
            CTT // "ctt" : constant, and linear and quadratic trend.
        }

        public static uint ntrend(this Trend that)
        {
            if (that == Trend.N)
                return 0;
            else if (that == Trend.C || that == Trend.T)
                return 1;
            else if (that == Trend.CT)
                return 2;
            else // ADF.Trend.CTT
                return 3;
        }

        /// <summary>
        /// Add a trend and/or constant to a matrix.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/tsatools.py#L38
        /// </summary>
        public static Matrix<double> AddTrend(
            this Matrix<double> that,
            Trend trend=Trend.C,
            bool prepend=false
        )
        {
            uint trendOrder;
            if (trend == Trend.N)
                return that;
            else if (trend == Trend.C)
                trendOrder = 0;
            else if (trend == Trend.CT || trend == Trend.T)
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
            if (trend == Trend.T)
                trendArr = trendArr.SubMatrix(
                    rowIndex: 0,
                    rowCount: trendArr.RowCount,
                    columnIndex: 1,
                    columnCount: 1
                );
            return prepend ? trendArr.Append(that) : that.Append(trendArr);
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