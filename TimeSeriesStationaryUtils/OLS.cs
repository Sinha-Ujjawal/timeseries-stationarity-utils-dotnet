namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Factorization;

    public static class OLS
    {
        public struct OLSResult
        {
            private Matrix<double> xs;
            private Vector<double> ys;
            private QR<double> qrSolver;

            // private values to store the cached result
            private Vector<double>? _parameters=null;
            private double? _xsRank=null;
            private double? _dfResids=null;
            private double? _sse = null;
            private double? _llf=null;
            private double? _aic=null;
            private double? _bic=null;
            private Matrix<double>? _covParams=null;
            private Vector<double>? _bse=null;
            private Vector<double>? _tValues=null;

            public OLSResult(
                Matrix<double> xs,
                Vector<double> ys,
                QR<double> qrSolver
            )
            {
                this.xs = xs;
                this.ys = ys;
                this.qrSolver = qrSolver;
            }

            public Vector<double> parameters()
            {
                if (_parameters == null)
                    _parameters = qrSolver.Solve(ys);
                return (Vector<double>) _parameters;
            }

            public Matrix<double> covParams()
            {
                if (_covParams == null)
                    _covParams = (qrSolver.R.Transpose() * qrSolver.R).Inverse() * scale();
                return (Matrix<double>) _covParams;
            }

            public Vector<double> bse()
            {
                if (_bse == null)
                    _bse = covParams().Diagonal().PointwiseSqrt();
                return (Vector<double>) _bse;
            }

            public Vector<double> tValues()
            {
                if (_tValues == null)
                    _tValues = parameters() / bse();
                return (Vector<double>) _tValues;
            }

            public Vector<double> estimate(Matrix<double>? xsPrime=null)
            {
                if (xsPrime is null)
                    xsPrime = xs;
                return xsPrime.Multiply(parameters());
            }

            public double sse(Matrix<double>? xsPrime=null)
            {
                if (xsPrime == null && _sse != null)
                    return  (double) _sse;
                if (xsPrime == null)
                    _sse = estimate(xsPrime).SumSquaredError(ys);
                return estimate(xsPrime).SumSquaredError(ys);
            }

            public double llf()
            {
                if (_llf == null)
                {
                    double nobs = ys.Count();
                    double nobs2 = nobs / 2;
                    _llf = -nobs2*Math.Log(2*Math.PI) - nobs2*Math.Log(sse() / nobs) - nobs2;
                }
                return (double) _llf;
            }

            public double xsRank()
            {
                if (_xsRank == null)
                    _xsRank = xs.Rank();
                return (double) _xsRank;
            }

            public double dfResids()
            {
                if (_dfResids == null)
                    _dfResids = ys.Count() - xsRank();
                return (double) _dfResids;
            }

            public double scale()
            {
                return sse() / dfResids();
            }

            public double aic()
            {
                if (_aic == null)
                {
                    var k = xsRank();
                    _aic = -2*llf() + 2*k;
                }
                return (double) _aic;
            }

            public double bic()
            {
                if (_bic == null)
                {
                    var k = xsRank();
                    _bic = -2*llf() + Math.Log(ys.Count()) * k;
                }
                return (double) _bic;
            }
        }

        public static OLSResult Fit(
            Matrix<double> xs,
            Vector<double> ys,
            bool addConstant = false,
            // also fit the intercept by adding a constant column containing all 1.0s
            QRMethod qRMethod = QRMethod.Thin
            // qRMethod to use, QRMethod.Thin is the fastest
        )
        {
            if (addConstant)
                xs = xs.WithConstantColumn(1.0);
            var qrSolver = xs.QR(method: qRMethod);
            return new OLSResult(
                xs: xs,
                ys: ys,
                qrSolver: qrSolver
            );
        }

        public struct AutoLagResult
        {
            public readonly double icbest;
            public readonly uint bestlag;
            public readonly Dictionary<uint, OLSResult>? regressionResults;

            public AutoLagResult(
                double icbest,
                uint bestlag,
                Dictionary<uint, OLSResult>? regressionResults=null
            )
            {
                this.icbest = icbest;
                this.bestlag = bestlag;
                this.regressionResults = regressionResults;
            }
        }

        public enum AutoLagCriterion
        {
            AIC,
            BIC,
            TStat
        }

        /// <summary>
        /// Returns the results for the lag length that maximizes the info criterion.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L69
        /// </summary>
        public static AutoLagResult AutoLag(
            Matrix<double> xs,
            Vector<double> ys,
            uint startlag,
            uint maxlag,
            AutoLagCriterion criterion,
            bool returnRegressionResults=false,
            QRMethod qRMethod = QRMethod.Thin
            // qRMethod to use, QRMethod.Thin is the fastest
        )
        {
            double icbest = double.PositiveInfinity;
            uint bestlag = 0;
            Dictionary<uint, OLSResult> regressionResults = new Dictionary<uint, OLSResult>();
            for (uint lag = startlag; lag <= startlag + maxlag; lag += 1)
            {
                var xsLag = xs.SubMatrix(
                    rowIndex: 0,
                    rowCount: xs.RowCount,
                    columnIndex: 0,
                    columnCount: (int) lag
                );
                var ysLag = ys;
                var olsResult = OLS.Fit(xs: xsLag, ys: ysLag, qRMethod: qRMethod);
                regressionResults.Add(lag, olsResult);
            }
            switch (criterion)
            {
                case AutoLagCriterion.AIC:
                    (icbest, bestlag) = (
                        regressionResults
                            .AsEnumerable()
                            .Select(kv => (kv.Value.aic(), kv.Key))
                            .Min()
                    );
                    break;

                case AutoLagCriterion.BIC:
                    (icbest, bestlag) = (
                        regressionResults
                            .AsEnumerable()
                            .Select(kv => (kv.Value.bic(), kv.Key))
                            .Min()
                    );
                    break;

                case AutoLagCriterion.TStat:
                    // stop = stats.norm.ppf(.95)
                    double stop = 1.6448536269514722;
                    // Default values to ensure that always set
                    bestlag = startlag + maxlag;
                    icbest = 0.0;
                    for (uint lag = startlag + maxlag; lag >= startlag; lag -= 1)
                    {
                        var olsResult = regressionResults[lag];
                        icbest = Math.Abs(olsResult.tValues().Last());
                        bestlag = lag;
                        if (Math.Abs(icbest) >= stop)
                            // Break for first lag with a significant t-stat
                            break;
                    }
                    break;

                default:
                    break;
            }
            return new AutoLagResult(
                icbest: icbest,
                bestlag: bestlag,
                regressionResults: returnRegressionResults ? regressionResults : null
            );
        }
    }
}