namespace TimeSeriesStationaryUtils
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra.Factorization;

    public static class ADF
    {
        public struct ADFStatistic
        {
            public readonly double adfStat; // The test statistic.
            public readonly double pvalue; // MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
            public readonly uint usedlag; // The number of lags used.
            public readonly uint nobs; // The number of observations used for the ADF regression and calculation of the critical values.
            public readonly MackinnonCritValues mackinnonCritValues;
            // Critical values for the test statistic at the 1 %, 5 %, and 10 %
            // levels. Based on MacKinnon (2010).
            public readonly double? icbest; // The maximized information criterion if autolag is not None.

            public ADFStatistic(
                double adfStat,
                double pvalue,
                uint usedlag,
                uint nobs,
                MackinnonCritValues mackinnonCritValues,
                double? icbest
            )
            {
                this.adfStat = adfStat;
                this.pvalue = pvalue;
                this.usedlag = usedlag;
                this.nobs = nobs;
                this.mackinnonCritValues = mackinnonCritValues;
                this.icbest = icbest;
            }

            public override string ToString()
            {
                var sb = new StringWriter();
                sb.WriteLine("ADF Statistics:");
                sb.WriteLine($"ADF Result: {adfStat}");
                sb.WriteLine($"P Value: {pvalue}");
                sb.WriteLine($"Used Lag: {usedlag}");
                sb.WriteLine($"Num Obs.: {nobs}");
                sb.Write(mackinnonCritValues);
                if (icbest is not null)
                    sb.WriteLine($"Best Information Criterion: {icbest}");
                return sb.ToString();
            }
        }

        public enum Regression
        {
            N,  // "n" : no constant, no trend.
            C,  // "c" : constant only.
            CT, // "ct" : constant and trend.
            CTT // "ctt" : constant, and linear and quadratic trend.
        }

        public static Trend.Regression ToTrendRegression(this Regression that)
        {
            if (that == Regression.N)
                return Trend.Regression.N;
            else if (that == Regression.C)
                return Trend.Regression.C;
            else if (that == Regression.CT)
                return Trend.Regression.CT;
            else // that == Regression.CTT
                return Trend.Regression.CTT;
        }

        /// <summary>
        /// Augmented Dickey-Fuller unit root test.
        /// The Augmented Dickey-Fuller test can be used to test for a unit root in a
        /// univariate process in the presence of serial correlation.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/stattools.py#L166
        ///</summary>
        public static ADFStatistic Run(
            double[] timeSeries,
            uint? maxlag=null, // Maximum lag which is included in test, default value of 12*(nobs/100)^{1/4} is used when ``null``.
            Regression regression=Regression.C,
            // Constant and trend order to include in regression.
            // * "c" : constant only (default).
            // * "ct" : constant and trend.
            // * "ctt" : constant, and linear and quadratic trend.
            // * "n" : no constant, no trend.
            OLS.AutoLagCriterion? autoLagCriterion=OLS.AutoLagCriterion.AIC,
            //  Method to use when automatically determining the lag length among the
            // values 0, 1, ..., maxlag.
            // * If "AIC" (default) or "BIC", then the number of lags is chosen
            // to minimize the corresponding information criterion.
            // * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
            // lag until the t-statistic on the last lag length is significant
            // using a 5%-sized test.
            // * If None, then the number of included lags is set to maxlag.
            QRMethod qRMethod = QRMethod.Thin
            // qRMethod to use, QRMethod.Thin is the fastest
        )
        {
            if (timeSeries.Min() == timeSeries.Max())
                throw new ArgumentException("Invalid input, timeSeries is constant");
            
            uint nobs = (uint) timeSeries.Length;
            uint ntrend = regression.ToTrendRegression().ntrend();
            
            if (maxlag is null)
            {
                // from Greene referencing Schwert 1989
                var maxlagCandidate = (int)(Math.Ceiling(12.0 * Math.Pow(nobs / 100.0, 1 / 4.0)));
                // -1 for the diff
                maxlagCandidate = Math.Min((int) nobs / 2 - (int) ntrend - 1, (int) maxlagCandidate);
                if (maxlagCandidate < 0)
                    throw new NotSupportedException(@"sample size is too short to use selected regression component");
                maxlag = (uint) maxlagCandidate;
            }
            else if (maxlag > (nobs / 2 - ntrend - 1))
                throw new ArgumentException(@"maxlag must be less than (nobs/2 - 1 - ntrend) where ntrend is the number of included deterministic regressors");
            
            double? icbest;
            uint usedlag;
            Matrix<double> fullRHS;
            Vector<double> timeSeriesDiff = timeSeries.Diff().AsVector();
            Matrix<double> timeSeriesDAll = (
                timeSeriesDiff.ToColumnMatrix()
                .LagMat((uint) maxlag)
                .OriginalWithLaggedValues(trim: Extensions.Trim.Both) // original="in" // with lagged values; with trim="both"
            );
            nobs = (uint) timeSeriesDAll.RowCount;
            timeSeriesDAll.SetSubMatrix(
                rowIndex: 0,
                rowCount: timeSeriesDAll.RowCount,
                columnIndex: 0,
                columnCount: 1,
                timeSeries
                    .TakeLast((int) nobs + 1)
                    .SkipLast(1)
                    .AsMatrix(nRows: (uint) timeSeriesDAll.RowCount, nCols: 1)
            ); // replace 0 timeSeriesDiff with level of timeSeries
            Vector<double> timeSeriesDShort = timeSeriesDiff.TakeLast((int) nobs).AsVector();
            if (autoLagCriterion is not null)
            {
                if (regression != Regression.N)
                    fullRHS = timeSeriesDAll.AddTrend(trend: regression.ToTrendRegression(), prepend: true);
                else
                    fullRHS = timeSeriesDAll;
                
                var startlag = fullRHS.ColumnCount - timeSeriesDAll.ColumnCount + 1;
                // 1 for level
                // search for lag length with smallest information criteria
                // Note: use the same number of observations to have comparable IC
                // aic and bic: smaller is better

                OLS.AutoLagResult autoLagResult = OLS.AutoLag(
                    xs: fullRHS,
                    ys: timeSeriesDShort,
                    (uint) startlag,
                    (uint) maxlag,
                    criterion: (OLS.AutoLagCriterion) autoLagCriterion,
                    qRMethod: qRMethod
                );
                var bestlag = (uint) (autoLagResult.bestlag - startlag);
                // rerun ols with icbest best autolag
                timeSeriesDAll = (
                    timeSeriesDiff.ToColumnMatrix()
                    .LagMat(bestlag)
                    .OriginalWithLaggedValues(trim: Extensions.Trim.Both) // original="in" // with lagged values; with trim="both"
                );
                nobs = (uint) timeSeriesDAll.RowCount;
                timeSeriesDAll.SetSubMatrix(
                    rowIndex: 0,
                    rowCount: timeSeriesDAll.RowCount,
                    columnIndex: 0,
                    columnCount: 1,
                    timeSeries
                        .TakeLast((int) nobs + 1)
                        .SkipLast(1)
                        .AsMatrix(nRows: (uint) timeSeriesDAll.RowCount, nCols: 1)
                ); // replace 0 timeSeriesDiff with level of timeSeries
                timeSeriesDShort = timeSeriesDiff.TakeLast((int) nobs).AsVector();
                usedlag = bestlag;
                icbest = autoLagResult.icbest;
            }
            else
            {
                usedlag = (uint) maxlag;
                icbest = null;
            }

            timeSeriesDAll = timeSeriesDAll.SubMatrix(
                rowIndex: 0,
                rowCount: timeSeriesDAll.RowCount,
                columnIndex: 0,
                columnCount: (int) usedlag + 1
            );
            if (regression != Regression.N)
                fullRHS = timeSeriesDAll.AddTrend(trend: regression.ToTrendRegression(), prepend: false);
            else
                fullRHS = timeSeriesDAll;

            OLS.OLSResult olsResult = OLS.Fit(xs: fullRHS, ys: timeSeriesDShort, qRMethod: qRMethod);

            double adfStat = olsResult.tValues()[0];
            
            double pvalue = MackinnonPValue(
                testStat: adfStat,
                regression: regression
            );
            MackinnonCritValues mackinnonCritValues = MackinnonCrit(regression: regression, nobs: nobs);

            return new ADFStatistic(
                adfStat: adfStat,
                pvalue: pvalue,
                usedlag: usedlag,
                nobs: nobs,
                mackinnonCritValues: mackinnonCritValues,
                icbest: icbest
            );
        }

        public static double TauMax(this Regression that)
        {
            if (that == Regression.N)
                return double.PositiveInfinity;
            else if (that == Regression.C)
                return 2.74;
            else if (that == Regression.CT)
                return 0.7;
            else // that == Regression.CTT
                return 0.54;
        }

        public static double TauMin(this Regression that)
        {
            if (that == Regression.N)
                return -19.04;
            else if (that == Regression.C)
                return -18.83;
            else if (that == Regression.CT)
                return -16.18;
            else // that == Regression.CTT
                return -17.17;
        }

        public static double TauStar(this Regression that)
        {
            if (that == Regression.N)
                return -1.04;
            else if (that == Regression.C)
                return -1.61;
            else if (that == Regression.CT)
                return -2.89;
            else // that == Regression.CTT
                return -3.21;
        }

        public static double [] SmallPS(this Regression that)
        {
            if (that == Regression.N)
                return new double[] {0.6344, 1.2378, 0.032496};
            else if (that == Regression.C)
                return new double[] {2.1659, 1.4412, 0.038269};
            else if (that == Regression.CT)
                return new double[] {3.2512, 1.6047, 0.049588};
            else // that == Regression.CTT
                return new double[] {4.0003, 1.658, 0.048288};
        }

        public static double[] LargePS(this Regression that)
        {
            if (that == Regression.N)
                return new double[] {0.4797, 0.93557, -0.06999, 0.033066};
            else if (that == Regression.C)
                return new double[] {1.7339, 0.93202, -0.12745, -0.010368};
            else if (that == Regression.CT)
                return new double[] {2.5261, 0.61654, -0.37956, -0.060285};
            else // that == Regression.CTT
                return new double[] {3.0778, 0.49529, -0.41477, -0.059359};
        }

        /// <summary>
        /// Returns MacKinnon's approximate p-value for teststat.
        /// Taken from https://github.com/statsmodels/statsmodels/blob/142287c84a0afc80abbf57bb8fb2ec215a0af066/statsmodels/tsa/adfvalues.py#L223
        /// </summary>
        public static double MackinnonPValue(
            double testStat,
            // "T-value" from an Augmented Dickey-Fuller regression.
            Regression regression=Regression.C
            // This is the method of regression that was used.  Following MacKinnon's
            // notation, this can be "c" for constant, "n" for no constant, "ct" for
            // constant and trend, and "ctt" for constant, trend, and trend-squared.
        )
        {
            double maxStat = regression.TauMax();
            double minStat = regression.TauMin();
            double starStat = regression.TauStar();
            if (testStat > maxStat)
                return 1.0;
            else if (testStat < minStat)
                return 0.0;
            IEnumerable<double> tauCoef;
            if (testStat <= starStat)
                tauCoef = regression.SmallPS();
            else
                // Note: above is only for z stats
                tauCoef = regression.LargePS();
            tauCoef = tauCoef.Reverse();
            return Normal.CDF(mean: 0, stddev: 1, x: tauCoef.PolyEval(testStat));
        }

        public struct MackinnonCritValues
        {
            public readonly double crit1Percent;
            public readonly double crit5Percent;
            public readonly double crit10Percent;

            public MackinnonCritValues(
                double crit1Percent,
                double crit5Percent,
                double crit10Percent
            )
            {
                this.crit1Percent = crit1Percent;
                this.crit5Percent = crit5Percent;
                this.crit10Percent = crit10Percent;
            }

            public override string ToString()
            {
                var sb = new StringWriter();
                sb.WriteLine($"Mackinnon Critical Values:");
                sb.WriteLine($"Critical Value (1%):\t{crit1Percent}");
                sb.WriteLine($"Critical Value (5%):\t{crit5Percent}");
                sb.WriteLine($"Critical Value (10%):\t{crit10Percent}");
                return sb.ToString();
            }
        }

        public static MackinnonCritValues MackinnonCrit(
            Regression regression=Regression.C,
            // This is the method of regression that was used.  Following MacKinnon's
            // notation, this can be "c" for constant, "n" for no constant, "ct" for
            // constant and trend, and "ctt" for constant, trend, and trend-squared.
            uint? nobs=null
        )
        {
            if (nobs == null)
            {
                if (regression == Regression.N)
                    return new MackinnonCritValues(-2.56574, -1.941, -1.61682);
                else if (regression == Regression.C)
                    return new MackinnonCritValues(-3.43035, -2.86154, -2.56677);
                else if (regression == Regression.CT)
                    return new MackinnonCritValues(-3.95877, -3.41049, -3.12705);
                else // regression == Regression.CTT
                    return new MackinnonCritValues(-4.37113, -3.83239, -3.55326);
            }
            else
            {
                double x = 1.0 / (double) nobs;
                if (regression == Regression.N)
                    return new MackinnonCritValues(
                        crit1Percent: (new double[] {0.0, -3.627, -2.2358, -2.56574}).PolyEval((double) x),
                        crit5Percent: (new double[] {31.223, -3.365, -0.2686 , -1.941}).PolyEval((double) x),
                        crit10Percent: (new double[] {25.364, -2.714, 0.2656, -1.61682}).PolyEval((double) x)
                    );
                else if (regression == Regression.C)
                    return new MackinnonCritValues(
                        crit1Percent: (new double[] {-79.433, -16.786, -6.5393, -3.43035}).PolyEval((double) x),
                        crit5Percent: (new double[] {-40.04, -4.234, -2.8903, -2.86154}).PolyEval((double) x),
                        crit10Percent: (new double[] {0.0, -2.809, -1.5384, -2.56677}).PolyEval((double) x)
                    );
                else if (regression == Regression.CT)
                    return new MackinnonCritValues(
                        crit1Percent: (new double[] {-134.155, -28.428, -9.0531, -3.95877}).PolyEval((double) x),
                        crit5Percent: (new double[] {-45.374, -9.036, -4.3904, -3.41049}).PolyEval((double) x),
                        crit10Percent: (new double[] {-22.38, -3.925, -2.5856, -3.12705}).PolyEval((double) x)
                    );
                else // regression == Regression.CTT
                    return new MackinnonCritValues(
                        crit1Percent: (new double[] {-334.047, -35.819, -11.5882, -4.37113}).PolyEval((double) x),
                        crit5Percent: (new double[] {-118.284, -12.49, -5.9057, -3.83239}).PolyEval((double) x),
                        crit10Percent: (new double[] {-63.559, -5.293, -3.6596, -3.55326}).PolyEval((double) x)
                    );
            }
        }
    }
}