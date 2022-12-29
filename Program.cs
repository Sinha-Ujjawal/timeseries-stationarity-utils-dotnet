using System.Diagnostics;
using TimeSeriesStationaryUtils;

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
        var kpssTestStatistic = TimeSeriesStationaryUtils.KPSS.Run(
            timeSeries: sunActivities,
            isStationaryAroundTrend: false
        );
        var kpssTestStatisticExpected = new TimeSeriesStationaryUtils.KPSS.KPSSTestStatistic(
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
        var kpssTestStatistic = TimeSeriesStationaryUtils.KPSS.Run(
            timeSeries: sunActivities,
            isStationaryAroundTrend: true
        );
        var kpssTestStatisticExpected = new TimeSeriesStationaryUtils.KPSS.KPSSTestStatistic(
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
        var acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(timeSeries: sunActivities);
        var acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
        var acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(
            timeSeries: sunActivities,
            nlags: 10
        );
        var acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
        acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(
            timeSeries: sunActivities,
            nlags: 7
        );
        acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
        var acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(
            timeSeries: sunActivities,
            alpha: 0.05
        );
        var acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
        var acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(
            timeSeries: sunActivities,
            alpha: 0.05,
            bartlettConfInt: false
        );
        var acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
        var acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(timeSeries: sunActivities, qstat: true);
        var acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
            qStatistics: new TimeSeriesStationaryUtils.QStat.QStatistics(
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
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatistic.qStatistics).qStats,
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatisticExpected.qStatistics).qStats
        );
        AlmostEqual(
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatistic.qStatistics).pvalues,
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatisticExpected.qStatistics).pvalues
        );
    }

    public void testingACFWithAlphaPointQStatAlphaPoint05Bartlett()
    {
        var acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(
            timeSeries: sunActivities,
            qstat: true,
            alpha: 0.05
        );
        var acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
            qStatistics: new TimeSeriesStationaryUtils.QStat.QStatistics(
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
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatistic.qStatistics).qStats,
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatisticExpected.qStatistics).qStats
        );
        AlmostEqual(
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatistic.qStatistics).pvalues,
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatisticExpected.qStatistics).pvalues
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
        var acfTestStatistic = TimeSeriesStationaryUtils.ACF.Run(
            timeSeries: sunActivities,
            qstat: true,
            alpha: 0.05,
            bartlettConfInt: false
        );
        var acfTestStatisticExpected = new TimeSeriesStationaryUtils.ACF.ACFStatistic(
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
            qStatistics: new TimeSeriesStationaryUtils.QStat.QStatistics(
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
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatistic.qStatistics).qStats,
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatisticExpected.qStatistics).qStats
        );
        AlmostEqual(
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatistic.qStatistics).pvalues,
            ((TimeSeriesStationaryUtils.QStat.QStatistics) acfTestStatisticExpected.qStatistics).pvalues
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
