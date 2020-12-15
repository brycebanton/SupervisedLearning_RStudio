Neural Networks
================

## General Overview

This is the tuning of Neural Networks using the nnet package in R. This
process is to tune the parameters to best find what will produce the
lowest misclassification rate in the SATimage data set.

## Librarys Used

``` r
library(nnet) 
```

## Investigating Data set

``` r
SATimage = read.csv("C:/Users/vn6415dw/Desktop/RStudio/DSCI 425/Data/SATimage.csv",header = T, sep = ",")
SATimage = data.frame(class=as.factor(SATimage$class),SATimage[,1:36])
head(SATimage)
```

    ##   class TL1 TL2 TL3 TL4 TC1 TC2 TC3 TC4 TR1 TR2 TR3 TR4 CL1 CL2 CL3 CL4 CC1 CC2
    ## 1     3  92 115 120  94  84 102 106  79  84 102 102  83 101 126 133 103  92 112
    ## 2     3  84 102 106  79  84 102 102  83  80 102 102  79  92 112 118  85  84 103
    ## 3     3  84 102 102  83  80 102 102  79  84  94 102  79  84 103 104  81  84  99
    ## 4     3  80 102 102  79  84  94 102  79  80  94  98  76  84  99 104  78  84  99
    ## 5     3  84  94 102  79  80  94  98  76  80 102 102  79  84  99 104  81  76  99
    ## 6     3  80  94  98  76  80 102 102  79  76 102 102  79  76  99 104  81  76  99
    ##   CC3 CC4 CR1 CR2 CR3 CR4 BL1 BL2 BL3 BL4 BC1 BC2 BC3 BC4 BR1 BR2 BR3 BR4
    ## 1 118  85  84 103 104  81 102 126 134 104  88 121 128 100  84 107 113  87
    ## 2 104  81  84  99 104  78  88 121 128 100  84 107 113  87  84  99 104  79
    ## 3 104  78  84  99 104  81  84 107 113  87  84  99 104  79  84  99 104  79
    ## 4 104  81  76  99 104  81  84  99 104  79  84  99 104  79  84 103 104  79
    ## 5 104  81  76  99 108  85  84  99 104  79  84 103 104  79  79 107 109  87
    ## 6 108  85  76 103 118  88  84 103 104  79  79 107 109  87  79 107 109  87

``` r
class(SATimage$class)
```

    ## [1] "factor"

The goal is to try and classify the ‘Class’ (which is a factor) column
in this data set using Neural Networks. To get the best model we will
run misclassification functions and tune the parameters in the Neural
Nets to acheive the lowest misclassifcation.

### This is a function for Regular Neural Networks

``` r
misclass = function(fit,y) {
  temp <- table(fit,y)
  cat("Table of Misclassification\n")
  cat("(row = predicted, col = actual)\n")
  print(temp)
  cat("\n\n")
  numcor <- sum(diag(temp))
  numinc <- length(y) - numcor
  mcr <- numinc/length(y)
  cat(paste("Misclassification Rate = ",format(mcr,digits=3)))
  cat("\n")
}
```

### This is a function for Classification Neural Networks

``` r
misclass.nnet <- function(fit,y) {
temp <- table(predict(fit,type="class"),y)
cat("Table of Misclassification\n")
cat("(row = predicted, col = actual)\n")
print(temp)
cat("\n\n")
numcor <- sum(diag(temp))
numinc <- length(y) - numcor
mcr <- numinc/length(y)
cat(paste("Misclassification Rate = ",format(mcr,digits=3)))
cat("\n")
  }
```

## Code for Neural Networks

Several Models have been done with different parameters to see which one
provides the best Missclassification

``` r
set.seed(888) 
testcases = sample(1:dim(SATimage)[1],1000,replace=F)
SATtest = SATimage[testcases,]
SATtrain = SATimage[-testcases,]

sat.nn = nnet(class~.,data=SATtrain,size=5,decay=.05,maxit=5000)
```

    ## # weights:  221
    ## initial  value 6171.900888 
    ## iter  10 value 5896.562119
    ## iter  20 value 5581.810845
    ## iter  30 value 5149.205716
    ## iter  40 value 4980.265392
    ## iter  50 value 4852.875750
    ## iter  60 value 4695.443358
    ## iter  70 value 4598.867517
    ## iter  80 value 4464.794734
    ## iter  90 value 4394.068209
    ## iter 100 value 4271.544310
    ## iter 110 value 3998.568690
    ## iter 120 value 3816.474662
    ## iter 130 value 3704.485283
    ## iter 140 value 3552.209573
    ## iter 150 value 3359.505703
    ## iter 160 value 3242.659150
    ## iter 170 value 3174.463731
    ## iter 180 value 3143.143621
    ## iter 190 value 3012.965369
    ## iter 200 value 2901.864205
    ## iter 210 value 2879.491682
    ## iter 220 value 2854.152902
    ## iter 230 value 2809.884841
    ## iter 240 value 2730.144675
    ## iter 250 value 2690.009384
    ## iter 260 value 2620.809370
    ## iter 270 value 2213.774175
    ## iter 280 value 2076.741632
    ## iter 290 value 2032.145159
    ## iter 300 value 1991.171780
    ## iter 310 value 1953.711348
    ## iter 320 value 1926.154760
    ## iter 330 value 1912.075750
    ## iter 340 value 1903.808132
    ## iter 350 value 1900.288656
    ## iter 360 value 1893.709824
    ## iter 370 value 1880.768706
    ## iter 380 value 1866.518774
    ## iter 390 value 1835.978574
    ## iter 400 value 1811.112590
    ## iter 410 value 1740.020823
    ## iter 420 value 1679.234667
    ## iter 430 value 1595.144295
    ## iter 440 value 1489.427133
    ## iter 450 value 1433.914660
    ## iter 460 value 1372.469028
    ## iter 470 value 1343.191423
    ## iter 480 value 1318.963544
    ## iter 490 value 1293.191371
    ## iter 500 value 1285.752355
    ## iter 510 value 1281.308911
    ## iter 520 value 1277.674095
    ## iter 530 value 1275.596443
    ## iter 540 value 1274.317215
    ## iter 550 value 1272.472008
    ## iter 560 value 1267.786432
    ## iter 570 value 1262.147901
    ## iter 580 value 1256.007373
    ## iter 590 value 1252.410909
    ## iter 600 value 1250.423661
    ## iter 610 value 1247.773206
    ## iter 620 value 1246.687805
    ## iter 630 value 1245.896910
    ## iter 640 value 1244.322308
    ## iter 650 value 1241.012490
    ## iter 660 value 1238.651610
    ## iter 670 value 1235.702248
    ## iter 680 value 1231.886511
    ## iter 690 value 1228.107444
    ## iter 700 value 1224.939434
    ## iter 710 value 1224.151542
    ## iter 720 value 1223.590932
    ## iter 730 value 1222.918107
    ## iter 740 value 1222.678459
    ## iter 750 value 1222.486906
    ## iter 760 value 1222.339652
    ## iter 770 value 1221.852691
    ## iter 780 value 1221.491928
    ## iter 790 value 1220.515265
    ## iter 800 value 1219.505900
    ## iter 810 value 1217.527319
    ## iter 820 value 1216.610785
    ## iter 830 value 1216.192362
    ## iter 840 value 1215.894388
    ## iter 850 value 1215.669731
    ## iter 860 value 1215.543784
    ## iter 870 value 1215.530143
    ## iter 880 value 1215.527262
    ## iter 890 value 1215.526612
    ## final  value 1215.526582 
    ## converged

``` r
misclass.nnet(sat.nn,SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ##       1   2   3   4   5   7
    ##   1 810   0  11   0  10   0
    ##   2   1 351   0   0   3   0
    ##   3   8   0 705  78   0  14
    ##   4   0   0  21 141   0  68
    ##   5   1  23   2   7 338  23
    ##   7   0   0   0  97  17 706
    ## 
    ## 
    ## Misclassification Rate =  0.112

``` r
sat.nn2 = nnet(class~.,data=SATtrain,size=15,decay=.05,maxit=10000)
```

    ## # weights:  651
    ## initial  value 9375.574608 
    ## iter  10 value 5212.924283
    ## iter  20 value 4707.436501
    ## iter  30 value 4575.665302
    ## iter  40 value 4503.460666
    ## iter  50 value 4439.346403
    ## iter  60 value 3505.341034
    ## iter  70 value 2964.684905
    ## iter  80 value 2879.262961
    ## iter  90 value 2832.559738
    ## iter 100 value 2792.078881
    ## iter 110 value 2771.088180
    ## iter 120 value 2755.622842
    ## iter 130 value 2749.424964
    ## iter 140 value 2724.568185
    ## iter 150 value 2676.969784
    ## iter 160 value 2597.069784
    ## iter 170 value 2424.677449
    ## iter 180 value 2258.666931
    ## iter 190 value 2217.491782
    ## iter 200 value 2165.348296
    ## iter 210 value 2067.345351
    ## iter 220 value 1990.555400
    ## iter 230 value 1908.436275
    ## iter 240 value 1836.269742
    ## iter 250 value 1788.537904
    ## iter 260 value 1742.155882
    ## iter 270 value 1717.142731
    ## iter 280 value 1684.314719
    ## iter 290 value 1654.278411
    ## iter 300 value 1612.277634
    ## iter 310 value 1552.699361
    ## iter 320 value 1525.080877
    ## iter 330 value 1478.596781
    ## iter 340 value 1432.753069
    ## iter 350 value 1378.265814
    ## iter 360 value 1322.714826
    ## iter 370 value 1272.854615
    ## iter 380 value 1229.213530
    ## iter 390 value 1199.748864
    ## iter 400 value 1177.900277
    ## iter 410 value 1162.361888
    ## iter 420 value 1148.348693
    ## iter 430 value 1134.563347
    ## iter 440 value 1123.078457
    ## iter 450 value 1109.277670
    ## iter 460 value 1100.044319
    ## iter 470 value 1094.912186
    ## iter 480 value 1092.484355
    ## iter 490 value 1089.274433
    ## iter 500 value 1086.761682
    ## iter 510 value 1084.487061
    ## iter 520 value 1082.808165
    ## iter 530 value 1080.641084
    ## iter 540 value 1078.115075
    ## iter 550 value 1069.599696
    ## iter 560 value 1063.812089
    ## iter 570 value 1057.495368
    ## iter 580 value 1051.333666
    ## iter 590 value 1043.617924
    ## iter 600 value 1030.218081
    ## iter 610 value 1017.247193
    ## iter 620 value 1009.380188
    ## iter 630 value 1003.073572
    ## iter 640 value 997.476890
    ## iter 650 value 989.253978
    ## iter 660 value 985.071221
    ## iter 670 value 982.668582
    ## iter 680 value 978.796595
    ## iter 690 value 975.616656
    ## iter 700 value 973.641021
    ## iter 710 value 971.172991
    ## iter 720 value 969.417477
    ## iter 730 value 967.934814
    ## iter 740 value 967.140101
    ## iter 750 value 966.726410
    ## iter 760 value 965.970669
    ## iter 770 value 965.410087
    ## iter 780 value 965.013332
    ## iter 790 value 964.794978
    ## iter 800 value 964.485187
    ## iter 810 value 964.040640
    ## iter 820 value 963.595391
    ## iter 830 value 962.983985
    ## iter 840 value 962.548587
    ## iter 850 value 962.053265
    ## iter 860 value 960.950698
    ## iter 870 value 959.880163
    ## iter 880 value 959.519170
    ## iter 890 value 958.559747
    ## iter 900 value 957.459426
    ## iter 910 value 956.599388
    ## iter 920 value 955.960994
    ## iter 930 value 955.504795
    ## iter 940 value 955.076410
    ## iter 950 value 954.232381
    ## iter 960 value 952.140829
    ## iter 970 value 949.860209
    ## iter 980 value 947.178428
    ## iter 990 value 945.603974
    ## iter1000 value 944.643097
    ## iter1010 value 943.472057
    ## iter1020 value 942.601996
    ## iter1030 value 942.353183
    ## iter1040 value 942.276471
    ## iter1050 value 942.236874
    ## iter1060 value 942.158591
    ## iter1070 value 942.110477
    ## iter1080 value 941.938976
    ## iter1090 value 941.807303
    ## iter1100 value 941.672493
    ## iter1110 value 941.324869
    ## iter1120 value 941.083016
    ## iter1130 value 940.034792
    ## iter1140 value 939.510463
    ## iter1150 value 938.585017
    ## iter1160 value 937.273917
    ## iter1170 value 936.113683
    ## iter1180 value 935.606442
    ## iter1190 value 935.191126
    ## iter1200 value 934.931852
    ## iter1210 value 934.802968
    ## iter1220 value 934.753218
    ## iter1230 value 934.695230
    ## iter1240 value 934.624202
    ## iter1250 value 934.592071
    ## iter1260 value 934.515441
    ## iter1270 value 934.374933
    ## iter1280 value 933.895427
    ## iter1290 value 933.528452
    ## iter1300 value 933.036987
    ## iter1310 value 932.783578
    ## iter1320 value 932.543133
    ## iter1330 value 932.332021
    ## iter1340 value 932.062520
    ## iter1350 value 931.320473
    ## iter1360 value 930.386425
    ## iter1370 value 929.179718
    ## iter1380 value 927.646145
    ## iter1390 value 926.533438
    ## iter1400 value 924.687558
    ## iter1410 value 923.024699
    ## iter1420 value 921.740051
    ## iter1430 value 920.830374
    ## iter1440 value 920.563150
    ## iter1450 value 920.386930
    ## iter1460 value 920.221068
    ## iter1470 value 920.115923
    ## iter1480 value 920.010117
    ## iter1490 value 919.993312
    ## iter1500 value 919.967123
    ## iter1510 value 919.867491
    ## iter1520 value 919.778362
    ## iter1530 value 919.745356
    ## iter1540 value 919.715323
    ## iter1550 value 919.690837
    ## iter1560 value 919.647120
    ## iter1570 value 919.586617
    ## iter1580 value 919.482743
    ## iter1590 value 919.394166
    ## iter1600 value 919.349522
    ## iter1610 value 919.322957
    ## iter1620 value 919.298765
    ## iter1630 value 919.282454
    ## iter1640 value 919.277594
    ## iter1650 value 919.276022
    ## iter1660 value 919.275371
    ## iter1670 value 919.275152
    ## iter1670 value 919.275147
    ## iter1670 value 919.275147
    ## final  value 919.275147 
    ## converged

``` r
misclass.nnet(sat.nn2,SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ##       1   2   3   4   5   7
    ##   1 809   0   2   1  12   0
    ##   2   0 367   0   0   0   0
    ##   3   4   0 711  64   0   6
    ##   4   0   1  26 158   0  60
    ##   5   7   6   0   2 326   4
    ##   7   0   0   0  98  30 741
    ## 
    ## 
    ## Misclassification Rate =  0.094

``` r
sat.nn4 = nnet(class~.,data=SATtrain,size=12,decay=.025,maxit=10000)
```

    ## # weights:  522
    ## initial  value 8082.623406 
    ## iter  10 value 5881.394154
    ## iter  20 value 5649.453171
    ## iter  30 value 5258.574754
    ## iter  40 value 5037.239834
    ## iter  50 value 4070.111192
    ## iter  60 value 3420.424418
    ## iter  70 value 3305.151000
    ## iter  80 value 3209.780096
    ## iter  90 value 3117.563878
    ## iter 100 value 3054.298087
    ## iter 110 value 2976.263209
    ## iter 120 value 2912.710349
    ## iter 130 value 2868.054088
    ## iter 140 value 2795.350276
    ## iter 150 value 2496.599125
    ## iter 160 value 2366.193897
    ## iter 170 value 2336.902974
    ## iter 180 value 2295.017763
    ## iter 190 value 2263.503813
    ## iter 200 value 2250.360914
    ## iter 210 value 2235.237894
    ## iter 220 value 2163.926476
    ## iter 230 value 2086.896472
    ## iter 240 value 2044.900953
    ## iter 250 value 2013.796557
    ## iter 260 value 1977.707072
    ## iter 270 value 1937.643162
    ## iter 280 value 1888.208953
    ## iter 290 value 1830.488163
    ## iter 300 value 1759.174271
    ## iter 310 value 1674.992865
    ## iter 320 value 1609.653163
    ## iter 330 value 1524.304068
    ## iter 340 value 1466.320129
    ## iter 350 value 1435.613659
    ## iter 360 value 1401.535783
    ## iter 370 value 1377.350406
    ## iter 380 value 1362.772481
    ## iter 390 value 1354.834093
    ## iter 400 value 1350.676359
    ## iter 410 value 1347.842889
    ## iter 420 value 1339.377652
    ## iter 430 value 1299.374337
    ## iter 440 value 1265.345622
    ## iter 450 value 1240.290379
    ## iter 460 value 1233.048634
    ## iter 470 value 1231.380371
    ## iter 480 value 1226.208277
    ## iter 490 value 1216.846028
    ## iter 500 value 1205.601507
    ## iter 510 value 1198.182248
    ## iter 520 value 1188.667644
    ## iter 530 value 1167.856233
    ## iter 540 value 1156.112495
    ## iter 550 value 1145.056438
    ## iter 560 value 1126.577352
    ## iter 570 value 1108.869209
    ## iter 580 value 1099.717920
    ## iter 590 value 1092.981680
    ## iter 600 value 1083.434926
    ## iter 610 value 1073.739365
    ## iter 620 value 1067.210754
    ## iter 630 value 1061.373043
    ## iter 640 value 1054.570686
    ## iter 650 value 1044.397610
    ## iter 660 value 1038.257461
    ## iter 670 value 1026.606471
    ## iter 680 value 1018.654524
    ## iter 690 value 1010.015079
    ## iter 700 value 999.910626
    ## iter 710 value 993.248324
    ## iter 720 value 980.821600
    ## iter 730 value 970.583137
    ## iter 740 value 951.058851
    ## iter 750 value 932.342577
    ## iter 760 value 919.090575
    ## iter 770 value 909.793532
    ## iter 780 value 903.152876
    ## iter 790 value 897.515544
    ## iter 800 value 891.763404
    ## iter 810 value 890.735892
    ## iter 820 value 889.707419
    ## iter 830 value 887.434438
    ## iter 840 value 884.457262
    ## iter 850 value 881.348595
    ## iter 860 value 877.249610
    ## iter 870 value 872.656907
    ## iter 880 value 866.187034
    ## iter 890 value 858.420475
    ## iter 900 value 848.308235
    ## iter 910 value 837.139158
    ## iter 920 value 824.056668
    ## iter 930 value 815.786937
    ## iter 940 value 804.822495
    ## iter 950 value 798.323332
    ## iter 960 value 793.293664
    ## iter 970 value 789.556442
    ## iter 980 value 785.947971
    ## iter 990 value 781.898488
    ## iter1000 value 780.676411
    ## iter1010 value 780.114172
    ## iter1020 value 779.486266
    ## iter1030 value 778.400419
    ## iter1040 value 775.970430
    ## iter1050 value 772.901356
    ## iter1060 value 768.346868
    ## iter1070 value 764.861582
    ## iter1080 value 759.994172
    ## iter1090 value 753.800710
    ## iter1100 value 746.862284
    ## iter1110 value 742.091531
    ## iter1120 value 734.678512
    ## iter1130 value 728.278376
    ## iter1140 value 720.592482
    ## iter1150 value 714.497685
    ## iter1160 value 710.620504
    ## iter1170 value 708.304532
    ## iter1180 value 705.868895
    ## iter1190 value 702.060438
    ## iter1200 value 699.590527
    ## iter1210 value 697.841560
    ## iter1220 value 695.436281
    ## iter1230 value 694.326449
    ## iter1240 value 692.803756
    ## iter1250 value 691.403172
    ## iter1260 value 690.835652
    ## iter1270 value 690.282702
    ## iter1280 value 689.459224
    ## iter1290 value 688.457537
    ## iter1300 value 687.494114
    ## iter1310 value 686.250049
    ## iter1320 value 684.814941
    ## iter1330 value 683.403248
    ## iter1340 value 682.197794
    ## iter1350 value 680.898349
    ## iter1360 value 678.412753
    ## iter1370 value 675.429164
    ## iter1380 value 672.980019
    ## iter1390 value 671.295914
    ## iter1400 value 669.692337
    ## iter1410 value 668.180379
    ## iter1420 value 666.851901
    ## iter1430 value 666.113912
    ## iter1440 value 665.878326
    ## iter1450 value 665.120446
    ## iter1460 value 664.705031
    ## iter1470 value 664.331322
    ## iter1480 value 664.031533
    ## iter1490 value 663.785114
    ## iter1500 value 663.346199
    ## iter1510 value 662.809041
    ## iter1520 value 662.420084
    ## iter1530 value 662.135413
    ## iter1540 value 661.714442
    ## iter1550 value 661.205332
    ## iter1560 value 660.991075
    ## iter1570 value 660.845594
    ## iter1580 value 660.412928
    ## iter1590 value 660.011897
    ## iter1600 value 659.939600
    ## iter1610 value 659.817476
    ## iter1620 value 659.592920
    ## iter1630 value 659.344010
    ## iter1640 value 658.845281
    ## iter1650 value 658.508199
    ## iter1660 value 658.073869
    ## iter1670 value 657.803439
    ## iter1680 value 657.302953
    ## iter1690 value 656.194342
    ## iter1700 value 654.831907
    ## iter1710 value 653.201545
    ## iter1720 value 650.326075
    ## iter1730 value 646.666568
    ## iter1740 value 641.921604
    ## iter1750 value 639.028341
    ## iter1760 value 637.738179
    ## iter1770 value 636.575878
    ## iter1780 value 635.497923
    ## iter1790 value 634.771070
    ## iter1800 value 633.707060
    ## iter1810 value 632.873741
    ## iter1820 value 632.200737
    ## iter1830 value 631.621427
    ## iter1840 value 631.083930
    ## iter1850 value 630.704266
    ## iter1860 value 630.328870
    ## iter1870 value 630.128504
    ## iter1880 value 630.035793
    ## iter1890 value 629.988900
    ## iter1900 value 629.933189
    ## iter1910 value 629.823767
    ## iter1920 value 629.269692
    ## iter1930 value 628.501081
    ## iter1940 value 626.773012
    ## iter1950 value 625.187861
    ## iter1960 value 624.208686
    ## iter1970 value 623.327744
    ## iter1980 value 621.973445
    ## iter1990 value 620.782946
    ## iter2000 value 620.380866
    ## iter2010 value 619.672954
    ## iter2020 value 618.304760
    ## iter2030 value 616.866438
    ## iter2040 value 616.289919
    ## iter2050 value 616.079590
    ## iter2060 value 615.992446
    ## iter2070 value 615.905390
    ## iter2080 value 615.840193
    ## iter2090 value 615.787618
    ## iter2100 value 615.735673
    ## iter2110 value 615.727840
    ## iter2120 value 615.722099
    ## iter2130 value 615.719879
    ## iter2140 value 615.719427
    ## final  value 615.719395 
    ## converged

``` r
misclass.nnet(sat.nn4,SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ##       1   2   3   4   5   7
    ##   1 816   0   0   0   3   0
    ##   2   0 370   0   0   1   0
    ##   3   1   1 709  47   0   4
    ##   4   0   1  28 239   0  34
    ##   5   3   1   0   2 349   9
    ##   7   0   1   2  35  15 764
    ## 
    ## 
    ## Misclassification Rate =  0.0547

``` r
sat.nn6 = nnet(class~.,data=SATtrain,size=12,decay=.02,maxit=10000)
```

    ## # weights:  522
    ## initial  value 6249.891891 
    ## iter  10 value 4669.611394
    ## iter  20 value 3793.638641
    ## iter  30 value 3295.414491
    ## iter  40 value 3176.320812
    ## iter  50 value 3032.663771
    ## iter  60 value 2934.988636
    ## iter  70 value 2866.475822
    ## iter  80 value 2785.104502
    ## iter  90 value 2730.585720
    ## iter 100 value 2698.930205
    ## iter 110 value 2674.404861
    ## iter 120 value 2650.772078
    ## iter 130 value 2611.349807
    ## iter 140 value 2580.038277
    ## iter 150 value 2563.221306
    ## iter 160 value 2549.998270
    ## iter 170 value 2528.479420
    ## iter 180 value 2507.563670
    ## iter 190 value 2486.228266
    ## iter 200 value 2463.807496
    ## iter 210 value 2434.175972
    ## iter 220 value 2398.428759
    ## iter 230 value 2372.140613
    ## iter 240 value 2355.823472
    ## iter 250 value 2338.545275
    ## iter 260 value 2322.885617
    ## iter 270 value 2306.122866
    ## iter 280 value 2269.242637
    ## iter 290 value 2213.209395
    ## iter 300 value 2173.035712
    ## iter 310 value 2121.899933
    ## iter 320 value 2104.428915
    ## iter 330 value 1930.497103
    ## iter 340 value 1809.222824
    ## iter 350 value 1720.158202
    ## iter 360 value 1573.233350
    ## iter 370 value 1482.962728
    ## iter 380 value 1404.695554
    ## iter 390 value 1338.423436
    ## iter 400 value 1279.166621
    ## iter 410 value 1253.876815
    ## iter 420 value 1224.544054
    ## iter 430 value 1190.655629
    ## iter 440 value 1164.940727
    ## iter 450 value 1149.697945
    ## iter 460 value 1134.618931
    ## iter 470 value 1118.881539
    ## iter 480 value 1112.574740
    ## iter 490 value 1105.501757
    ## iter 500 value 1104.017545
    ## iter 510 value 1103.313522
    ## iter 520 value 1101.943870
    ## iter 530 value 1099.957461
    ## iter 540 value 1097.467312
    ## iter 550 value 1093.007992
    ## iter 560 value 1086.273306
    ## iter 570 value 1079.600442
    ## iter 580 value 1067.838155
    ## iter 590 value 1055.903596
    ## iter 600 value 1054.029096
    ## iter 610 value 1052.735923
    ## iter 620 value 1051.514887
    ## iter 630 value 1048.514236
    ## iter 640 value 1046.388353
    ## iter 650 value 1044.516848
    ## iter 660 value 1043.690360
    ## iter 670 value 1042.728263
    ## iter 680 value 1041.683742
    ## iter 690 value 1040.725601
    ## iter 700 value 1038.672052
    ## iter 710 value 1035.021982
    ## iter 720 value 1034.397684
    ## iter 730 value 1033.840754
    ## iter 740 value 1033.270316
    ## iter 750 value 1032.523668
    ## iter 760 value 1031.966067
    ## iter 770 value 1031.337199
    ## iter 780 value 1029.634837
    ## iter 790 value 1025.815844
    ## iter 800 value 1020.093657
    ## iter 810 value 1016.816898
    ## iter 820 value 1006.058841
    ## iter 830 value 1005.429296
    ## iter 840 value 1004.174266
    ## iter 850 value 1002.647581
    ## iter 860 value 1000.961998
    ## iter 870 value 996.447970
    ## iter 880 value 995.761524
    ## iter 890 value 994.320613
    ## iter 900 value 990.935655
    ## iter 910 value 988.671942
    ## iter 920 value 984.337495
    ## iter 930 value 982.185567
    ## iter 940 value 977.365162
    ## iter 950 value 972.654150
    ## iter 960 value 966.166736
    ## iter 970 value 962.537612
    ## iter 980 value 959.649739
    ## iter 990 value 956.905539
    ## iter1000 value 955.669870
    ## iter1010 value 954.785769
    ## iter1020 value 953.344407
    ## iter1030 value 952.350610
    ## iter1040 value 951.746829
    ## iter1050 value 951.283844
    ## iter1060 value 950.015300
    ## iter1070 value 948.391220
    ## iter1080 value 946.245891
    ## iter1090 value 943.372193
    ## iter1100 value 942.907747
    ## iter1110 value 942.474196
    ## iter1120 value 941.731305
    ## iter1130 value 941.158106
    ## iter1140 value 940.605836
    ## iter1150 value 939.942521
    ## iter1160 value 939.179599
    ## iter1170 value 937.510410
    ## iter1180 value 936.054159
    ## iter1190 value 935.032727
    ## iter1200 value 934.238787
    ## iter1210 value 932.547615
    ## iter1220 value 930.792134
    ## iter1230 value 929.412592
    ## iter1240 value 927.809812
    ## iter1250 value 926.223888
    ## iter1260 value 925.868014
    ## iter1270 value 925.188773
    ## iter1280 value 924.543741
    ## iter1290 value 923.831115
    ## iter1300 value 923.284755
    ## iter1310 value 922.682740
    ## iter1320 value 922.161862
    ## iter1330 value 921.060884
    ## iter1340 value 920.131676
    ## iter1350 value 919.321482
    ## iter1360 value 918.813904
    ## iter1370 value 918.230458
    ## iter1380 value 917.820153
    ## iter1390 value 917.422090
    ## iter1400 value 916.863683
    ## iter1410 value 916.078996
    ## iter1420 value 915.389101
    ## iter1430 value 914.878628
    ## iter1440 value 914.449826
    ## iter1450 value 913.951018
    ## iter1460 value 913.673946
    ## iter1470 value 913.257227
    ## iter1480 value 912.945175
    ## iter1490 value 912.647467
    ## iter1500 value 912.075417
    ## iter1510 value 911.579499
    ## iter1520 value 910.815879
    ## iter1530 value 909.857624
    ## iter1540 value 908.891760
    ## iter1550 value 908.690381
    ## iter1560 value 908.455673
    ## iter1570 value 908.279260
    ## iter1580 value 908.124689
    ## iter1590 value 907.925358
    ## iter1600 value 907.766388
    ## iter1610 value 907.620201
    ## iter1620 value 907.451217
    ## iter1630 value 907.198532
    ## iter1640 value 907.000172
    ## iter1650 value 903.825671
    ## iter1660 value 901.403021
    ## iter1670 value 900.133766
    ## iter1680 value 899.579183
    ## iter1690 value 899.353013
    ## iter1700 value 899.304365
    ## iter1710 value 898.420350
    ## iter1720 value 898.079502
    ## iter1730 value 898.016131
    ## iter1740 value 898.000163
    ## iter1750 value 897.998160
    ## iter1760 value 897.997525
    ## iter1770 value 897.997307
    ## iter1780 value 897.997116
    ## final  value 897.997014 
    ## converged

``` r
misclass.nnet(sat.nn6,SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ##       1   2   3   4   5   7
    ##   1 807   0   2   1   3   0
    ##   2   1 365   0   2   1   0
    ##   3   8   0 708  66   0  11
    ##   4   1   0  26 185   1  51
    ##   5   3   9   0   1 339  14
    ##   7   0   0   3  68  24 735
    ## 
    ## 
    ## Misclassification Rate =  0.0862

## Conclusion

The third Neural Network had the lowest misclassifcation rate and
predicted the best for SATimages at .0547 or 5.47%
