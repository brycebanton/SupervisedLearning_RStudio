PCR\_and\_PLS
================

## General Overview

This does Principal Component Regression (PCR) and Partial Least Sqaures
(PLS) decompositon on the data set called gasoline from base R. The
overall goal is to acheive the best predictions by using minimal
variables to help predict hence decomposition.

## What is the Difference Between PCR and PLS?

PCR on the data set to summarize the original predictor variables into
few new variables also known as principal components (PCs), which are a
linear combination of the original data. These PCs are then used to
build the linear regression model. The number of principal components,
to incorporate in the model, is chosen by cross-validation (cv). Note
that, PCR is suitable when the data set contains highly correlated
predictors.

PLS identifies new principal components that not only summarizes the
original predictors, but also that are related to the outcome. These
components are then used to fit the regression model. So, compared to
PCR, PLS uses a dimension reduction strategy that is supervised by the
outcome.

\#\#Library’s Used

``` r
library(pls)
```

    ## Warning: package 'pls' was built under R version 3.6.3

    ## 
    ## Attaching package: 'pls'

    ## The following object is masked from 'package:stats':
    ## 
    ##     loadings

## Investigating Data

``` r
data('gasoline')
gasoline.x = gasoline$NIR
dim(gasoline.x)
```

    ## [1]  60 401

## Including Plots

``` r
matplot(t(gasoline.x),type="l",xlab="Variable",ylab="Spectral Intensity") +
title(main="Spectral Readings for Gasoline Data")
```

![](PCR_and_PLS_files/figure-gfm/Spectrical%20Graph-1.png)<!-- -->

    ## integer(0)

The Spectical Readings show big spikes at 150 variables and 250
variables

## PCR Analysis with Loading Plot

Let try PCR analysis on the whole data set

``` r
oct.pcr=pcr(octane~scale(NIR),data=gasoline,ncomp=40,validation="CV")
summary(oct.pcr)
```

    ## Data:    X dimension: 60 401 
    ##  Y dimension: 60 1
    ## Fit method: svdpc
    ## Number of components considered: 40
    ## 
    ## VALIDATION: RMSEP
    ## Cross-validated using 10 random segments.
    ##        (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
    ## CV           1.543    1.488    1.400   0.3565   0.2694   0.2244    0.204
    ## adjCV        1.543    1.486    1.397   0.3415   0.2680   0.2231    0.200
    ##        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
    ## CV      0.2197   0.2150   0.2179    0.2294    0.2312    0.2285    0.2367
    ## adjCV   0.2177   0.2151   0.2150    0.2261    0.2283    0.2245    0.2322
    ##        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps  20 comps
    ## CV       0.2394    0.2381    0.2379    0.2498    0.2478    0.2539    0.2560
    ## adjCV    0.2349    0.2347    0.2335    0.2453    0.2431    0.2480    0.2496
    ##        21 comps  22 comps  23 comps  24 comps  25 comps  26 comps  27 comps
    ## CV       0.2594    0.2587    0.2663    0.2645    0.2658    0.2632    0.2663
    ## adjCV    0.2527    0.2528    0.2614    0.2581    0.2591    0.2561    0.2593
    ##        28 comps  29 comps  30 comps  31 comps  32 comps  33 comps  34 comps
    ## CV       0.2673    0.2619    0.2554    0.2630    0.2502    0.2426    0.2424
    ## adjCV    0.2602    0.2554    0.2500    0.2579    0.2434    0.2366    0.2358
    ##        35 comps  36 comps  37 comps  38 comps  39 comps  40 comps
    ## CV       0.2376    0.2414    0.2345    0.2330    0.2329    0.2386
    ## adjCV    0.2297    0.2333    0.2270    0.2263    0.2259    0.2317
    ## 
    ## TRAINING: % variance explained
    ##         1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps  8 comps
    ## X        71.725    88.57    93.74    97.51    98.28    98.67    99.01    99.20
    ## octane    8.856    22.69    96.39    97.40    98.18    98.51    98.51    98.57
    ##         9 comps  10 comps  11 comps  12 comps  13 comps  14 comps  15 comps
    ## X         99.36     99.48     99.57     99.64     99.70     99.74     99.78
    ## octane    98.79     98.79     98.81     98.88     98.88     98.88     98.88
    ##         16 comps  17 comps  18 comps  19 comps  20 comps  21 comps  22 comps
    ## X          99.81     99.83     99.85     99.86     99.88     99.89     99.90
    ## octane     98.93     98.93     99.00     99.05     99.08     99.10     99.12
    ##         23 comps  24 comps  25 comps  26 comps  27 comps  28 comps  29 comps
    ## X          99.91     99.92     99.93     99.94     99.94     99.95     99.95
    ## octane     99.13     99.21     99.24     99.28     99.30     99.33     99.36
    ##         30 comps  31 comps  32 comps  33 comps  34 comps  35 comps  36 comps
    ## X          99.95     99.96     99.96     99.96     99.97     99.97     99.97
    ## octane     99.37     99.38     99.47     99.51     99.55     99.60     99.61
    ##         37 comps  38 comps  39 comps  40 comps
    ## X          99.98     99.98     99.98     99.98
    ## octane     99.62     99.62     99.64     99.64

``` r
loadingplot(oct.pcr,comps=1:2,lty=1:2,lwd=2,legendpos="topright")
```

![](PCR_and_PLS_files/figure-gfm/PCR%20analysis%20on%20whole%20data%20set-1.png)<!-- -->

When looking at the summary of the PCR analysis the number of PC’s you
want to keep is two in the validation set. This is because after two
PC’s the number is below one. You want to keep everything above one.

## Spliting into Training and Testing sets

``` r
gasoline.train = gasoline[1:50,]
gasoline.test = gasoline[51:60,]
attributes(gasoline.train)
```

    ## $names
    ## [1] "octane" "NIR"   
    ## 
    ## $row.names
    ##  [1] "1"  "2"  "3"  "4"  "5"  "6"  "7"  "8"  "9"  "10" "11" "12" "13" "14" "15"
    ## [16] "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30"
    ## [31] "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45"
    ## [46] "46" "47" "48" "49" "50"
    ## 
    ## $class
    ## [1] "data.frame"

``` r
dim(gasoline.train$NIR)
```

    ## [1]  50 401

``` r
oct.train = pcr(octane~scale(NIR),data=gasoline.train,ncomp=6)
ypred = predict(oct.train,ncomp=6,newdata=gasoline.test)
yact = gasoline.test$octane
sqrt(mean((ypred - yact)^2)) 
```

    ## [1] 0.1721793

The code above does PCR on the training set. It also predicts and get a
sudo R-sqaure of .172. This model does not predict well on this data
set.

## Next lets do PLS analysis, again starting with whole data set and working to more complex stratigies

``` r
oct.pls = plsr(octane~scale(NIR),data=gasoline,ncomp=40,validation="CV")
summary(oct.pls)
```

    ## Data:    X dimension: 60 401 
    ##  Y dimension: 60 1
    ## Fit method: kernelpls
    ## Number of components considered: 40
    ## 
    ## VALIDATION: RMSEP
    ## Cross-validated using 10 random segments.
    ##        (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
    ## CV           1.543    1.338   0.7638   0.2600   0.2312   0.2110   0.2159
    ## adjCV        1.543    1.337   0.7598   0.2565   0.2294   0.2096   0.2129
    ##        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
    ## CV      0.2113   0.2176   0.2301    0.2335    0.2314    0.2161    0.2230
    ## adjCV   0.2084   0.2117   0.2221    0.2249    0.2242    0.2090    0.2147
    ##        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps  20 comps
    ## CV       0.2377    0.2444    0.2560    0.2671    0.2682    0.2754    0.2758
    ## adjCV    0.2275    0.2341    0.2442    0.2545    0.2552    0.2618    0.2620
    ##        21 comps  22 comps  23 comps  24 comps  25 comps  26 comps  27 comps
    ## CV       0.2718    0.2749    0.2745    0.2748    0.2738    0.2734     0.273
    ## adjCV    0.2580    0.2610    0.2605    0.2608    0.2598    0.2594     0.259
    ##        28 comps  29 comps  30 comps  31 comps  32 comps  33 comps  34 comps
    ## CV       0.2732    0.2731    0.2732    0.2732    0.2733    0.2733    0.2734
    ## adjCV    0.2592    0.2591    0.2591    0.2592    0.2593    0.2593    0.2594
    ##        35 comps  36 comps  37 comps  38 comps  39 comps  40 comps
    ## CV       0.2734    0.2734    0.2734    0.2734    0.2734    0.2734
    ## adjCV    0.2593    0.2593    0.2593    0.2593    0.2593    0.2593
    ## 
    ## TRAINING: % variance explained
    ##         1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps  8 comps
    ## X         64.97    83.51    93.72    96.33    98.21    98.58    98.81    98.89
    ## octane    30.54    79.79    97.73    98.27    98.67    98.90    99.05    99.29
    ##         9 comps  10 comps  11 comps  12 comps  13 comps  14 comps  15 comps
    ## X         99.02     99.19     99.44     99.49     99.55     99.60     99.69
    ## octane    99.44     99.53     99.57     99.69     99.76     99.83     99.86
    ##         16 comps  17 comps  18 comps  19 comps  20 comps  21 comps  22 comps
    ## X          99.72     99.76     99.78     99.81     99.83     99.85     99.87
    ## octane     99.90     99.93     99.95     99.97     99.99     99.99     99.99
    ##         23 comps  24 comps  25 comps  26 comps  27 comps  28 comps  29 comps
    ## X          99.88     99.89      99.9     99.91     99.91     99.92     99.93
    ## octane    100.00    100.00     100.0    100.00    100.00    100.00    100.00
    ##         30 comps  31 comps  32 comps  33 comps  34 comps  35 comps  36 comps
    ## X          99.93     99.94     99.94     99.95     99.96     99.96     99.96
    ## octane    100.00    100.00    100.00    100.00    100.00    100.00    100.00
    ##         37 comps  38 comps  39 comps  40 comps
    ## X          99.96     99.97     99.97     99.97
    ## octane    100.00    100.00    100.00    100.00

``` r
loadingplot(oct.pls,comps=1:2,legendpos="topright")
```

![](PCR_and_PLS_files/figure-gfm/PLS%20analysis%20on%20whole%20data%20set-1.png)<!-- -->

The loading plot for this PLS analysis again says to keep two PC’s as it
goes below one in the validation set. If looking at the training set you
will want to use the number before it breaks .90’s.

``` r
oct.train.pls = plsr(octane~scale(NIR),data=gasoline.train,ncomp=6)
ypred.pls = predict(oct.train.pls,ncomp=6,newdata=gasoline.test)
yact.pls = gasoline.test$octane
sqrt(mean((ypred.pls - yact.pls)^2)) 
```

    ## [1] 0.2856796

Using PLS on the training set predicted better then PCR as its sudo R
square is better (.28 vs .17)

## Using Cross Validation Functions to do PCR and PLS

``` r
pls.cv = function(X,y,ncomp=4,p=.80,B=100) {
  n = length(y)
  X = scale(X)
  data = data.frame(X,y)
  cv <- rep(0,B)
  for (i in 1:B) {
    ss <- floor(n*p)
    sam <- sample(1:n,ss,replace=F)
    fit2 <- plsr(y~.,ncomps=ncomp,data=data[sam,])
    ynew <- predict(fit2,ncomp=ncomp,newdata=data[-sam,])
    cv[i] <- sqrt(mean((y[-sam]-ynew)^2))
  }
 cv
}

pcr.cv = function(X,y,ncomp=4,p=.667,B=100) {
  n = length(y)
  X = scale(X)
  data = data.frame(X,y)
  cv <- rep(0,B)
  for (i in 1:B) {
    ss <- floor(n*p)
    sam <- sample(1:n,ss,replace=F)
    fit2 <- pcr(y~.,ncomps=ncomp,data=data[sam,])
    ynew <- predict(fit2,ncomp=ncomp,newdata=data[-sam,])
    cv[i] <- sqrt(mean((y[-sam]-ynew)^2))
  }
  cv
}
```

## Using the Cross Validation Functions

``` r
pls.crossval=pls.cv(gasoline$NIR, gasoline$octane)
pls.crossval
```

    ##   [1] 0.2232033 0.2666422 0.2940267 0.3038268 0.3020835 0.2621828 0.1877082
    ##   [8] 0.1438986 0.3065006 0.1771191 0.1370519 0.2493953 0.2206531 0.2307875
    ##  [15] 0.1652600 0.1854639 0.2697279 0.1528323 0.2400835 0.2110875 0.1920128
    ##  [22] 0.2037738 0.2130527 0.2048147 0.1728266 0.2160767 0.2579587 0.1342757
    ##  [29] 0.2782620 0.2414259 0.1950601 0.2093947 0.2301157 0.1910217 0.2313843
    ##  [36] 0.1710089 0.1640905 0.2066811 0.2479011 0.2636936 0.2610378 0.2618787
    ##  [43] 0.1479526 0.2820870 0.2124435 0.2377730 0.1736803 0.2134593 0.3202932
    ##  [50] 0.1769058 0.2022907 0.2100016 0.1475917 0.2113962 0.2074433 0.2679530
    ##  [57] 0.2335013 0.2691380 0.2389725 0.2471979 0.1738455 0.2747832 0.3006829
    ##  [64] 0.1784196 0.1762783 0.2575400 0.2394515 0.1804769 0.2892900 0.2321126
    ##  [71] 0.1664541 0.2339744 0.1977339 0.2991858 0.2562193 0.2155825 0.2066700
    ##  [78] 0.2816559 0.1809444 0.2308995 0.2059705 0.2371415 0.2430964 0.1788173
    ##  [85] 0.1911893 0.1247390 0.2679290 0.3540676 0.1837401 0.2250956 0.2621384
    ##  [92] 0.2050958 0.2113061 0.2059476 0.2720375 0.2703615 0.2220863 0.2748078
    ##  [99] 0.2847951 0.2575220

``` r
mean(pls.crossval)
```

    ## [1] 0.2252745

``` r
pcr.crossval=pcr.cv(gasoline$NIR, gasoline$octane)
pcr.crossval
```

    ##   [1] 0.2554759 0.3028941 0.3196649 0.2214352 0.3025734 0.2418752 0.2301953
    ##   [8] 0.2315241 0.2278303 0.2770199 0.2478556 0.3218647 0.2486676 0.2903167
    ##  [15] 0.2988642 0.2024106 0.2461223 0.2652667 0.2753259 0.2926911 0.2571725
    ##  [22] 0.2743034 0.3121608 0.3201733 0.2122841 0.2611767 0.2711633 0.2615116
    ##  [29] 0.2534481 0.2478386 0.3313114 0.2679632 0.3643360 0.2844661 0.2541348
    ##  [36] 0.2382320 0.2494740 0.2934053 0.3059607 0.2649273 0.2044655 0.2417291
    ##  [43] 0.2466407 0.2853869 0.2517422 0.2627990 0.2649697 0.2311953 0.2744608
    ##  [50] 0.2103725 0.2198804 0.2480162 0.2844480 0.1999119 0.2086040 0.2237516
    ##  [57] 0.3210769 0.3189328 0.3272028 0.2118019 0.2615665 0.3645931 0.3098559
    ##  [64] 0.2706100 0.2484882 0.1897406 0.1789470 0.2660856 0.2909314 0.2646000
    ##  [71] 0.2108428 0.2759608 0.2817910 0.3096333 0.2852704 0.3560823 0.2779290
    ##  [78] 0.2620487 0.2689504 0.2077162 0.2396011 0.3402802 0.2733966 0.2557143
    ##  [85] 0.3301205 0.3194812 0.3107349 0.2824593 0.2272138 0.2496604 0.3069150
    ##  [92] 0.2975737 0.2620319 0.3270490 0.2525398 0.2773574 0.2368913 0.2716653
    ##  [99] 0.2843967 0.3084392

``` r
mean(pcr.crossval)
```

    ## [1] 0.2689787

The below output shows the average sudo R square for using PLS and PCR
cross validation techniques. In this case the PCR method is the better
regression technique to predict. You can also change the number of
compositions in the functions. Changing the parameters could increase
model accuracy.
