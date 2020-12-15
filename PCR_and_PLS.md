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
    ## CV           1.543    1.488    1.397   0.3612   0.2683   0.2276   0.2063
    ## adjCV        1.543    1.485    1.394   0.3173   0.2668   0.2263   0.2018
    ##        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
    ## CV      0.2133   0.2119   0.2083    0.2121    0.2132    0.2122    0.2082
    ## adjCV   0.2118   0.2135   0.2054    0.2092    0.2108    0.2084    0.2051
    ##        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps  20 comps
    ## CV       0.2107    0.2231    0.2184    0.2281    0.2370    0.2376    0.2464
    ## adjCV    0.2080    0.2209    0.2156    0.2254    0.2335    0.2324    0.2404
    ##        21 comps  22 comps  23 comps  24 comps  25 comps  26 comps  27 comps
    ## CV       0.2533    0.2479    0.2435    0.2496    0.2512    0.2527    0.2615
    ## adjCV    0.2472    0.2426    0.2396    0.2433    0.2450    0.2459    0.2554
    ##        28 comps  29 comps  30 comps  31 comps  32 comps  33 comps  34 comps
    ## CV       0.2607    0.2588    0.2559    0.2582    0.2504    0.2519    0.2513
    ## adjCV    0.2543    0.2535    0.2516    0.2553    0.2447    0.2449    0.2437
    ##        35 comps  36 comps  37 comps  38 comps  39 comps  40 comps
    ## CV       0.2418    0.2421    0.2407    0.2466    0.2442    0.2414
    ## adjCV    0.2332    0.2335    0.2323    0.2391    0.2361    0.2342
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
    ## CV           1.543    1.294   0.7798   0.2631   0.2212   0.2061   0.2066
    ## adjCV        1.543    1.300   0.7741   0.2585   0.2177   0.2049   0.2042
    ##        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
    ## CV      0.2062   0.2282   0.2360    0.2377    0.2398    0.2324    0.2282
    ## adjCV   0.2036   0.2215   0.2274    0.2288    0.2323    0.2240    0.2194
    ##        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps  20 comps
    ## CV       0.2415    0.2436    0.2434    0.2510    0.2613    0.2606    0.2669
    ## adjCV    0.2310    0.2333    0.2323    0.2393    0.2487    0.2477    0.2535
    ##        21 comps  22 comps  23 comps  24 comps  25 comps  26 comps  27 comps
    ## CV       0.2659    0.2642    0.2646    0.2641    0.2634    0.2637    0.2635
    ## adjCV    0.2525    0.2508    0.2511    0.2506    0.2499    0.2502    0.2500
    ##        28 comps  29 comps  30 comps  31 comps  32 comps  33 comps  34 comps
    ## CV       0.2638    0.2641    0.2642    0.2642    0.2643    0.2643    0.2643
    ## adjCV    0.2503    0.2505    0.2506    0.2506    0.2507    0.2507    0.2507
    ##        35 comps  36 comps  37 comps  38 comps  39 comps  40 comps
    ## CV       0.2643    0.2643    0.2643    0.2643    0.2643    0.2643
    ## adjCV    0.2507    0.2507    0.2507    0.2507    0.2507    0.2507
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

    ##   [1] 0.3224072 0.2197837 0.2367153 0.2155312 0.2620512 0.2004963 0.2778350
    ##   [8] 0.2460072 0.2043607 0.2945748 0.1424403 0.2449263 0.1840883 0.2629873
    ##  [15] 0.2251106 0.1652699 0.1923376 0.2114734 0.2737521 0.2392981 0.2281877
    ##  [22] 0.3021645 0.2532038 0.1355296 0.2172245 0.2232043 0.2467282 0.2497907
    ##  [29] 0.2645552 0.1794709 0.2675046 0.1903308 0.1973864 0.1547288 0.2716768
    ##  [36] 0.1778234 0.2270563 0.2202109 0.1865925 0.2144948 0.1560663 0.3009293
    ##  [43] 0.1953605 0.2114847 0.1614572 0.2036159 0.2691055 0.2334211 0.1523916
    ##  [50] 0.2504608 0.2012337 0.2492258 0.2834041 0.2095546 0.2650500 0.1506578
    ##  [57] 0.2768648 0.2689073 0.2096557 0.2846348 0.3014957 0.2210626 0.2298695
    ##  [64] 0.2626078 0.1969373 0.2815009 0.2351960 0.2093972 0.1622049 0.2396915
    ##  [71] 0.2853066 0.1681175 0.2152494 0.2474455 0.2453772 0.1705606 0.2530776
    ##  [78] 0.2958717 0.2828778 0.2825875 0.2825208 0.1773270 0.2845992 0.2444106
    ##  [85] 0.2683705 0.1389355 0.2665479 0.1612520 0.2621601 0.1687306 0.3106729
    ##  [92] 0.1469986 0.2881012 0.2832057 0.1982923 0.2876148 0.1371770 0.1678551
    ##  [99] 0.2103229 0.2170482

``` r
mean(pls.crossval)
```

    ## [1] 0.2285137

``` r
pcr.crossval=pcr.cv(gasoline$NIR, gasoline$octane)
pcr.crossval
```

    ##   [1] 0.2492383 0.2662957 0.3243972 0.3559781 0.3104557 0.2959004 0.2750705
    ##   [8] 0.2744061 0.2217687 0.2785373 0.2393288 0.2700857 0.2951414 0.3131760
    ##  [15] 0.2381639 0.3115828 0.2040896 0.2529246 0.3075770 0.1875682 0.2590172
    ##  [22] 0.1859456 0.1998754 0.3000725 0.1945556 0.2280205 0.2645185 0.2020133
    ##  [29] 0.1646301 0.2496139 0.2524050 0.2660806 0.2429252 0.2755797 0.2307654
    ##  [36] 0.2578912 0.2693405 0.2620384 0.2296483 0.2439011 0.2820646 0.2433081
    ##  [43] 0.2470958 0.3399640 0.2048538 0.2835726 0.2767849 0.2125087 0.2941100
    ##  [50] 0.2559909 0.3107059 0.2973431 0.3517508 0.2727659 0.3592711 0.2576667
    ##  [57] 0.2629251 0.2736667 0.3320088 0.2781148 0.1877032 0.2291960 0.3111345
    ##  [64] 0.2869707 0.2574018 0.2550815 0.2803968 0.2224497 0.3472068 0.2682084
    ##  [71] 0.2356231 0.2628191 0.2527756 0.2557058 0.2236403 0.3739143 0.2219054
    ##  [78] 0.2740331 0.2917411 0.2882388 0.2723877 0.2771148 0.2662969 0.2206409
    ##  [85] 0.2764618 0.2483238 0.2608383 0.2641757 0.2594915 0.2149338 0.2877277
    ##  [92] 0.2225505 0.2687677 0.2963035 0.2532331 0.1797216 0.2956904 0.3072531
    ##  [99] 0.2529992 0.3347439

``` r
mean(pcr.crossval)
```

    ## [1] 0.264728

The below output shows the average sudo R square for using PLS and PCR
cross validation techniques. In this case the PCR method is the better
regression technique to predict. You can also change the number of
compositions in the functions. Changing the parameters could increase
model accuracy.

## Bootstrap Functions

``` r
bootols.pcr = function(fit,ncomp,B=100) {
  yt=fit$fitted.values+fit$residuals
  yact = yt
  yhat = fit$fitted.values
  poo = yact-yhat
  ASR=mean(poo^2)
  AAR=mean(abs(poo))
  APE=mean(abs(poo)/yact)
  boot.sqerr=rep(0,B)
  boot.abserr=rep(0,B)
  boot.perr=rep(0,B)
  y = fit$model[,1]
  x = fit$model[,-1]
  data = fit$model
  n = nrow(data)
  for (i in 1:B) {
    sam=sample(1:n,n,replace=T)
    samind=sort(unique(sam))
    temp=pcr(y~.,ncomp=ncomp,data=data[sam,])
    ytp=predict(temp,newdata=data[-samind,])
    ypred = ytp
    boot.sqerr[i]=mean((y[-samind]-ypred)^2)
    boot.abserr[i]=mean(abs(y[-samind]-ypred))
    boot.perr[i]=mean(abs(y[-samind]-ypred)/y[-samind])
  }
  ASRo=mean(boot.sqerr)
  AARo=mean(boot.abserr)
  APEo=mean(boot.perr)
  OPsq=.632*(ASRo-ASR)
  OPab=.632*(AARo-AAR)
  OPpe=.632*(APEo-APE)
  RMSEP=sqrt(ASR+OPsq)
  MAEP=AAR+OPab
  MAPEP=(APE+OPpe)*100
  cat("RMSEP\n")
  cat("===============\n")
  cat(RMSEP,"\n\n")
  cat("MAE\n")
  cat("===============\n")
  cat(MAEP,"\n\n")
  cat("MAPE\n")
  cat("===============\n")
  cat(MAPEP,"\n\n")
  return(data.frame(RMSEP=RMSEP,MAE=MAEP,MAPE=MAPEP))  
}
bootols.pls = function(fit,ncomp,B=100) {
  yt=fit$fitted.values+fit$residuals
  yact = yt
  yhat = fit$fitted.values
  poo = yact-yhat
  ASR=mean(poo^2)
  AAR=mean(abs(poo))
  APE=mean(abs(poo)/yact)
  boot.sqerr=rep(0,B)
  boot.abserr=rep(0,B)
  boot.perr=rep(0,B)
  y = fit$model[,1]
  x = fit$model[,-1]
  data = fit$model
  n = nrow(data)
  for (i in 1:B) {
    sam=sample(1:n,n,replace=T)
    samind=sort(unique(sam))
    temp=plsr(y~.,ncomp=ncomp,data=data[sam,])
    ytp=predict(temp,newdata=data[-samind,])
    ypred = ytp
    boot.sqerr[i]=mean((y[-samind]-ypred)^2)
    boot.abserr[i]=mean(abs(y[-samind]-ypred))
    boot.perr[i]=mean(abs(y[-samind]-ypred)/y[-samind])
  }
  ASRo=mean(boot.sqerr)
  AARo=mean(boot.abserr)
  APEo=mean(boot.perr)
  OPsq=.632*(ASRo-ASR)
  OPab=.632*(AARo-AAR)
  OPpe=.632*(APEo-APE)
  RMSEP=sqrt(ASR+OPsq)
  MAEP=AAR+OPab
  MAPEP=(APE+OPpe)*100
  cat("RMSEP\n")
  cat("===============\n")
  cat(RMSEP,"\n\n")
  cat("MAE\n")
  cat("===============\n")
  cat(MAEP,"\n\n")
  cat("MAPE\n")
  cat("===============\n")
  cat(MAPEP,"\n\n")
  return(data.frame(RMSEP=RMSEP,MAE=MAEP,MAPE=MAPEP))  
}
```

``` r
bootols.pcr(oct.pcr,14)
```

    ## RMSEP
    ## ===============
    ## 1.35256 
    ## 
    ## MAE
    ## ===============
    ## 0.9579371 
    ## 
    ## MAPE
    ## ===============
    ## 1.104583

    ##     RMSEP       MAE     MAPE
    ## 1 1.35256 0.9579371 1.104583

``` r
bootols.pls(oct.pls,18)
```

    ## RMSEP
    ## ===============
    ## 1.607289 
    ## 
    ## MAE
    ## ===============
    ## 1.036891 
    ## 
    ## MAPE
    ## ===============
    ## 1.193705

    ##      RMSEP      MAE     MAPE
    ## 1 1.607289 1.036891 1.193705

These functions allow you to see the models RSMEP, MAE, and MAPE. To
determine how well the model does using the Mean Absolute Error (MAE)
will tell you how well it does. Adjusting the parameters in the
functions may increase or decrease model peformance. But be careful for
over-fitting
