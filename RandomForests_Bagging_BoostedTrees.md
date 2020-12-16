Randomforests\_and\_Boosting
================

## Librarys

``` r
library(rpart)
```

    ## Warning: package 'rpart' was built under R version 3.6.3

``` r
library(rpart.plot)
```

    ## Warning: package 'rpart.plot' was built under R version 3.6.3

``` r
library(partykit)
```

    ## Warning: package 'partykit' was built under R version 3.6.3

    ## Loading required package: grid

    ## Loading required package: libcoin

    ## Warning: package 'libcoin' was built under R version 3.6.3

    ## Loading required package: mvtnorm

    ## Warning: package 'mvtnorm' was built under R version 3.6.3

``` r
library(ipred)
```

    ## Warning: package 'ipred' was built under R version 3.6.3

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.6.3

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
#library(adabag)
```

## Reading in Dataset and looking what is inside

``` r
SATimage = read.csv("C:/Users/vn6415dw/Desktop/RStudio/DSCI 425/Data/SATimage.csv",header = T, sep = ",")
head(SATimage)
```

    ##   TL1 TL2 TL3 TL4 TC1 TC2 TC3 TC4 TR1 TR2 TR3 TR4 CL1 CL2 CL3 CL4 CC1 CC2 CC3
    ## 1  92 115 120  94  84 102 106  79  84 102 102  83 101 126 133 103  92 112 118
    ## 2  84 102 106  79  84 102 102  83  80 102 102  79  92 112 118  85  84 103 104
    ## 3  84 102 102  83  80 102 102  79  84  94 102  79  84 103 104  81  84  99 104
    ## 4  80 102 102  79  84  94 102  79  80  94  98  76  84  99 104  78  84  99 104
    ## 5  84  94 102  79  80  94  98  76  80 102 102  79  84  99 104  81  76  99 104
    ## 6  80  94  98  76  80 102 102  79  76 102 102  79  76  99 104  81  76  99 108
    ##   CC4 CR1 CR2 CR3 CR4 BL1 BL2 BL3 BL4 BC1 BC2 BC3 BC4 BR1 BR2 BR3 BR4 class
    ## 1  85  84 103 104  81 102 126 134 104  88 121 128 100  84 107 113  87     3
    ## 2  81  84  99 104  78  88 121 128 100  84 107 113  87  84  99 104  79     3
    ## 3  78  84  99 104  81  84 107 113  87  84  99 104  79  84  99 104  79     3
    ## 4  81  76  99 104  81  84  99 104  79  84  99 104  79  84 103 104  79     3
    ## 5  81  76  99 108  85  84  99 104  79  84 103 104  79  79 107 109  87     3
    ## 6  85  76 103 118  88  84 103 104  79  79 107 109  87  79 107 109  87     3

## Splitting into Test and Training

Have to make sure our x is a factor

``` r
SATimage$class = as.factor(SATimage$class)
set.seed(888)
dim(SATimage)
```

    ## [1] 4435   37

``` r
testcases = sample(1:dim(SATimage)[1],1000,replace=F)
SATtest = SATimage[testcases,]
SATtrain = SATimage[-testcases,]
```

## Misclassification Function

``` r
misclass.rpart = function (tree) {  
  temp <- table(predict(tree, type = "class"), tree$y)
  cat("Table of Misclassification\n")
  cat("(row = predicted, col = actual)\n")
  print(temp)
  cat("\n\n")
  numcor <- sum(diag(temp))
  numinc <- length(tree$y) - numcor
  mcr <- numinc/length(tree$y)
  cat(paste("Misclassification Rate = ", format(mcr,digits = 3)))
  cat("\n")
}

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

# CART analysis

## Doing a CART using defualt parameters

``` r
mod.default = rpart(class~.,data=SATtrain)
prp(mod.default)  
```

![](RandomForests_Bagging_BoostedTrees_files/figure-gfm/Random%20Forest%20default-1.png)<!-- -->

``` r
yhat = predict(mod.default,newdata=SATtest,type="class") 
misclass(yhat,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 231   7   2   5  18   0
    ##   2   0  88   0   0   2   0
    ##   3   4   0 204  33   0  10
    ##   4   1   2  15  23   0  18
    ##   5   4   6   0   0  61   3
    ##   7  12   2   1  31  21 196
    ## 
    ## 
    ## Misclassification Rate =  0.197

### Doing a CART but controlling how it splits

``` r
control = rpart.control(minsplit=3,minbucket=2,cp=.001)
mod.control = rpart(class~.,data=SATtrain,control=control)
prp(mod.control)
```

![](RandomForests_Bagging_BoostedTrees_files/figure-gfm/Random%20Forest%20Controled-1.png)<!-- -->

``` r
yhat = predict(mod.control,newdata=SATtest,type="class")
misclass(yhat,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 239   2   2   0   8   0
    ##   2   0  99   0   0   5   0
    ##   3   4   0 207  23   0   7
    ##   4   2   1  10  49   0  19
    ##   5   5   2   0   1  76   5
    ##   7   2   1   3  19  13 196
    ## 
    ## 
    ## Misclassification Rate =  0.134

``` r
misclass(mod.control$y,SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 820   0   0   0   0   0
    ##   2   0 374   0   0   0   0
    ##   3   0   0 739   0   0   0
    ##   4   0   0   0 323   0   0
    ##   5   0   0   0   0 368   0
    ##   6   0   0   0   0   0 811
    ## 
    ## 
    ## Misclassification Rate =  0

Above we are doing a controled Random Forest to give it how it can
split. Then the misclassification is given for both training and testing
sets. The Forest had a misclassifcation rate of .134 or 13.4% on the
test and 0 on the training set.

## Crossvalidation on CART

``` r
crpart.sscv = function(fit,y,data,B=25,p=.333) {
  n = length(y)
  cv <- rep(0,B)
  for (i in 1:B) {
    ss <- floor(n*p)
    sam <- sample(1:n,ss)
    temp <- data[-sam,]
    fit2 <- rpart(formula(fit),data=temp,parms=fit$parms,control=fit$control)
    ynew <- predict(fit2,newdata=data[sam,],type="class")
    tab <- table(y[sam],ynew)
    mc <- ss - sum(diag(tab))
    cv[i] <- mc/ss
  }
  cv
}

results = crpart.sscv(mod.control,SATimage$class,data=SATimage,B=50)
results
```

    ##  [1] 0.1375339 0.1361789 0.1273713 0.1409214 0.1334688 0.1449864 0.1470190
    ##  [8] 0.1537940 0.1334688 0.1314363 0.1355014 0.1510840 0.1544715 0.1395664
    ## [15] 0.1470190 0.1510840 0.1355014 0.1415989 0.1341463 0.1307588 0.1537940
    ## [22] 0.1429539 0.1375339 0.1578591 0.1463415 0.1558266 0.1429539 0.1368564
    ## [29] 0.1422764 0.1402439 0.1653117 0.1551491 0.1497290 0.1388889 0.1449864
    ## [36] 0.1388889 0.1409214 0.1443089 0.1456640 0.1470190 0.1334688 0.1341463
    ## [43] 0.1341463 0.1429539 0.1341463 0.1246612 0.1497290 0.1388889 0.1504065
    ## [50] 0.1565041

``` r
summary(results)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.1247  0.1357  0.1419  0.1427  0.1491  0.1653

The above results is using cross validation on the controled Random
Forest Model. Using this function on the original set because the
functions splits into traing and test within itself, we get a
misclassifcation of .1430 or 14.30% on average. This could be better if
it was better tuned.

# Bagging Analysis

``` r
sat.bag = bagging(class~.,data=SATtrain,coob=T)
sat.bag
```

    ## 
    ## Bagging classification trees with 25 bootstrap replications 
    ## 
    ## Call: bagging.data.frame(formula = class ~ ., data = SATtrain, coob = T)
    ## 
    ## Out-of-bag estimate of misclassification error:  0.1132

``` r
phat = predict(sat.bag,newdata=SATtest,type="prob")
head(phat)
```

    ##      1 2    3    4 5    7
    ## [1,] 0 0 0.00 0.16 0 0.84
    ## [2,] 0 0 0.92 0.08 0 0.00
    ## [3,] 1 0 0.00 0.00 0 0.00
    ## [4,] 0 1 0.00 0.00 0 0.00
    ## [5,] 0 0 0.00 0.08 0 0.92
    ## [6,] 0 0 0.00 0.00 1 0.00

``` r
yhat = predict(sat.bag,newdata=SATtest,type="class")
misclass(yhat,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 244   0   2   1   5   0
    ##   2   0 101   0   0   1   0
    ##   3   3   0 215  20   0   5
    ##   4   0   1   4  51   0   6
    ##   5   5   3   0   0  85   4
    ##   7   0   0   1  20  11 212
    ## 
    ## 
    ## Misclassification Rate =  0.092

``` r
misclass(predict(sat.bag,newdata = SATtrain),SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 820   0   0   0   0   0
    ##   2   0 374   0   0   0   0
    ##   3   0   0 739   1   0   0
    ##   4   0   0   0 321   0   0
    ##   5   0   0   0   0 367   0
    ##   7   0   0   0   1   1 811
    ## 
    ## 
    ## Misclassification Rate =  0.000873

The code above shows the default bagging method. The probability table
shows the probability of the actual values being misclassified. The
other tables show the actuall classifcation percentage that this default
bagging method had. This method had a misclassification rate of .1 or
10% on the test set and 0.001 on the training.

## Next we can do a Controled Bagging Method

``` r
control.bag = rpart.control(minsplit=3,minbucket=2,cp=0,xval=0)
sat.bag.control = bagging(class~.,data=SATtrain,nbagg=100,coob=T,control=control)
sat.bag.control
```

    ## 
    ## Bagging classification trees with 100 bootstrap replications 
    ## 
    ## Call: bagging.data.frame(formula = class ~ ., data = SATtrain, nbagg = 100, 
    ##     coob = T, control = control)
    ## 
    ## Out-of-bag estimate of misclassification error:  0.1144

``` r
yhat = predict(sat.bag.control,newdata=SATtest,type="class")
misclass(yhat,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 245   0   3   2   6   0
    ##   2   0 101   0   0   2   0
    ##   3   3   0 212  23   0   4
    ##   4   0   0   5  50   0  12
    ##   5   4   2   0   0  83   3
    ##   7   0   2   2  17  11 208
    ## 
    ## 
    ## Misclassification Rate =  0.101

``` r
misclass(predict(sat.bag.control,newdata = SATtrain),SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 812   4   3   7  16   2
    ##   2   0 367   0   0   2   0
    ##   3   7   0 733  56   2  11
    ##   4   0   0   1 229   0  14
    ##   5   1   1   0   1 335   4
    ##   7   0   2   2  30  13 780
    ## 
    ## 
    ## Misclassification Rate =  0.0521

It looks like the default model did better on this run. The control
model could do better if its better tuned by adjusting its parameters.

## Cross Validation Function for the Bagging method

``` r
bagg.sscv = function(fit,y,data,B=25,nbagg=100,p=.333) {
  n = length(y)
  cv <- rep(0,B)
  for (i in 1:B) {
    ss <- floor(n*p)
    sam <- sample(1:n,ss)
    temp <- data[-sam,]
    fit2 <- bagging(formula(fit),data=temp,control=fit$control,coob=F)
    ynew <- predict(fit2,newdata=data[sam,],type="class")
    tab <- table(y[sam],ynew)
    mc <- ss - sum(diag(tab))
    cv[i] <- mc/ss
  }
  cv
}
results = bagg.sscv(sat.bag.control,SATimage$class,data=SATimage,B=5,nbagg=25)
results
```

    ## [1] 0.1917344 0.1531165 0.1808943 0.1829268 0.1565041

``` r
summary(results)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.1531  0.1565  0.1809  0.1730  0.1829  0.1917

Again this model was done on the orignal data set because it splits into
training and test within itself. The crossvalidation for bagging got a
misclassifcation of .1690 or 16.90% on average.

# Random Forests

``` r
sat.rf = randomForest(class~.,data=SATtrain,mtry=2,importance=T)
sat.rf
```

    ## 
    ## Call:
    ##  randomForest(formula = class ~ ., data = SATtrain, mtry = 2,      importance = T) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 9.4%
    ## Confusion matrix:
    ##     1   2   3   4   5   7 class.error
    ## 1 798   2  17   0   3   0  0.02682927
    ## 2   0 364   0   2   5   3  0.02673797
    ## 3   2   1 716  13   0   7  0.03112314
    ## 4   3   4  66 188   3  59  0.41795666
    ## 5  27   2   0   5 307  27  0.16576087
    ## 7   0   1  16  43  12 739  0.08877928

``` r
yhat = predict(sat.rf,newdata=SATtest,type="class")
misclass(yhat,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 248   0   2   2   5   0
    ##   2   0 102   0   0   2   0
    ##   3   2   0 212  19   0   6
    ##   4   0   0   5  51   0   7
    ##   5   2   1   0   0  85   3
    ##   7   0   2   3  20  10 211
    ## 
    ## 
    ## Misclassification Rate =  0.091

``` r
misclass(predict(sat.rf,newdata = SATtrain),SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 820   0   0   0   0   0
    ##   2   0 374   0   0   0   0
    ##   3   0   0 739   0   0   0
    ##   4   0   0   0 323   0   0
    ##   5   0   0   0   0 368   0
    ##   7   0   0   0   0   0 811
    ## 
    ## 
    ## Misclassification Rate =  0

Doing the Random Forest Method we got a misclassifcation rate of .093 or
9.3% on the test set.

## Improving Model

``` r
sat.rf_tune = randomForest(class~.,data=SATtrain,mtry=4,importance=T)
sat.rf_tune
```

    ## 
    ## Call:
    ##  randomForest(formula = class ~ ., data = SATtrain, mtry = 4,      importance = T) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 4
    ## 
    ##         OOB estimate of  error rate: 8.94%
    ## Confusion matrix:
    ##     1   2   3   4   5   7 class.error
    ## 1 800   3  13   0   4   0  0.02439024
    ## 2   0 364   1   3   5   1  0.02673797
    ## 3   4   1 714  13   0   7  0.03382950
    ## 4   3   2  63 199   2  54  0.38390093
    ## 5  20   2   0   4 315  27  0.14402174
    ## 7   0   1  13  48  13 736  0.09247842

``` r
yhat = predict(sat.rf_tune,newdata=SATtest,type="class")
misclass(yhat,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 246   0   1   2   5   0
    ##   2   0 102   0   0   2   0
    ##   3   3   0 212  19   0   6
    ##   4   0   0   5  54   0   7
    ##   5   3   1   0   0  85   4
    ##   7   0   2   4  17  10 210
    ## 
    ## 
    ## Misclassification Rate =  0.091

``` r
misclass(predict(sat.rf_tune,newdata = SATtrain),SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 820   0   0   0   0   0
    ##   2   0 374   0   0   0   0
    ##   3   0   0 739   0   0   0
    ##   4   0   0   0 323   0   0
    ##   5   0   0   0   0 368   0
    ##   7   0   0   0   0   0 811
    ## 
    ## 
    ## Misclassification Rate =  0

Adjusting the parameters did get a model that has a lower
misclassification of .087 or 8.70%

## Cross Validation Function for Random Forests

``` r
 crf.sscv = function(fit,y,data,B=25,p=.333,mtry=fit$mtry,ntree=fit$ntree) {
  n = length(y)
  cv <- rep(0,B)
  for (i in 1:B) {
    ss <- floor(n*p)
    sam <- sample(1:n,ss)
    temp <- data[-sam,]
    fit2 <- randomForest(formula(fit),data=temp,mtry=mtry,ntree=ntree)
    ynew <- predict(fit2,newdata=data[sam,],type="class")
    tab <- table(y[sam],ynew)
    mc <- ss - sum(diag(tab))
    cv[i] <- mc/ss
  }
  cv
 }
results = crf.sscv(sat.rf_tune,SATimage$class,data=SATimage)
results
```

    ##  [1] 0.07859079 0.09417344 0.08807588 0.08739837 0.07655827 0.09823848
    ##  [7] 0.08536585 0.09417344 0.09823848 0.08875339 0.10162602 0.08739837
    ## [13] 0.09417344 0.09281843 0.09823848 0.09146341 0.09214092 0.09485095
    ## [19] 0.09485095 0.09959350 0.09146341 0.09688347 0.08875339 0.08739837
    ## [25] 0.09552846

``` r
summary(results)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## 0.07656 0.08808 0.09282 0.09187 0.09553 0.10163

# Boosted Trees

Library has to be loaded after the bagging method other wise the code
breaks.

``` r
library(adabag)
```

    ## Warning: package 'adabag' was built under R version 3.6.3

    ## Loading required package: caret

    ## Warning: package 'caret' was built under R version 3.6.3

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.6.3

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

    ## Loading required package: foreach

    ## Warning: package 'foreach' was built under R version 3.6.3

    ## Loading required package: doParallel

    ## Warning: package 'doParallel' was built under R version 3.6.3

    ## Loading required package: iterators

    ## Warning: package 'iterators' was built under R version 3.6.3

    ## Loading required package: parallel

    ## 
    ## Attaching package: 'adabag'

    ## The following object is masked from 'package:ipred':
    ## 
    ##     bagging

``` r
sat.boost = boosting(class~.,data=SATtrain,mfinal=200)
misclass(sat.boost$class,SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 809   0   2   2   8   0
    ##   2   1 372   0   0   0   0
    ##   3  10   0 732  54   0   6
    ##   4   0   0   1 237   1  19
    ##   5   0   2   0   1 343   5
    ##   7   0   0   4  29  16 781
    ## 
    ## 
    ## Misclassification Rate =  0.0469

``` r
yhat = predict(sat.boost,newdata=SATtest,type="class")
misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 244   0   1   1   5   0
    ##   2   0 102   0   0   1   0
    ##   3   4   0 213  20   0   4
    ##   4   0   0   4  53   0  17
    ##   5   4   2   0   0  83   3
    ##   7   0   1   4  18  13 203
    ## 
    ## 
    ## Misclassification Rate =  0.102

``` r
control = rpart.control(minsplit=4,minbucket=2,cp=0)
sat.boost.control = boosting(class~.,data=SATtrain,mfinal=150,control = control)
misclass(sat.boost.control$class,SATtrain$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 820   0   0   0   0   0
    ##   2   0 374   0   0   0   0
    ##   3   0   0 739   0   0   0
    ##   4   0   0   0 323   0   0
    ##   5   0   0   0   0 368   0
    ##   7   0   0   0   0   0 811
    ## 
    ## 
    ## Misclassification Rate =  0

``` r
yhat = predict(sat.boost.control,newdata=SATtest,type="class")
misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   5   0
    ##   2   0 103   0   0   2   0
    ##   3   3   0 212  20   0   5
    ##   4   0   0   7  56   0  11
    ##   5   2   1   0   1  84   4
    ##   7   0   1   2  15  11 207
    ## 
    ## 
    ## Misclassification Rate =  0.091

Doing a Controlled boosted tree acheived a lower misclassfication rate
of 9% or .09.

# Cross Validation Function for Boosted Trees

``` r
boost.sscv = function(fit,y,data,p=.333,B=25,control=rpart.control()){
  n = length(y)
  cv <- rep(0,B)
  for (i in 1:B) {
    ss <- floor(n*p)
    sam <- sample(1:n,ss,replace=F)
    temp <- data[-sam,]
    fit2 <- boosting(formula(fit),data=temp,control=control)
    ypred <- predict(fit2,newdata=data[sam,])
    tab = ypred$confusion
    mc <- ss - sum(diag(tab))
    cv[i] <- mc/ss
  }
  cv
}
```
