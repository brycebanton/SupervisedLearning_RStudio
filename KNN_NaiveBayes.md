NaiveBayes\_KNearestNeighbor
================

## Librarys

``` r
library(MASS)
```

    ## Warning: package 'MASS' was built under R version 3.6.3

``` r
library(kknn)
```

    ## Warning: package 'kknn' was built under R version 3.6.3

``` r
library(class)
```

    ## Warning: package 'class' was built under R version 3.6.3

``` r
library(klaR)
```

    ## Warning: package 'klaR' was built under R version 3.6.3

``` r
library(ggplot2)
```

    ## Warning: package 'ggplot2' was built under R version 3.6.3

## Reading in Dataset and Splitting in Test and Training

``` r
SATimage = read.csv("C:/Users/vn6415dw/Desktop/RStudio/DSCI 425/Data/SATimage.csv",header = T, sep = ",")
oils = read.csv("C:/Users/vn6415dw/Desktop/RStudio/DSCI 425/Data/Oils.csv",header = T, sep = ",")
SATimage = data.frame(class=as.factor(SATimage$class),SATimage[,1:36])
set.seed(888)
testcases = sample(1:dim(SATimage)[1],1000,replace=F)
SATtest = SATimage[testcases,]
SATtrain = SATimage[-testcases,]
```

## Misclassification and Crossvalidation Functions

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

NB.cv = function(X,y,B=25,p=.333,fL=0,usekernel=F) {
  y = as.factor(y)
  data = data.frame(y,X)
  n = length(y)
  cv <- rep(0,B)
  leaveout = floor(n*p)
  for (i in 1:B) {
    sam <- sample(1:n,leaveout,replace=F)
    temp <- data[-sam,]
    fit <- NaiveBayes(y~.,data=temp,fL=fL,usekernel=usekernel)
    pred = predict(fit,newdata=X[sam,])$class
    tab <- table(y[sam],pred)
    mc <- leaveout - sum(diag(tab))
    cv[i] <- mc/leaveout
  }
  cv
}
sknn.cv = function(train,y,B=25,p=.333,k=4,gamma=2) {
  y = as.factor(y)
  data = data.frame(y,train)
  n = length(y)
  cv <- rep(0,B)
  leaveout = floor(n*p)
  for (i in 1:B) {
    sam <- sample(1:n,leaveout,replace=F)
    temp <- data[-sam,]
    fit <- sknn(y~.,data=temp,k=k,gamma=gamma)
    pred = predict(fit,newdata=train[sam,])$class
    tab <- table(y[sam],pred)
    mc <- leaveout - sum(diag(tab))
    cv[i] <- mc/leaveout
  }
  mean(cv)
}
knn.cv = function(train,y,B=25,p=.333,k=3) {
  y = as.factor(y)
  data = data.frame(y,train)
  n = length(y)
  cv <- rep(0,B)
  leaveout = floor(n*p)
  for (i in 1:B) {
    sam <- sample(1:n,leaveout,replace=F)
    pred <- knn(train[-sam,],train[sam,],y[-sam],k=k)
    tab <- table(y[sam],pred)
    mc <- leaveout - sum(diag(tab))
    cv[i] <- mc/leaveout
  }
  cv
}
```

### K Nearest Neighbors

``` r
oil.sknn = sknn(class~.,data=SATtrain,kn=3)
yhat = predict(oil.sknn,newdata=SATtest)
misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 246   0   1   0   4   0
    ##   2   0 102   0   1   2   1
    ##   3   5   0 200  11   0   4
    ##   4   0   1  14  65   0  10
    ##   5   1   1   1   0  89   5
    ##   7   0   1   6  15   7 207
    ## 
    ## 
    ## Misclassification Rate =  0.091

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=3)
yhat2 = predict(sat.sknn,newdata=SATtest)
misclass(yhat2$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 246   0   1   0   4   0
    ##   2   0 102   0   1   2   1
    ##   3   5   0 201  11   0   4
    ##   4   0   0  15  63   0  11
    ##   5   1   1   0   2  88   5
    ##   7   0   2   5  15   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.094

Both models got a approximate 9% misclassifcation rate. This model can
be tuned by changing the kn parameter.

## Splitting the oil data set up into test and training

``` r
oils = oils[,-7]
summary(oils)
```

    ##  Type      Palmitic        Stearic          Oleic          Linoleic    
    ##  A:37   Min.   : 4.50   Min.   :1.700   Min.   :22.80   Min.   : 7.90  
    ##  B:26   1st Qu.: 6.20   1st Qu.:3.475   1st Qu.:26.30   1st Qu.:43.10  
    ##  C: 3   Median : 9.85   Median :4.200   Median :30.70   Median :50.80  
    ##  D: 7   Mean   : 9.04   Mean   :4.200   Mean   :36.73   Mean   :46.49  
    ##  E:11   3rd Qu.:11.12   3rd Qu.:5.000   3rd Qu.:38.62   3rd Qu.:58.08  
    ##  F:10   Max.   :14.90   Max.   :6.700   Max.   :76.70   Max.   :66.10  
    ##  G: 2                                                                  
    ##    Linolenic       Eicosenoic    
    ##  Min.   :0.100   Min.   :0.1000  
    ##  1st Qu.:0.375   1st Qu.:0.1000  
    ##  Median :0.800   Median :0.1000  
    ##  Mean   :2.272   Mean   :0.3115  
    ##  3rd Qu.:2.650   3rd Qu.:0.3000  
    ##  Max.   :9.500   Max.   :1.8000  
    ## 

``` r
names(oils)
```

    ## [1] "Type"       "Palmitic"   "Stearic"    "Oleic"      "Linoleic"  
    ## [6] "Linolenic"  "Eicosenoic"

``` r
oilx = oils[,-1]
oilx = scale(oilx)
oilx = as.data.frame(oilx)
```

## Naive Bayes

``` r
oil.nb = NaiveBayes(Type~.,data=oils) 
yhat.oil = predict(oil.nb)
```

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 6

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 12

``` r
misclass(yhat.oil$class,oils$Type)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit  A  B  C  D  E  F  G
    ##   A 36  0  0  0  0  0  0
    ##   B  0 26  0  0  0  0  0
    ##   C  0  0  3  0  0  0  0
    ##   D  0  0  0  7  0  0  0
    ##   E  1  0  0  0 11  0  0
    ##   F  0  0  0  0  0 10  0
    ##   G  0  0  0  0  0  0  2
    ## 
    ## 
    ## Misclassification Rate =  0.0104

``` r
oil.nb = sknn(Type~.,data=oils) 
yhat.oil = predict(oil.nb)
misclass(yhat.oil$class,oils$Type)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit  A  B  C  D  E  F  G
    ##   A 36  0  0  0  0  0  2
    ##   B  1 26  0  0  0  0  0
    ##   C  0  0  3  0  0  0  0
    ##   D  0  0  0  6  0  0  0
    ##   E  0  0  0  0 11  0  0
    ##   F  0  0  0  1  0 10  0
    ##   G  0  0  0  0  0  0  0
    ## 
    ## 
    ## Misclassification Rate =  0.0417

The Naive Bayes model predicts better for the oils data set than the
K-Nearest Neighbor. The misclassifcations are 0.010 and 0.041
respectivly.
