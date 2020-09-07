# Types of Statistics

## Descriptive Statistics
* Describe and summarize data 

## Inferential Statistics
* Use sample data to make inferences about larger population 


# Data
Numeric 
* Continuous 
* Discrete

Categorical
* Nominal
* Ordinal

# Measure of center 
* mean 
  * sensitive to extreme values 
* sorted median 
* mode 
```
import statistics
statistics.mode(df[col])
```
---

Mean follow tail 

* left skew , left tail 
  * mean < median 
  
 
 * right skew , right tail
  * median < mean 

# Measure of Spread 

## Variance 
The higher the variance , the higher the spread is

> Average distance from each data point to the data's mean 
```
df[col] - df[col].mean()
```
```
# w/o formula
dists = df[col] - df[col].mean()
sq_dists = dists**2
sum_sq_dists = np.sum(sq_dists)

# sample
variance = sum_sq_dists / (len(df)-1)

# with formula

# sample
variance = np.var(df[col],ddof=1)


```

## Standard Deviation 


* calculate by taking the square root of variance

```
# w/o formula 
np.sqrt(np.var(df[col],ddof=1))

# with formula
np.std(df[col],ddof=1)
```
## Mean absolute deviation

```
dists = df[col] - np.mean(df[col])
np.mean(np.abs(dists))
```

## Quantiles
```
np.quantile(df[col],0.5)
np.quantile(df[col].[0,0.25,0.5,0.75,1
```
* Boxplots use quartiles

## Quantiles using np.linspace
```
np.quantile(df[col],np.linspace(0,1,5))

```
## Interquartile range IQR
Height of the box in a boxplot 

```
np.quantile(df[col],0.75) - np.quantile(df[col],0.25)

from scipy.stats import iqr
iqr(df[col])
```

# Outliers

data point is an outlier if 
```
data < Q1 - 1.5 * IQR

data > Q3 + 1.5 * IQR
```

```
from scipy.stats import iqr
iqr = iqr(df[col])

lower_threshold = np.quantile(df[col],0.25) - 1.5 * iqr
upper_threshold = np.quantile(df[col],0.75) + 1.5 * iqr 

df[(df[col] < lower_threshold)|(df[col] > upper_threshold)]
```



