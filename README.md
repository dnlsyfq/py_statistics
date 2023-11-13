# Statistics

## Population vs Sample

```
import pandas as pd
wnba = pd.read_csv('wnba.csv')

print(wnba.head())
print(wnba.tail())
print(wnba.shape) 

parameter = wnba['Games Played'].max()

sample = wnba.sample(30,random_state = 1)

statistic = sample['Games Played'].max()

sampling_error = parameter - statistic
```
When we sample, we want to minimize the sampling error as much as we can. We want our sample to represent the population as accurately as possible.

If we sampled to measure the mean height of adults in the U.S., we'd like our sample statistic (sample mean height) to get as close as possible to the population's parameter (population mean height)

```
import pandas as pd
import matplotlib.pyplot as plt

wnba_list = []
wnba = pd.read_csv('wnba.csv')

for i in range(0,100):
    wnba_list.append(wnba['PTS'].sample(10, random_state = i).mean())
    
plt.scatter(range(1,101), wnba_list)
plt.axhline(wnba['PTS'].mean())
plt.show()
```
We can solve this problem by increasing the sample size. As we increase the sample size, the sample means vary less around the population mean, and the chances of getting an unrepresentative sample decrease.
The downside of simple random sampling is that it can exclude individuals playing a certain position
 stratified sampling, and we call each stratified group a stratum.

```
wnba['Scored Per Game'] = wnba['PTS'] / wnba['Games Played'] 
print(wnba['Pos'].unique())


pos_mean = {}

for pos in wnba['Pos'].unique():
    pos_mean[pos] = wnba.query(f" Pos=='{pos}' ")['Scored Per Game'].sample(10, random_state=0).mean()
    
position_most_points = max(pos_mean, key=pos_mean.get)
```

---

```
wnba['Scored Per Game'] = wnba['PTS'] / wnba['Games Played'] 
print(wnba['Pos'].unique())


pos_mean = {}

for pos in wnba['Pos'].unique():
    pos_mean[pos] = wnba.query(f" Pos=='{pos}' ")['Scored Per Game'].sample(10, random_state=0).mean()
    
position_most_points = max(pos_mean, key=pos_mean.get)
```

```
print(wnba['Games Played'].value_counts(bins = 3, normalize = True) * 100)
```

```
_1st = wnba[wnba['Games Played'] <= 12] 
_2nd = wnba[(wnba['Games Played'] > 12) 
            & (wnba['Games Played'] <= 22)]
_3rd = wnba[wnba['Games Played'] > 22]

pts_list = []
for i in range(0,100):
    df = pd.concat(a
        [
            _3rd['PTS'].sample(7,random_state=i),
            _2nd['PTS'].sample(2,random_state=i),
            _1st['PTS'].sample(1,random_state=i)
        ])
    pts_list.append(df.mean())
    
plt.scatter(range(1,101),pts_list)
plt.axhline(wnba['PTS'].mean())
plt.show()
```

```
# cluster = wnba['Team'].unique().sample(4,random_state=0)

# print(wnba.query(f" Team.isin(cluster) "))

# print(type(pd.Series(wnba['Team'].unique())))


cluster = pd.Series(wnba['Team'].unique()).sample(4,random_state=0)
df = wnba.query(f" Team.isin({cluster.tolist()}) ")

sampling_error_height = wnba['Height'].mean() - df['Height'].mean()
sampling_error_age = wnba['Age'].mean() - df['Age'].mean()
sampling_error_BMI = wnba['BMI'].mean() - df['BMI'].mean()
sampling_error_points = wnba['PTS'].mean() - df['PTS'].mean()

```

When we try to use a sample to draw conclusions about a population, we do inferential statistics (we infer information from the sample about the population).
 
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

```
import numpy as np
from scipy.stats import mode

print("\n\nUsing NumPy and SciPy")
print("MEAN: ", np.mean(dog_sample_weights))
print("MEDIAN: ", np.median(dog_sample_weights))
print("MODES: ", mode(dog_sample_weights))
```


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



