#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Hadnett D00223305
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# =============================================================================
# Exploratiry Data Analysis - STEP 1 - Data Import
# =============================================================================

import os
os.chdir('/Users/williamhadnett/Documents/Data_Science/Data_Science_CA2_WilliamHadnett')

data= pd.read_csv("insuranceCA2v1.csv")

# =============================================================================
# Exploratiry Data Analysis - STEP 1 - Specify data characteristics. 
# =============================================================================

data.info()
data.head()
data.describe()

#   Column                Non-Null Count  Dtype  
# ---  ------                --------------  -----  
# 0   AccountNumber         1338 non-null   object 
# 1   Age                   1337 non-null   float64
# 2   YearsHealthInsurance  1337 non-null   float64
# 3   Gender                1337 non-null   object 
# 4   BMI                   1336 non-null   float64
# 5   Children              1336 non-null   float64
# 6   Smoker                1338 non-null   object 
# 7   Region                1329 non-null   object 
# 8   TotalClaims           1338 non-null   float64

# =============================================================================
# Exploratiry Data Analysis - STEP 1 - Identify Variables
# =============================================================================

# TotalClaims - Response Variable - Numerical - Regression Model Requried
# AccountNumber - Categorical - Unique - Cannot be used due to uniquness
# Age - Predictor Variable - Numerical
# YearsHealthInsurance - Predictor Variable - Numerical
# Gender - Predictor Variable - Categorical 
# BMI - Predictor Variable - Numerical
# Childern - Predictor Variable - Numerical
# Smoker - Predictor Variable - Categorical
# Region - Predictor Variable - Categorical

# =============================================================================
# Exploratiry Data Analysis - STEP 2 - Clean the Data
# =============================================================================

#Age
print(data.Age.min()) #Min Age = 3
#Dataset should only contain adults. Therefore over 18.
data[data.Age < 18].Age.count() #Count under 18 = 1
data = data.drop(data[data.Age < 18].index) 

print(data.Age.max()) #Not Acceptable: Max Age = 559
data = data.drop(data[data.Age == 559].index) 

print(data.Age.min()) #Min: 18
print(data.Age.max()) #Max: 64

#YearsHealthInsurance
print(data.YearsHealthInsurance.min()) #Acceptable: 1.0
print(data.YearsHealthInsurance.max()) #Acceptable: 52.0

#Gender
numberGender = data.Gender.value_counts()
print(numberGender)
#Misspelled Genders 'fmale' & 'femael' remove since only one of each mistake.
#fmale could mean either male or female due to one letter difference.
data = data.drop(data[data.Gender == 'fmale'].index) 
data = data.drop(data[data.Gender == 'femael'].index) 
print(data.Gender.unique()) #Deal with 'nan' in Step 3.
#Changed to numerical step 7.

#BMI
print(data.BMI.min()) #Acceptable: 15.96
print(data.BMI.max()) #Acceptable: 53.13

#Children
print(data.Children.min()) #Min: 0
print(data.Children.max()) #Max: 21 (Acceptable: Possible to have 21 children)
#May be considered an outlier in step 4.

#Smoker
numberSmoker = data.Smoker.value_counts()
print(numberSmoker) #Acceptable 'yes' or 'no'. Changed to numerical step7.

#Region
numberRegion = data.Region.value_counts()
print(numberRegion) 
#Acceptable 'southeast', 'northeast', 'southwest' or 'northwest'. Changed to 
#numerical step 7.

#TotalClaims 
print(data.TotalClaims.min()) #Not Acceptable: -46889.26 (Refund/Rebate)
data[data.TotalClaims < 0].TotalClaims.count() # 3 TotalClaims under 0
data = data.drop(data[data.TotalClaims < 0].index) 
data[data.TotalClaims < 0].TotalClaims.count() # 3 TotalClaims under 0

print(data.TotalClaims.max()) #Acceptable: 63770.43

# =============================================================================
# Exploratiry Data Analysis - STEP 3 - Identify and Deal with Missing Values
# =============================================================================

data.isnull().sum()

#Age                     1
#YearsHealthInsurance    1
#Gender                  1
#BMI                     2
#Children                2
#Smoker                  0
#Region                  8
#TotalClaims             0

#Drop Missing Age
data = data.drop(data[data.Age.isnull()].index) 

#Drop Missing YearsHealthInsurance
data = data.drop(data[data.YearsHealthInsurance.isnull()].index) 

#Drop Missing Gender
data = data.drop(data[data.Gender.isnull()].index) 

#Drop Missing BMI
data = data.drop(data[data.BMI.isnull()].index) 

#Drop Missing Children
data = data.drop(data[data.Children.isnull()].index) 

#Handle Missing Region
mode = data.mode()['Region'][0] #Mode = 'southeast'
data['Region'].fillna("southeast", inplace = True)

data.isnull().sum()

#Age                     0
#YearsHealthInsurance    0
#Gender                  0
#BMI                     0
#Children                0
#Smoker                  0
#Region                  0
#TotalClaims             0

# =============================================================================
# Exploratiry Data Analysis - STEP 4 - Identify and Deal with Outliers
# =============================================================================

data.info()

#Examine Age - OK
data.Age.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.Age)
plt.show()
#Values look ok, with no outliers.

#Examine YearsHealthInsurance - OK
data.YearsHealthInsurance.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.YearsHealthInsurance)
plt.show()
#Values look ok, with no outliers.

#Examine BMI - OK
data.BMI.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.BMI)
plt.show()
#Outliers are present, but look resonible - no extreme outliers.

#Examine Children - NOT OK 
data.Children.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.Children)
plt.show()
#One outlier present - extreme outlier - Removed below:  
data = data.drop(data[data.Children == 21].index) 

#Examine TotalClaims - Possible Extreme Outliers
data.TotalClaims.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.TotalClaims)
plt.show()

#Interquartile Range
q1 = data.TotalClaims.quantile(.25)
q3 = data.TotalClaims.quantile(.75)
iqr = q3 - q1 

#Detect if outliers are mild or extreme.
#https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
#upper inner fence = q3+1.5*iqr = mild outliers
#upper outer fence = q3+3*iqr = extreme outliers

innerFence = q3+(1.5*iqr) #InnerFence: 34213.6
outerFence = q3+(3*iqr) #OuterFence: 51910.4
data[data.TotalClaims > outerFence].TotalClaims.count()
#6 Extreme outliers identified - dropped below:
data = data.drop(data[data.TotalClaims > outerFence].index) 

# =============================================================================
# Exploratiry Data Analysis - STEP 5 - Exploratory Analysis - Univariate
# =============================================================================

########## Categorical Analysis #############
#Gender - Good Gender Ratio - Almost equal split = Male: 50.57, Female: 49.43
percentFemale = (data[data.Gender == 'female'].Age.count() / data.Gender.count()) * 100
percentMale = (data[data.Gender == 'male'].Age.count() / data.Gender.count()) * 100
print("Female: ", percentFemale, "Male: " , percentMale)

numberGender = data.Gender.value_counts(normalize=True)
numberGender.plot.pie()
plt.show()

#Smoker
#Greater number of non-smokers than smokers = Smoker: 19.89, Non Smoker: 80.11 
percentSmoker = (data[data.Smoker == 'yes'].Smoker.count() / data.Smoker.count()) * 100
percentNonSmoker = (data[data.Smoker == 'no'].Smoker.count() / data.Smoker.count()) * 100
print("Smoker: ", percentSmoker, "Non Smoker: " , percentNonSmoker)

numberSmoker = data.Smoker.value_counts(normalize=True)
numberSmoker.plot.pie()
plt.show()

#Region
numberRegion = data.Region.value_counts() 
#Number of claims from regions almost equal, with southeast having slightly more
#total claims than northwest, northeast and southwest. 
numberRegion.plot.barh()
plt.title("Regions")
plt.show()

########## Numerical Analysis #############

#Age
ageMin = data.Age.min()
ageMax = data.Age.max()
ageMean = data.Age.mean()
ageMode = data.mode()['Age'][0]

print("Max Age: ", ageMax , " Min Age: ", ageMin)
print("Mean Age: ", ageMean, " Mode Age: ", ageMode)

#Mean Age 39.12, Mode Age: 18, Min: 18, Max: 64
data.describe().Age
sns.boxplot(x=data.Age)
plt.show()

sns.distplot(data.Age, kde = False, bins=5)
plt.show()

#YearsHealthInsurance
yearMin = data.YearsHealthInsurance.min()
yearMax = data.YearsHealthInsurance.max()
yearMean = data.YearsHealthInsurance.mean()

print("Max Year: ", yearMax , " Min Year: ", yearMin)
print("Mean Years: ", yearMean)

#Min Number of Years with this insurance company 1
#Max Number of Years with this insurance company 52
#Average Number of Years with this insurance company 20.61
data.describe().YearsHealthInsurance
sns.boxplot(x=data.YearsHealthInsurance)
plt.show()

#The must frequent type of claim comes from new customers
#who are with this insurance company under 10 years. 400+
sns.distplot(data.YearsHealthInsurance, kde = False, bins=5)
plt.show()

#BMI
bmiMin = data.BMI.min() #15.96
bmiMax = data.BMI.max() #53.13
bmiMean = data.BMI.mean() #30.61
bmiMedian = data.BMI.median() #30.3

print("Max BMI: ", bmiMax , " Min BMI: ", bmiMin)
print("Mean BMI: ", bmiMean, " Median BMI: ", bmiMedian)


#BMI follows bell curve (normal distribution), which to be expected if we 
#take the BMI of 1300+ individuals at random. 
sns.distplot(data.BMI, kde = True, bins=5)
plt.show()

bmiSTD = np.std(data.BMI) #6.08

#TotalClaims
totalMin = data.TotalClaims.min() 
totalMax = data.TotalClaims.max() 
totalMean = data.TotalClaims.mean() 
totalMedian = data.TotalClaims.median() 

print("Max Total Claims: ", totalMax , " Min Total Claims: ", totalMin)
print("Mean Total Claims: ", totalMean, " Median Total Claims: ", totalMedian)

#The majority of customers seem to have accumlated claims which total between
#$0 and $20,000 with only a small proportion of the customer base exceeding
#$20,000
sns.distplot(data.TotalClaims, kde = False, bins=5)
plt.show()

(data[data.TotalClaims < 20000].TotalClaims.count() / data.TotalClaims.count()) * 100 #80.18
(data[data.TotalClaims >= 20000].TotalClaims.count() / data.TotalClaims.count()) * 100 #80.18

totalClaimsSTD = np.std(data.TotalClaims) #11595.07

# =============================================================================
# Exploratiry Data Analysis - STEP 5 - Exploratory Analysis - Bivariate 
# =============================================================================

#### Numerical - Numerical #### (Possibly do age vs totalclaims)

#I expected to see a much large correlation here. However, only in some cases
#does the total price of claims increase when BMI increases. If a correlation
#is present between BMI and TotalClaims it is very slight.
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.BMI,data.TotalClaims)
plt.title("BMI vs Total Claims")
plt.xlabel("BMI")
plt.ylabel("Total Claims")
plt.show()

#None of the variables in question have an extremely strong correlation with
#the TotalClaims variable. Age and YearsHealthInsurance seem to have the 
#strongest correlation of 0.3, followed by number of Children 0.078. However, There is
#a strong correlation between Age and YearsHealthInsurance. This will have to
#investigated in step 9 a they may be multicolinear.
data[['Age','YearsHealthInsurance','Children','TotalClaims']].corr()
figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(data[['Age', 'YearsHealthInsurance','Children','TotalClaims']].corr(), annot=True, cmap = 'Reds')
plt.show()

#### Numerical - Categorical ####

#After reviewing both the groupby table and the boxplot it is clear that 
#smokers have significantly larger total calims than non smokers. With the 
#mean total claims price being $31140.57 for smokers and only $8433.87 for
#non smokers.
data.groupby('Smoker')['TotalClaims'].mean()
data.groupby('Smoker')['TotalClaims'].median()

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.Smoker, data.TotalClaims)
plt.show()

#Gender vs BMI
#Both males and females have similiar BMI's within the dataset. 
#Males have an mean BMI of 30.59 and Females have a BMI of 30.02
data.groupby('Gender')['BMI'].mean()
data.groupby('Gender')['BMI'].median()

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.Gender, data.BMI)
plt.show()

#Gender vs Age
#The the age of males and females is well balanced within this dataset.
#Males have a mean age of 38.72 and females have a mean age of 39.53.#
data.groupby('Gender')['Age'].mean()
data.groupby('Gender')['Age'].median()

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.Gender, data.Age)
plt.show()

#Region vs TotalClaims 
#The mean totalClaims across each region is very similiar, with northeast
#having a slight large mean than the other three. The total claims also 
#seem to vary more in the southeast region and northeast region. The souteast
#region also seems to have significantly large max totalClaims
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.Region, data.TotalClaims)
plt.show()

#### Categorical - Categorical ####

#Larger number of non smokers (1055) vs smokers (262)
data.Smoker.value_counts()

#Smoker by Region
data['smoker_rate']= np.where(data.Smoker=='yes',1,0)
smokerByRegion= data.groupby('Region')['smoker_rate'].mean()

#From this bar chart we can infer that there are more smokers in the southeast
#region than any of the other three regions.
smokerByRegion.plot.bar()
plt.title("Smoker by Region")
plt.show()

#Smokers by Gender
smokerByGender = data.groupby('Gender')['smoker_rate'].mean()
#From this bar chart we can infer that there are more male smokers than female
#smokers.
smokerByGender.plot.bar()
plt.title("Smoker by Gender")
plt.show()

# =============================================================================
# Exploratiry Data Analysis - STEP 5 - Exploratory Analysis - Multivariate
# =============================================================================

result = pd.pivot_table(data=data, index='Region', columns='Gender',values='smoker_rate')
print(result)

#Males from the southeast region are more likely to smoke and females from the
#southwest region are least likely to smoke.
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

#From the heatmap below we can infer that males from the souteast region have higher
#totalclaims and females from southwest have smaller totalClaims.
result1 = pd.pivot_table(data=data, index='Region', columns='Gender',values='TotalClaims')
print(result1)

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result1, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()


#From that heatmap below we can see that for each region people who smoked had
#a much higher TotalClaims mean than people who do not smoke. With the mean value
#of TotalClaims being the largest for smokers in the southeast region and the
#mean value of claims being the least for non smokers in the southwest region.
result2 = pd.pivot_table(data=data, index='Region', columns='Smoker',values='TotalClaims')
print(result2)

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result2, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

result2 = pd.pivot_table(data=data, index='Region', columns='Smoker',values='BMI')
print(result2)

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result2, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

# =============================================================================
# Exploratiry Data Analysis - STEP 6 - Characterise & Drop Variables 
# =============================================================================

# TotalClaims - Response Variable - Numerical - Regression Model Requried
# AccountNumber - Categorical - Unique - Cannot be used due to uniquness
# Age - Predictor Variable - Numerical
# YearsHealthInsurance - Predictor Variable - Numerical
# Gender - Predictor Variable - Categorical 
# BMI - Predictor Variable - Numerical
# Childern - Predictor Variable - Numerical
# Smoker - Predictor Variable - Categorical
# Region - Predictor Variable - Categorical

print(len(data.AccountNumber.unique())) #1317 Unique Values - Of No Value due to uniqueness
data.drop('AccountNumber', axis = 1, inplace = True)

# =============================================================================
# Exploratiry Data Analysis - STEP 6 - Characterise & Drop Variables 
# =============================================================================

# TotalClaims - Response Variable - Numerical - Regression Model Requried
# AccountNumber - Categorical - Unique - Cannot be used due to uniquness
# Age - Predictor Variable - Numerical
# YearsHealthInsurance - Predictor Variable - Numerical
# Gender - Predictor Variable - Categorical 
# BMI - Predictor Variable - Numerical
# Childern - Predictor Variable - Numerical
# Smoker - Predictor Variable - Categorical
# Region - Predictor Variable - Categorical

print(len(data.AccountNumber.unique())) #1317 Unique Values - Of No Value due to uniqueness
data.drop('AccountNumber', axis = 1, inplace = True)

# =============================================================================
# Exploratiry Data Analysis - STEP 7 - Construct New Variables 
# =============================================================================

#Smoker - Already Converted to Numerical Step 5 Bivariate Analysis.

#Gender
data['gender_num']= np.where(data.Gender=='male',1,0)

#Region - 3 Variables required.
data['southeast_num']=np.where(data.Region =="southeast",1,0)
data['southwest_num']=np.where(data.Region =="southwest",1,0)
data['northeast_num']=np.where(data.Region =="northeast",1,0)

# =============================================================================
# Exploratiry Data Analysis - STEP 8 - Support Construction of Model
# =============================================================================

figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
sns.pairplot(data)

figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(data.corr(), annot=True, cmap = 'Reds')
plt.show()

# Order   Variable                  Correlation  
# -----   --------                  -----------
# 1       Smoker_Rate               0.78
# 2       Age                       0.3
# 3       YearsHealthInsurance      0.3
# 4       BMI                       0.18
# 5       Children                  0.078
# 6       southeast_num             0.067
# 7       gender_num                0.056
# 8       southwest_num            -0.036
# 9       northeast_num             0.0058

# =============================================================================
# Exploratiry Data Analysis - STEP 9 - Multicolinear Data
# =============================================================================

data[['Age','YearsHealthInsurance']].corr()

#Age and YearsHealthInsurance have a correlation of 0.99, suggesting they
#are multicolinear. I have decided to drop YearsHealthInsurance as a persons
#age typically affects their health and this in turn will result in more
#claims against their Health Insurance Company.

data.drop('YearsHealthInsurance', axis = 1, inplace = True)