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
#claims than northwest, northeast and southwest. 
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

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.BMI,data.TotalClaims)
plt.title("BMI vs Total Claims")
plt.xlabel("BMI")
plt.ylabel("Total Claims")
plt.show()

sns.heatmap(data[['BMI','TotalClaims']].corr(), annot=True, cmap = 'Reds')
plt.show()

sns.heatmap(data[['Age','BMI','TotalClaims', 'YearsHealthInsurance', 'Children']].corr(), annot=True, cmap = 'Reds')
plt.show()



