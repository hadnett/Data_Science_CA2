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
