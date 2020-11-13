#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:24:47 2020

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
