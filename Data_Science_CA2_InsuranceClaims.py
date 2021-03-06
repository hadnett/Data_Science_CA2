#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Hadnett
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# =============================================================================
# Exploratiry Data Analysis - STEP 1 - Data Import
# =============================================================================

import os
os.chdir('****Insert File Path*****')

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
#a strong correlation between Age and YearsHealthInsurance. This will have to be
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
# Regression Modelling - STEP 6 - Characterise & Drop Variables 
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
# Regression Modelling - STEP 7 - Construct New Variables 
# =============================================================================

#Smoker - Already Converted to Numerical Step 5 Bivariate Analysis.

#Gender
data['gender_num']= np.where(data.Gender=='male',1,0)

#Region - 3 Variables required.
data['southeast_num']=np.where(data.Region =="southeast",1,0)
data['southwest_num']=np.where(data.Region =="southwest",1,0)
data['northeast_num']=np.where(data.Region =="northeast",1,0)

# =============================================================================
# Regression Modelling - STEP 8 - Support Construction of Model
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
# 3       YearsHealthInsurance      0.3 #Removed due to close correlation with Age
# 4       BMI                       0.18
# 5       Children                  0.078
# 6       southeast_num             0.067
# 7       gender_num                0.056
# 8       southwest_num            -0.036
# 9       northeast_num             0.0058

# =============================================================================
# Regression Modelling - STEP 9 - Multicolinear Data
# =============================================================================

data[['Age','YearsHealthInsurance']].corr()

#Age and YearsHealthInsurance have a correlation of 0.99, suggesting they
#are multicolinear. I have decided to drop YearsHealthInsurance as a persons
#age typically affects their health and this in turn will result in more
#claims against their Health Insurance Company.

data.drop('YearsHealthInsurance', axis = 1, inplace = True)

# =============================================================================
# Regression Modelling - STEP 10 - Split Train and Test Data
# =============================================================================

#x: Predictors
x = data[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num', 'gender_num', 'southwest_num', 'northeast_num']]
#y: Response
y = data['TotalClaims'] #Pandas series

from sklearn.model_selection import train_test_split

#split train 66.7%, test 33.3%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.333)

y_train
x_train

# =============================================================================
# Regression Modelling - STEP 11 - Develop Linear Regression Model
# =============================================================================

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

#First add smoker_rate to model
model1.fit(x_train[['smoker_rate']], y_train)

print(model1.coef_)
print(model1.intercept_)
#So TotalClaims = 8248.47 + 23661.170821*smoker_rate

Output = pd.DataFrame(model1.coef_, ['Smoker'], columns = ['Coeff'])

predictions_train = model1.predict(x_train[['smoker_rate']])

raw_sum_sq_errors = sum((y_train.mean() - y_train)**2) # 124509375920.14389
prediction_sum_sq_errors = sum((predictions_train - y_train)**2) # 45058994169.77169

Rsquared1 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N= data.TotalClaims.count() #1317
p=1 # one predictor used
Rsquared_adj1 = 1 - (1-Rsquared1)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate: "+str(Rsquared1)) 
#Rquared = 0.63810 Explained about 63.8% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate: "+str(Rsquared_adj1)) #0.6378

####### Model 2 add Age variable #######
model2 = LinearRegression()

model2.fit(x_train[['smoker_rate', 'Age']], y_train)

print(model2.coef_)
print(model2.intercept_)
#So TotalClaims = -2034.76 + 23773.97*smoker_rate + 262.71*Age

Output = pd.DataFrame(model2.coef_, ['Smoker', 'Age'], columns = ['Coeff'])

predictions_train = model2.predict(x_train[['smoker_rate', 'Age']])

raw_sum_sq_errors = sum((y_train.mean() - y_train)**2) # 124509375920.14389
prediction_sum_sq_errors = sum((predictions_train - y_train)**2) # 32825068905.40724

Rsquared2 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N= data.TotalClaims.count() #1317
p=2 
Rsquared_adj2 = 1 - (1-Rsquared2)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate & Age: "+str(Rsquared2)) 
#0.7363 Explained about 73.6% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate & Age: "+str(Rsquared_adj2)) 
#0.7359 - improvement. Age add's further value to model.

####### Model 3 add BMI variable #######
model3 = LinearRegression()

model3.fit(x_train[['smoker_rate', 'Age', 'BMI']], y_train)

print(model3.coef_)
print(model3.intercept_)
#So TotalClaims = -11306.08 + 23854.48*smoker_rate + 249.02*Age + 317.61*BMI

Output = pd.DataFrame(model3.coef_, ['Smoker', 'Age', 'BMI'], columns = ['Coeff'])

#Generate predictions for the train data
predictions_train = model3.predict(x_train[['smoker_rate', 'Age', 'BMI']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2) # 29620581307.7543

Rsquared3 = 1-prediction_sum_sq_errors/raw_sum_sq_errors 

N= data.TotalClaims.count() #1317
p=3 
Rsquared_adj3 = 1 - (1-Rsquared3)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate, Age and BMI: "+str(Rsquared3)) 
#0.762 Explained about 76.2% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate, Age and BMI: "+str(Rsquared_adj3)) 
#0.761 - improvement. BMI add's further value to model.

####### Model 4 add Children variable #######
model4 = LinearRegression()

model4.fit(x_train[['smoker_rate', 'Age', 'BMI', 'Children']], y_train)

print(model4.coef_)
print(model4.intercept_)
#So TotalClaims = -11755.38 + 23836.89*smoker_rate + 247.60*Age + 317.88*BMI + 464.23*Children

Output = pd.DataFrame(model4.coef_, ['Smoker', 'Age', 'BMI', 'Children'], columns = ['Coeff'])

#Generate predictions for the train data
predictions_train = model4.predict(x_train[['smoker_rate', 'Age', 'BMI', 'Children']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2) # 29348839102.758663

Rsquared4 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N= data.TotalClaims.count() #1317
p=4 
Rsquared_adj4 = 1 - (1-Rsquared4)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate, Age, BMI and Children: "+str(Rsquared4)) 
#0.764 Explained about 76.4% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate, Age, BMI & Children: "+str(Rsquared_adj4)) 
#0.763 - improvement. Children adds value to the model.

####### Model 5 add southeast_num variable #######
model5 = LinearRegression()

model5.fit(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num']], y_train)

print(model5.coef_)
print(model5.intercept_)
#So TotalClaims = -11988.41 + 23890.70*smoker_rate + 246.42*Age + 334.90*BMI + 449.47*Children - 858.21*southeast_num  

Output = pd.DataFrame(model5.coef_, ['Smoker', 'Age', 'BMI', 'Children', 'southeast_num'], columns = ['Coeff'])

#Generate predictions for the train data
predictions_train = model5.predict(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2) #29228284279.2455

Rsquared5 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N= data.TotalClaims.count() #1317
p=5 
Rsquared_adj5 = 1 - (1-Rsquared5)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate, Age, BMI, Children & southeast_num: "+str(Rsquared5)) 
#0.765 Explained about 76.5% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate, Age, BMI, Children & southeast_num: "+str(Rsquared_adj5)) 
#0.764 - improvement. Southeast Region adds value to the model.

####### Model 6 add gender_num variable #######
model6 = LinearRegression()

model6.fit(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num', 'gender_num']], y_train)

print(model6.coef_)
print(model6.intercept_)
#So TotalClaims = -11948.87 + 23896.25*smoker_rate + 246.28*Age + 335.08*BMI + 449.60*Children -857.73*southeast_num -79.56*gender_num

Output = pd.DataFrame(model6.coef_, ['Smoker', 'Age', 'BMI', 'Children', 'southeast_num', 'gender_num'], columns = ['Coeff'])

#Generate predictions for the train data
predictions_train = model6.predict(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num', 'gender_num']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2) #29226904473.65604

Rsquared6 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N= data.TotalClaims.count() #1317
p=6 
Rsquared_adj6 = 1 - (1-Rsquared6)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate, Age, BMI, Children, southeast_num & gender_num: "+str(Rsquared6)) 
#0.765 Explained about 76.5% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate, Age, BMI, Children, southeast_num, gender_num: "+str(Rsquared_adj6)) 
#0.764188 - Reduction - Due to this reduction gender_num will be exlcuded from future models.

####### Model 7 add southwest_num variable #######
model7 = LinearRegression()

model7.fit(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num','southwest_num']], y_train)

print(model7.coef_)
print(model7.intercept_)
#So TotalClaims = -11894.02 + 23886.50*smoker_rate + 246.81*Age + 338.26*BMI + 449.36*Children -1078.59*southeast_num  -605.68*southwest_num

Output = pd.DataFrame(model7.coef_, ['Smoker', 'Age', 'BMI', 'Children', 'southeast_num', 'southwest_num'], columns = ['Coeff'])

#Generate predictions for the train data
predictions_train = model7.predict(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num', 'southwest_num']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2) #29176285900.884792

Rsquared7 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N= data.TotalClaims.count() #1317
p=6
Rsquared_adj7 = 1 - (1-Rsquared7)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate, Age, BMI, Children, southeast_num & southwest_num: "+str(Rsquared7)) 
#0.765 Explained about 76.5% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate, Age, BMI, Children, southeast_num & southwest_num : "+str(Rsquared_adj7)) 
#0.7645 - improvement. Southwest Region adds value to the model.

####### Model 8 add northeast_num variable #######
model8 = LinearRegression()

model8.fit(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num', 'southwest_num', 'northeast_num']], y_train)

print(model8.coef_)
print(model8.intercept_)
#So TotalClaims = -12146.35 + 23887.38*smoker_rate + 246.85*Age + 337.61*BMI + 455.26*Children -812.40*southeast_num  -342.03*southwest_num + 520.82*northeast_num 

Output = pd.DataFrame(model8.coef_, ['Smoker', 'Age', 'BMI', 'Children', 'southeast_num', 'southwest_num', 'northeast_num'], columns = ['Coeff'])

#Generate predictions for the train data
predictions_train = model8.predict(x_train[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num', 'southwest_num', 'northeast_num']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2) #29148269354.463562
Rsquared8 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N= data.TotalClaims.count() #1317
p=7
Rsquared_adj8 = 1 - (1-Rsquared8)*(N-1)/(N-p-1)
print("Rsquared Regression Model with Smoker Rate, Age, BMI, Children, southeast_num, southwest_num & northeast_num: "+str(Rsquared8)) 
#0.765 Explained about 76.5% of variation
print("Rsquared Adjusted Regression Model with Smoker Rate, Age, BMI, Children, southeast_num, southwest_num & northeast_num: "+str(Rsquared_adj8)) 
#0.7646 - improvement. Northeast Region adds value to the model.

# DECISION ------ Model 8 as proven by the rsquared adjusted value directly above.
#TotalClaims = -12146.35 + 23887.38*smoker_rate + 246.85*Age + 337.61*BMI + 455.26*Children -812.40*southeast_num  -342.03*southwest_num + 520.82*northeast_num 

# =============================================================================
# Regression Modelling - STEP 12 - Evaluating The Model Produced
# =============================================================================

predictions_test = model8.predict(x_test[['smoker_rate', 'Age', 'BMI', 'Children', 'southeast_num', 'southwest_num', 'northeast_num']])

Prediction_test_MAE = sum(abs(predictions_test - y_test))/len(y_test)
Prediction_test_MAPE = sum(abs((predictions_test - y_test)/y_test))/len(y_test)
Prediction_test_RMSE = (sum((predictions_test - y_test)**2)/len(y_test))**0.5

print(Prediction_test_MAE) #4101.619
print(Prediction_test_MAPE) #0.3918
print(Prediction_test_RMSE) #6068.646


#We can see from the scatter graph below that the regression model chosen (Model 8) preforms 
#relatively well. However, it appears to experience issues when making predictions between
#approximately 15k and 35k. It can predict total claims < 15k relatively well and is very good at
#predicting total claims over 35k. I believe there are a number of reasons for this. Firstly
#I believe it can predict claims under 15k with more accuracy because 73% of the total claims
#in this dataset are under 15k. Though, this theory is null when we review the predictions
#over 35k. However, upon seeing this graph I furthered my analysis to discover that 97% of
#people who had total claims greater than 35k smoke. Since smoker_rate is one of this models
#strongest correlation coefficients it may explain why predictions are so good over 35k.  
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test)
#Discovered how to plot trend line on scatter plot:
#https://www.kite.com/python/answers/how-to-plot-a-linear-regression-line-on-a-scatter-plot-in-python
m, b = np.polyfit(y_test, predictions_test, 1)
plt.plot(y_test, m*y_test+b, 'red')
plt.title("Predictions v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.show() 

### Further Analysis to explain Anomalies in model predictions ######
#73% of claims are below 15k
totalClaimsUnder15 = data[data.TotalClaims < 15000].TotalClaims.count()
print("Percentage of total claims under 15k: ", (totalClaimsUnder15/data.TotalClaims.count()) * 100)

#97% of people with total claims over 35k smoke.
smokersOver35k = data[(data.TotalClaims > 35000) & (data.smoker_rate == 1)].TotalClaims.count()
totalOver35k = data[data.TotalClaims > 35000].TotalClaims.count()
print("Percentage of Smokers over 35k: ", (smokersOver35k/totalOver35k) * 100)

#Only 17% of the dataset is between 15k and 35k.
totalClaimsBetween15and35k = data[(data.TotalClaims >= 15000) & (data.TotalClaims <= 35000)].TotalClaims.count()
print("Percentage of total claims under 15k: ", (totalClaimsBetween15and35k/data.TotalClaims.count()) * 100)

#60% of people with total claims between 15k and 35k Smoke
smokersBetween15and35 = data[(data.TotalClaims >= 15000) & (data.TotalClaims <= 35000) & (data.smoker_rate == 1)].TotalClaims.count()
totalBetween15and35 = data[(data.TotalClaims >= 15000) & (data.TotalClaims <= 35000)].TotalClaims.count()
print("Percentage of Smokers over 35k: ", (smokersBetween15and35/totalBetween15and35) * 100)

#######################################################################
#This percentage error graph shows model 8 has a maximum positive error of approximately +3.3%
#maximum negative error of approximately -2%. However, the majority of predictions where within 
#the ±1% range. If we take the highest total claim within this dataset (51194.56) the predicted
#value will likely be within the ±1% range(51194.56 * 1% = 511.94). I believe a prediction with this level of
#accuracy should allow an insurance company to make an accurate decision regarding what premiums to
#charge there customers. It is then a business decision as to whether or not some level of security
#should be built into this premium to compensate for the models anomalies. It is also worth noting 
#that the majority of predictions recieved a percentage error of 0 or above. Therefore, this model 
#seems to predict slightly higher than the actual total claims in most cases. 

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, (predictions_test - y_test)/y_test) 
plt.title("Percentage Errors v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Error Values")
plt.show()

# =============================================================================
# Additional Analysis  - STEP 13 - Additional Analysis
# =============================================================================
#I discoverd MPLClassifier after reviewing Dr Kevin McDaids code from the titanicAnalysisv1
#I then done my own research to discover MPLRegressor can be used for regression neural networks.
#MLPRegressor is trained using backpropagation and implements multi-layer percpetrions and 
#my learning/understanding of these concepts was supported by Joel Grus Data Science from
#Scratch as well as other articles and videos. 
#https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
model = MLPRegressor()

@ignore_warnings(category=ConvergenceWarning)
def findBestParams(): 
    print("Checking for the optimal hyperparameters this may take several minutes...")
    #GridSearchCV is imported from sklearn and it is used to run a brute force search to 
    #determine the best parameters to use for model in this case my neural network. The 
    #check_parameters json is populated with possible parameters and these parameters and
    #run against the training data using the fit method. fit() returns the best parameters
    #for the model in a key value pair. 
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    check_parameters = {
        'hidden_layer_sizes': [(100,)],
        'activation': ['identity', 'relu'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant'],
        'learning_rate_init': [0.001, 0.05]
    }

    gridsearchcv = GridSearchCV(model, check_parameters, cv=3)
    return gridsearchcv.fit(x_train, y_train)

gridsearchcv = findBestParams()

#The best parameters for the model.
print(gridsearchcv.best_params_)
#{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'learning_rate_init': 0.05, 'solver': 'lbfgs'}

#I incorprated these parameters into the model below so that each time this script is run the best
#parameters for the training data are selected.
model1 = MLPRegressor(hidden_layer_sizes=gridsearchcv.best_params_['hidden_layer_sizes'], activation=gridsearchcv.best_params_['activation'], alpha=gridsearchcv.best_params_['alpha'], 
                      learning_rate=gridsearchcv.best_params_['learning_rate'], solver=gridsearchcv.best_params_['solver'], 
                      max_iter=3000, learning_rate_init=gridsearchcv.best_params_['learning_rate_init'])

model1.fit(x_train, y_train)

predictions = model1.predict(x_train)

#Discovered a sklearn library that includes functions for working out the metrics
#of the model produced.
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


MSE = mean_squared_error(y_train, predictions) # 21057749.042644326

Rsquared = r2_score(y_train, predictions) #0.84198
#Explained 84.2% of variation. 7.7% more than regression model 8.

prediction_sum_sq_errors = sum((predictions - y_train)**2) 

########## Evaluate New Model ############
predictions_test_NM = model1.predict(x_test)

#Return the coefficient of determination R^2 of the prediction.
model1.score(x_test, y_test)

Prediction_test_MAE = sum(abs(predictions_test_NM - y_test))/len(y_test)
Prediction_test_MAPE = sum(abs((predictions_test_NM - y_test)/y_test))/len(y_test)
Prediction_test_RMSE = (sum((predictions_test_NM - y_test)**2)/len(y_test))**0.5

print(Prediction_test_MAE) #2769.3231535905315 - Mean Absolute Error decreased by 1332.296 from model 8.
print(Prediction_test_MAPE) #0.29962781836477104 - 9% better than model 8.
print(Prediction_test_RMSE) #4273.943251422306 - Root mean sqaured error decreased by 1794.703 from model 8.

#We can see from the scatter plot below that the neural network produced preforms relatively
#well. We can see that there is a major increase in accuracy between 0 and 15k approximately
#with most predicts occuring on or very close to the trend line. Again this model experiences
#issues between 15k and 35k. Suggesting to me that there is something different about
#the data between these two points. One theory is outlined in describe of model 8's scatter
#plot. However, with that being said the predictions this model made in this range were much closer to the
#trend line, proving that this model has a greater prediction accuracy even when the predictions
#are not correct. 
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test_NM)
m, b = np.polyfit(y_test, predictions_test_NM, 1)
plt.plot(y_test, m*y_test+b, 'red')
plt.title("Predictions v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.show()

#In this graph we can see that the maximum positive percentage error is just below 0.75%
#and the maximum negative percentage error is approximately -1%. This graph again proves
#that this model has a greater level of accuracy than the previous model (model 8). It 
#is also worth noting that the majority of the points in this graph are 0 or above. 
#Therefore, this tells us that the model is more likely to over predict the total claims
#rather than under predict, offering the security to the insurance company. It again
#is a business decision as to whether or not to accept the models prediction to lower
#the prediction slightly to remain competive when providing quotes for premiums. 
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, (predictions_test_NM - y_test)/y_test) 
plt.title("Percentage Errors v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Percentage Error Values")
plt.show()

# =============================================================================
# Additional Analysis  - Final Thoughts
# =============================================================================
'''
It is apparent that neither of the models produced are prefect and both seem to encounter
an issue when dealing with values between 15k and 35k approximately. However, while both
models have flaws I feel that they are both capable of predicting total claims to a level
of accuracy that will allow an insurance company to provide an accurate quote to a customer.
I feel that I have proved the best model for predicting total claims is the neural network
model. By using the neural network model we can achieve a greater Rsquared value of 84.2% which is 7.7%
more than model 8 and the mean absolute percentage error is decreased by 9%. However, even with
this greater level of accuracy the neural network still does produce inacurrate predictions but 
these inaccurate predictions are still more accurate than the inaccurate predictions made 
by model 8. 
'''

