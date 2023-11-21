  
'''

REGRESSION :- 1] SIMPLE LINEAR REGRESSION :
    
    
 phases of crisp ML-Q (cross industry standard process for machine learning with quality assurance):
     
 1] data understanding and business understanding 
 2] data preparation(data cleaning)
 3] model building (data mining)    
 4] model evaluation
 5] model deployment
 6] Monitoring and Maintainance      
   
    
    
Problem Statement: -

1.	A logistics company recorded the time taken for delivery and the time taken for the sorting of the items 
    for delivery. Build a Simple Linear Regression model to find the relationship between delivery time and 
    sorting time with the delivery time as the target variable. Apply necessary transformations and record 
    the RMSE and correlation coefficient values for different models.


1] step : data understanding and business understanding: 

business objectives  : Improve the delivery operation through predictive modelling.
business constraints : Reduce the time for delivery.
    

Success Criteria:-
    
Business success criteria        : Reduction in delivery times or improved delivery time predictability.
Machine Learning success criteria: R^2 (coefficient determination) have good accuracy near by 1.to reach the higher R^2 value near by 1. then our model is best fit.
Economic success criteria        : increased revenue, Improved delivery efficiency to leads increase in revenue
    

Data Collection:
    
dataset is collected from logistic industry. 
dataset contain 21 rows which contain different different delivery times and sorting times for taking each product.  
    

Data description:

delivery time : this is the target(Y) variable or dependent variable to build regression model which contain delivery time taken for each product.
sorting time : this is independent variable or (x) variable to build regression model which contain time taken for sorting of product before they deliver.

'''



'''
2] step : data preprocessing (data cleaning) :
'''        

import pandas as pd

slr = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\ASSIGNMENTS\SUPERVISED LEARNING\REGRESSION\1 SIMPLE LINEAR\delivery_time.csv" ) 

# MySQL Database connection
# Creating engine which connect to MySQL
user = 'user1' # user name
pw = 'user1' # password
db = 'slr_db' # database

from sqlalchemy import create_engine
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
slr.to_sql('logistic', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from logistic'
df = pd.read_sql_query(sql, con = engine) 


df.shape
df.dtypes
df.info()
df.describe()
df.duplicated().sum() # not any duplicated rows are present in our dataset.

# outliers treatment : 
# i check any outliers are present or not here .
import seaborn as sns
sns.boxplot(df) 
# another method
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# not any outliers are present in our target features and input features.

# sort our values in ascending order
df.head 
df.sort_values('Delivery Time', ascending = True, inplace = True)
df.head
# reset index in ascending order for our values
df.reset_index(inplace = True, drop = True)
df.head






# separate data into independent variable (X) and Dependent variable (Y) 

X = pd.DataFrame(df['Sorting Time'])    # independent variable (X), input variable
Y = pd.DataFrame(df['Delivery Time'])   # Dependent variable (Y), target variable, output 


# Steps for linear regression :-

# 1st step :   
# Bivariate Analysis : Scatter plot 
## Measure the strength of the relationship between two variables using Correlation coefficient.

import matplotlib.pyplot as plt
plt.scatter(x = df['Sorting Time'], y = df['Delivery Time']) 

import numpy as np 
np.corrcoef(df['Sorting Time'], df['Delivery Time'] )

# we have another method
df.corr()
# for visualization
dataplot = sns.heatmap(df.corr(), annot = True, cmap = "YlGnBu") 

# covariance :
df.cov()



#------------------------------------------------------------------------------------------------------

# 2nd step : OLS method is used to finding best fit line on scatterplot 
#            OLS = ORDINAL LEAST SQUARED     

import statsmodels.formula.api as smf 
# # Linear Regression using statsmodels package
# Simple Linear Regression 


# Replace spaces in column names with underscores (if needed)
df.columns = df.columns.str.replace(' ', '_') 

model = smf.ols('Delivery_Time ~ Sorting_Time', data=df).fit() 
model.summary()
pred1 = model.predict(pd.DataFrame(df['Sorting_Time']))
pred1


# Regression Line
plt.scatter(df.Sorting_Time, df.Delivery_Time)
plt.plot(df.Sorting_Time, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

#------------------------------------------------------------------------------------------------------
# 3rd step :
# Error calculation (error = AV - PV)

res1 = df.Delivery_Time - pred1 
print(np.mean(res1))  # error value is not a negative, so we want positive value  

# here we get a square value of negative error then calculate square root and we get actual error.
res_sqr1 = res1 * res1 
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 


# ============================================================================
# 4th step :
# choose best model :  

# 1] Log Transformation : x = log(Sorting_Time); y = Delivery_Time
# 2] Exponential transformation : x = Sorting_Time; y = log(Delivery_Time) 
# 3] Polynomial transformation : x = Sorting Time; x^2 = Sorting Time*Sorting Time; y = log(Delivery Time)





# # Model Tunning with Transformations
# ## Log Transformation
# x = log(Sorting_Time); y = Delivery_Time
import matplotlib.pyplot as plt

plt.scatter(x = np.log(df['Sorting_Time']), y = df['Delivery_Time'], color = 'brown')
np.corrcoef(np.log(df.Sorting_Time), df.Delivery_Time)  #correlation

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = df).fit() 
model2.summary()
pred2 = model2.predict(pd.DataFrame(df['Sorting_Time'])) 
pred2 




# Regression Line
plt.scatter(np.log(df.Sorting_Time), df.Delivery_Time) 
plt.plot(np.log(df.Sorting_Time), pred2, "r") 
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res2 = df.Delivery_Time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2





# ## Exponential transformation
# x = Sorting_Time; y = log(Delivery_Time) 

plt.scatter(x = df['Sorting_Time'], y = np.log(df['Delivery_Time']), color = 'orange')
np.corrcoef(df.Sorting_Time, np.log(df.Delivery_Time))      #correlation
model3 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data = df).fit()
model3.summary()
pred3 = model3.predict(pd.DataFrame(df['Sorting_Time'])) 
pred3  


# Regression Line
plt.scatter(df.Sorting_Time, np.log(df.Delivery_Time))
plt.plot(df.Sorting_Time, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

pred3_at = np.exp(pred3) 
print(pred3_at)

res3 = df.Delivery_Time - pred3
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3



# ## Polynomial transformation 
# x = Sorting Time; x^2 = Sorting Time*Sorting Time; y = log(Delivery Time)

model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time * Sorting_Time)', data = df).fit()
model4.summary()
pred4 = model4.predict(pd.DataFrame(df))
print(pred4)


# X = pd.DataFrame(df['Sorting Time'])    # independent variable (X), input variable
# Y = pd.DataFrame(df['Delivery Time'])   # Dependent variable (Y), target variable, output 


plt.scatter(X['Sorting Time'], np.log(Y['Delivery Time'])) 
plt.plot(X['Sorting Time'], pred4, color = 'red')
plt.plot(X['Sorting Time'], pred3, color = 'green', label = 'linear')
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])
plt.show()


pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = df.Delivery_Time - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4



# ### Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)

table_rmse




'''
3] step : model building (data mining)
    
'''


# Evaluate the best model 
# Data Split 

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, random_state = 0)

plt.scatter(train.Sorting_Time, np.log(train.Delivery_Time))
plt.figure(2)
plt.scatter(test.Sorting_Time, np.log(test.Delivery_Time)) 

# Fit the best model on train data
finalmodel = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = train).fit() 


# Predict on test data
test_pred = finalmodel.predict(test)
pred_test_sort = np.exp(test_pred)

# Model Evaluation on Test data
test_res = test.Delivery_Time - pred_test_sort 
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse      # this is the accuracy for testing data 





# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train)) 
pred_train_delivery = np.exp(train_pred)
pred_train_delivery

# Model Evaluation on train data
train_res = train.Delivery_Time - pred_train_delivery
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse         # this is the accuracy for training data 



# save the best model
import pickle 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_model = (PolynomialFeatures(degree = 2), LinearRegression()) 
pickle.dump(poly_model, open('poly_model.pkl', 'wb')) 


'''
SOLUTION :   

    here we can say, reduce the delivery time on the basis of sorting time.
    using simple Linear regression we can analyse linear relationship between sorting time and delivery time 
    how sorting time affected to the delivery time and we reach the solution to get minimize sorting time and 
    go to the customer very fastly for deliver the product.


'''


















