## Time series modelling to predict weekly meal boxes
## Urvashi Tuteja
## 27th Sep 2019

##################### Import required libraries #####################
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.regression.linear_model as smf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from math import sqrt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools


##################### Fetch data #####################
## Define the path where input and output data is (to be modified in case change of data/path)
path="C:/Users/tuteja urvashi/Desktop/HelloF/"
filename="HF_SeniorDS-DemandForecasting_Challenge_Data.csv"

## Read the data
data=pd.read_csv(path+filename)


##################### Exploratory data Analysis #####################
## Perform basic checks
data.describe()
data.isnull().sum()

## Plot the series against each of active customer base to see correlation
series=data.boxes
series.plot()
plt.show()

plt.plot(data.iso_week,data.boxes,'r', label='Boxes')
plt.plot(data.iso_week,data.active1,'b', label='Active1')
plt.plot(data.iso_week,data.active2,'g', label='Active2')
plt.plot(data.iso_week,data.active3, label='Active3')
plt.plot(data.iso_week,data.active4, label='Active4')
plt.show()


##################### Model 1 - Multi Variable Analysis #####################
## Define training data
df=data[(data.boxes.notnull())]

##Train Test split
train, test = train_test_split(df, test_size=0.25, random_state=40)
X_train=train[['active1','active2','active3', 'active4']]
X_test=test[['active1','active2','active3', 'active4']]
Y_train=train[['boxes']]
Y_test=test[['boxes']]

## Define the data for which prediction is needed
output= data.loc[data.boxes.isnull(),['active1','active2','active3', 'active4']]

## Fitting lasso Regression
res=Lasso(alpha=0.01).fit(X_train,Y_train)
print(res.coef_,res.intercept_)
y_train_pred=res.predict(X_train)
y_test_pred=res.predict(X_test)

## Find the goodness of fit on train data
print('RMSE:',np.sqrt(mean_squared_error(Y_train,y_train_pred)))
r2=r2_score(Y_train,y_train_pred)
print('R2:',r2)
print('Adj R2:', 1-((1-r2)*(Y_train.shape[0]-1)/(Y_train.shape[0]-4-1)))

## Find the goodness of fit on test data
print(np.sqrt(mean_squared_error(Y_test,y_test_pred)))
print(r2_score(Y_test,y_test_pred))

## Predict for the next 12 weeks
y_new_pred= res.predict(output)

##Add week numbers and save the output
output_forecast=data.loc[data.boxes.isnull(),['iso_week']]
output_forecast['boxes']=np.round(y_new_pred)

## Save the output
output_forecast.to_csv(path + 'Output Forecast (Lasso).csv', index=False,sep=';')


##################### Model 2 - Time series modelling #####################

## Convert isoweeks to datetime format and set the time series as index
data['iso_week1']= pd.to_datetime(data.iso_week.str.replace("W","") + '0',format='%Y-%W%w')
data1=data[(data.boxes.notnull())].set_index('iso_week1')
predict_weeks= data[(data.boxes.isnull())].set_index('iso_week1')

# Define y
y=data1[['boxes']]

## Decompose the time series into trend, seasonality and residual
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

## Forecast using seasonal ARIMAX
p = d = q = range(0, 2) # using combination of p,d,q values to find the optimal result
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]


## Run a loop for diff values of p,d,q and store in a table
results_sarimax=pd.DataFrame(columns={'pdq','seasonal_order','AIC'})
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,
                                            enforce_stationarity=False,enforce_invertibility=False).fit()
            res_pdq = pd.DataFrame({'pdq':format(param),'seasonal_order':format(param_seasonal),'AIC':mod.aic},index=[0]).reset_index(drop=True)
            results_sarimax=results_sarimax.append(res_pdq)
        except:
            continue

##find the pdq with least AIC
best_fit=results_sarimax.pdq[results_sarimax.AIC.min()]