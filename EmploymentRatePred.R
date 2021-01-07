library(forecast)
library(urca)
library(ggplot2)
library(vars)
setwd("/Users/shubhamsmac/Desktop/R_ECON")
employ<-read.csv("PAYNSA.csv")

employee=ts(employ[,2], start = c(1960,1), frequency = 12, end = c(2020,10)) #January 1960 - October 2020 (end of entire dataset)
employee_holdout = ts(employ[,2], start = c(1960,1), end = c(2020,6), frequency = 12) #January 1960 - June 2020
plot(employee_holdout)
plot(log(employee_holdout))
#Discussion - Plotted a logarithmic transformation since there is a non-linear trend and there is 
#changes in the volatility of the sample data.
#Also allows our forecasting values to be non-negative and have the same scale as original dataset
#which for our data is a scale of per 1000 people.

##Box-Jenkins
#1). Check for stationarity
#     Trend? Yes
#     Non-zero constant?
#We do not think data is from a covariance stationary model due to the trend 
#and seasonality present in the time series; there is predictability present in the data
# H0: alpha=0
# HA: alpha<0
#We used an ADF unit root test (transformation) to see if we needed to difference the data.
summary(ur.df(log(employee_holdout), lags=37, type="trend", selectlags="AIC"))
#We fail to reject the null hypothesis that there is a unit root, so we 
#believe a unit root is present in this time series:
#non-seasonal unit root (2)
summary(ur.df(diff(log(employee_holdout)), type = "drift", lags=20, selectlags="AIC"))
summary(ur.df(diff(diff(log(employee_holdout)), type = "drift", lags=20, selectlags="AIC")))
ndiffs(log(employee_holdout))
#seasonal unit root (1)
summary(ur.df(diff(diff(log(employee_holdout)),12), type = "drift", lags=20, selectlags="AIC"))
nsdiffs(log(employee_holdout))
#We reject the null that there is a seasonal unit root, 
#so we have evidence that the seasonally differenced data is stationary

ggtsdisplay(diff(diff(log(employee_holdout)),12), lag.max=60)
# i. Seasonality - There is seasonality present in the data
# ii. Deterministic trends - There is trend because the model appears to be steadily increasing for the time series
# iii. Other variation (anything other than trend/seasonality) 
#        - For the ACF, some statistically significant lags with a dominant spike at lag 1, indicating at least a non-seasonal MA (1) piece 
#         Significant seasonal coefficient at lag 12 indicating a seasonal MA (1) piece and remaining lags decaying theoretically non-zero
#        - For the PACF, Some significant lags mainly with the first 4 lags dominant, indicating a non-seasonal AR (2) or (3) piece is possible
#         There is a significant seasonal coefficient at lag 12 but other lags are decaying theoretically non-zero


ggtsdisplay(diff(diff(log(employee_holdout),12)), lag.max=60)
#Appropriate models have to be a SARIMA(p,1,q)x(P,1,Q) with 
#a non-seasonal d=1 and seasonal D=1
#Initial Guess
#p = 1,2,3  
#q = 0,1,2
#P = 0,1,2
#Q = 0,1,2
#lets see our initial guesses for candidate models
model_first = Arima(log(employee_holdout), order = c(1,2,2), seasonal=c(1,1,1))
#AIC=-6884.73   AICc=-6884.64   BIC=-6855.5
model_sec=Arima(log(employee_holdout), order = c(2,2,2), seasonal=c(1,1,2))
#AIC=-6884.24   AICc=-6884.09   BIC=-6845.27
model_3=Arima(log(employee_holdout), order = c(3,2,3), seasonal=c(1,1,2))
#AIC=-6912.97   AICc=-6912.74   BIC=-6864.26
model_4=Arima(log(employee_holdout), order = c(4,2,3), seasonal=c(1,1,2))
#AIC=-6901.68   AICc=-6901.41   BIC=-6848.1
model_5 = Arima(log(employee_holdout), order = c(4,2,4), seasonal=c(1,1,1))
#AIC=-6911.05   AICc=-6910.77   BIC=-6857.47
model_6 = Arima(log(employee_holdout), order = c(3,2,3), seasonal=c(1,1,1))
#AIC=-6913.17   AICc=-6912.98   BIC=-6869.33
#His answer: SARIMA(0,2,1)x(0,1,1) or SARIMA(0,2,1)x(1,1,0) models

# 
# diagnostic checks on our residuals. 
# Provide a plot of the correlogram of the residuals and describe why the fitted model is appropriate.
ggtsdisplay(model_6$residuals, lag.max = 60)
#We believe that model 6 is appropriate because the residuals for the sample PACF 
#are well within the confidence bands. For the sample ACF, the lags are also within the confidence bands. 
#The AIC is lowest out of all the models tested: -6913.17
qchisq(.95, 20)
Box.test(model_6$residuals, lag=20, type="Ljung-Box")
#The value from our Box-Ljung test come up as 4.3911 < 31.41043, which is less 
#than our chi-squared value for 20 lags at 95% of critical value. 
#Hence, I fail to reject that the first 20 autocorrelation coefficients are jointly equal to 0.
# 

Forecast1 = forecast(model_6, lambda = 0, biasadj = TRUE, h=4)
Forecast1
upper=ts(Forecast1$upper[,2],start=c(2020,7),frequency=12) 
lower=ts(Forecast1$lower[,2],start=c(2020,7),frequency=12) 
winDATA1=window(employee,start=c(2018,1))
plot(cbind(Forecast1$mean,winDATA1,upper,lower),plot.type="single",ylab="FORECAST",
     col=c("BLACK","BLUE","RED","RED"),
     lty=c("solid","solid","dotted","dotted"))
legend("bottomleft",legend=c("Forecast","Number of Employees","Upper","Lower"),
       col=c("BLACK","BLUE","RED", "RED"),lty=c("solid","solid","dotted","dotted"),
       text.font=2)
#It appears that the number of gainfully employed persons in the U.S. 
#is forecasted to be lower than what the current data has been showing. 
#Though people are getting hired especially during the holiday season,
#unemployment is still a major issue and will continue to be so into 2021 
#due to the pandemic.
# 

modelemployee=nnetar(employee_holdout)
modelemployee
modelneural=nnetar(employee_holdout, p=5, P=2, repeats=100)
z=forecast(modelneural, PI=TRUE, h=4)
#Comparing the accuracy between neural net method and ARIMA model #6:
#Neural net 
accuracy(z,window(employee, start=c(2020,7)))
#Arima
accuracy(Forecast1, window(employee, start=c(2020,7)))
#We select the ARIMA method as our model of choice for part h due to a low RMSE value
#for both the training and test sets
# 

model_6 = Arima(log(employee), order = c(3,2,3), seasonal=c(1,1,1))
Forecast1 = forecast(model_6, lambda = 0, biasadj = TRUE, h=6)
Forecast1
upper=ts(Forecast1$upper[,2],start=c(2020,11),frequency=12) 
lower=ts(Forecast1$lower[,2],start=c(2020,11),frequency=12) 
winDATA1=window(employee,start=c(2018,1)) #Number of Employees
plot(cbind(Forecast1$mean,winDATA1,upper,lower),plot.type="single",ylab="FORECAST",
     col=c("BLACK","BLUE","RED","RED"),
     lty=c("solid","solid","dotted","dotted"))
legend("bottomleft",legend=c("Forecast","Number of Employees","Upper","Lower"),
       col=c("BLACK","BLUE","RED", "RED"),lty=c("solid","solid","dotted","dotted"),
       text.font=2)
#For those predicting the employment rate for the next quarter, 
#it is not looking to be too promising; in fact, it seems that hiring
#will decrease going into the Spring months.
#COVID-19 cases are projected to rise in the United States
#even with the pending emergency use of a vaccine, so again, unemployment will
#be something to contend with until public health changes or upturn in the job market.