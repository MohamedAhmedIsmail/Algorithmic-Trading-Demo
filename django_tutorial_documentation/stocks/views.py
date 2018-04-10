from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import pandas as pd
import numpy as np
import math
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from my_site.settings import STOCKS_DATA_FOLDER
from sklearn import preprocessing,cross_validation,svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima_model import ARIMA
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from statsmodels.tsa.arima_model import ARIMA
global graph
from . import utility as util
from . import preprocessing as pp
def get_stock(request, stock_symbol):

    stock_file_name = util.get_stock_file_name(stock_symbol)
    stock_df = pp.get_stock_df(stock_file_name)

    chart_data = util.get_chart_data(stock_df)

    return JsonResponse(chart_data)

def vote(request, question_id):

    return HttpResponse("You're voting on question %s." % question_id)

class MadinetNasrModels:
	def index(request):
		return render(request,'stocks/index.html')
	def AboutPage(request):
		return render(request,'stocks/About.html')
	def ContactPage(request):
		return render(request,'stocks/Contact.html')
	def ClosePrice(request):
		madinet_Nasr_Housing=pd.read_csv(STOCKS_DATA_FOLDER + "Medinet Nasr Housing.csv")
		madinet_Nasr_Housing=madinet_Nasr_Housing[-500:]
		dates=[]
		for date in madinet_Nasr_Housing['TRADE_DATE'].values:
			dates.append(str(date)[:10])
		data={
			'labels':dates,
			'datasets':[{
				'data':madinet_Nasr_Housing['CLOSE_PRICE'].values.tolist(),
				'label':"CLOSE_PRICE",
				'borderColor':"red",
				'fill':'false'
			}]}
		return JsonResponse(data)


	def RegressionModels(request):
		madinet_Nasr_Housing=pd.read_csv(STOCKS_DATA_FOLDER + "Medinet Nasr Housing.csv")
		madinet_Nasr_Housing=madinet_Nasr_Housing[-500:]
		dates=[]
		for date in madinet_Nasr_Housing['TRADE_DATE'].values:
			dates.append(str(date)[:10])
		madinet_Nasr_Housing['EWMA_12']=madinet_Nasr_Housing['CLOSE_PRICE'].ewm(span=12).mean()
		madinet_Nasr_Housing['HL_PCT']=(madinet_Nasr_Housing['HIGH_PRICE']-madinet_Nasr_Housing['LOW_PRICE'])/madinet_Nasr_Housing['LOW_PRICE']*100.0
		madinet_Nasr_Housing['PCT_change']=(madinet_Nasr_Housing['CLOSE_PRICE']-madinet_Nasr_Housing['OPEN_PRICE'])/madinet_Nasr_Housing['OPEN_PRICE']*100.0
		madinet_Nasr_Housing.dropna(inplace=True)
		madinet_Nasr_Housing['CLOSE_PRICE'].shift(-3).tail()
		X=madinet_Nasr_Housing[['EWMA_12','HL_PCT','PCT_change']].values
		y=madinet_Nasr_Housing['CLOSE_PRICE'].values
		X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,shuffle=False,random_state=1)
		sc=StandardScaler()
		sc.fit(X_train)
		X_train=sc.transform(X_train)
		X_test=sc.transform(X_test)
		lr=LinearRegression()
		KNN=KNeighborsRegressor()
		SVR=svm.SVR()
		lr.fit(X_train,y_train)
		KNN.fit(X_train,y_train)
		SVR.fit(X_train,y_train)
		y_pred_lr=lr.predict(X_test)
		y_pred_SVR = SVR.predict(X_test)
		y_pred_KNN = KNN.predict(X_test)
		madinet_Nasr_Housing['LR_pred']=None
		madinet_Nasr_Housing['KNN_pred']=None
		madinet_Nasr_Housing['SVR_pred']=None
		madinet_Nasr_Housing['LR_pred'].iloc[350:]=y_pred_lr
		madinet_Nasr_Housing['KNN_pred'].iloc[350:]=y_pred_KNN
		madinet_Nasr_Housing['SVR_pred'].iloc[350:]=y_pred_SVR
		data={
			'labels':dates,
			'datasets':[{
				'data':madinet_Nasr_Housing['SVR_pred'].values.tolist(),
				'label':"SVR_pred",
				'borderColor':"blue",
				'fill':'false'
			},
			{
				'data':madinet_Nasr_Housing['LR_pred'].values.tolist(),
				'label':"LR_pred",
				'borderColor':"green",
				'fill':'false'
			},
			{
				'data':madinet_Nasr_Housing['KNN_pred'].values.tolist(),
				'label':"KNN_pred",
				'borderColor':"orange",
				'fill':'false'
			},
			{
				'data':madinet_Nasr_Housing['CLOSE_PRICE'].values.tolist(),
				'label':"CLOSE_PRICE",
				'borderColor':"Red",
				'fill':'false'
			}]
		}
		return JsonResponse(data)

	def ForeCastModel(request):
		madinet_Nasr_Housing=pd.read_csv(STOCKS_DATA_FOLDER + "Medinet Nasr Housing.csv")
		dates=[]
		for date in madinet_Nasr_Housing['TRADE_DATE'].values:
			dates.append(str(date)[:10])
		madinet_Nasr_Housing['HL_PCT']=(madinet_Nasr_Housing['HIGH_PRICE']-madinet_Nasr_Housing['LOW_PRICE'])/madinet_Nasr_Housing['CLOSE_PRICE']*100.0
		madinet_Nasr_Housing['PCT_change']=(madinet_Nasr_Housing['CLOSE_PRICE']-madinet_Nasr_Housing['OPEN_PRICE'])/madinet_Nasr_Housing['OPEN_PRICE']*100.0
		madinet_Nasr_Housing=madinet_Nasr_Housing[['TRADE_VOLUME','CLOSE_PRICE','HL_PCT','PCT_change']]
		madinet_Nasr_Housing.fillna(value=-999999,inplace=True)
		forecast_out=int(math.ceil(0.041*len(madinet_Nasr_Housing)))
		madinet_Nasr_Housing['Label']=madinet_Nasr_Housing['CLOSE_PRICE'].shift(-forecast_out)
		X=np.array(madinet_Nasr_Housing.drop(['Label'],1))
		X=preprocessing.scale(X)
		X_lately=X[-forecast_out:]
		X=X[:-forecast_out]
		madinet_Nasr_Housing.dropna(inplace=True)
		y=np.array(madinet_Nasr_Housing['Label'])
		X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
		clf=LinearRegression()
		clf.fit(X_train,y_train)
		forecast_set=clf.predict(X_lately)
		y_pred_lr=clf.predict(X_test)
		madinet_Nasr_Housing['Forecast']=None
		last_date=madinet_Nasr_Housing.iloc[-1].name
		one_day=86400 #minutes
		ts = datetime.datetime.now().timestamp()
		last_day=ts
		next_day=last_day+one_day
		for i in forecast_set:
			next_date=datetime.datetime.fromtimestamp(next_day)
			next_day+=86400
			madinet_Nasr_Housing.loc[next_date]=[np.nan for j in range(len(madinet_Nasr_Housing.columns)-1)]+[i]
		#print(madinet_Nasr_Housing['CLOSE_PRICE'][:300])
		#print(madinet_Nasr_Housing['Forecast'][1170:])
		data={
			'labels':dates,
			'datasets':[{
				'data':madinet_Nasr_Housing['CLOSE_PRICE'][:300].values.tolist(),
				'label':"Real Price",
				'borderColor':"blue",
				'fill':'false'
			},
			{
				'data':madinet_Nasr_Housing['Forecast'][1170:].values.tolist(),
				'label':"Predict Price",
				'borderColor':"Red",
				'fill':'false'
			}]
		}
		return JsonResponse(data)

	def ArimaModel(request):
		
		series =pd.read_csv(STOCKS_DATA_FOLDER+"Medinet Nasr Housing.csv")
		X = series.iloc[:,-1].values
		size = int(len(X) * 0.66)
		train, test = X[0:size], X[size:len(X)]

		history = [x for x in train]
		predictions = list()

		for t in range(len(test)):
			model = ARIMA(history, order=(6,1,0))
			model_fit = model.fit(disp=0)
			output = model_fit.forecast()
			yhat = output[0]
			predictions.append(yhat)
			obs = test[t]
			history.append(obs)
			print('predicted=%f, expected=%f' % (yhat, obs))
		    
		#error = mean_squared_error(test, predictions)
		#print('Test MSE: %.3f' % error)
		data={
			'labels':predictions,
			'datasets':[{
				'data':predictions,
				'label':"Real Price",
				'borderColor':"blue",
				'fill':'false'
			},
			{
				'data':test.tolist(),
				'label':"Predict Price",
				'borderColor':"Red",
				'fill':'false'
			}]
			}
		return JsonResponse(data)

	def LSTMModel(request):
		training_set=pd.read_csv(STOCKS_DATA_FOLDER +"Medinet Nasr Housing.csv")
		dates=[]
		for date in training_set['TRADE_DATE'].values:
			dates.append(str(date)[350:])
		madinet_Nasr_Housing_training=training_set[:1198]
		training_set=madinet_Nasr_Housing_training.iloc[:,8:9].values
		sc=MinMaxScaler()
		training_set=sc.fit_transform(training_set)
		X_train=training_set[0:1197]
		y_train=training_set[1:1199]
		X_train=np.reshape(X_train,(1197,1,1))
		graph=tf.get_default_graph()
		with graph.as_default():
			regressor=Sequential()
			regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
			regressor.add(Dense(units=1))
			regressor.compile(optimizer='adam',loss='mean_squared_error')
			regressor.fit(X_train,y_train,batch_size=1,epochs=10)
		test_set=pd.read_csv(STOCKS_DATA_FOLDER + "Madinet Nasr Housing 2.csv")
		real_stock_price=test_set.iloc[:,8:9].values
		inputs=real_stock_price
		inputs=sc.transform(inputs)
		inputs=np.reshape(inputs,(21,1,1))
		predicted_stock_prices=regressor.predict(inputs)
		predicted_stock_prices=sc.inverse_transform(predicted_stock_prices)
		data={
			'labels':dates,
			'datasets':[{
				'data':real_stock_price.tolist(),
				'label':"Real Price",
				'borderColor':"blue",
				'fill':'false'
			},
			{
				'data':predicted_stock_prices.tolist(),
				'label':"Predict Price",
				'borderColor':"Red",
				'fill':'false'
			}]
		}
		return JsonResponse(data)

	def ClosePriceSuez(request):
		madinet_Nasr_Housing=pd.read_csv(STOCKS_DATA_FOLDER + "Suez Cement.csv")
		madinet_Nasr_Housing=madinet_Nasr_Housing[-500:]
		dates=[]
		for date in madinet_Nasr_Housing['TRADE_DATE'].values:
			dates.append(str(date)[:10])
		data={
			'labels':dates,
			'datasets':[{
				'data':madinet_Nasr_Housing['CLOSE_PRICE'].values.tolist(),
				'label':"CLOSE_PRICE",
				'borderColor':"red",
				'fill':'false'
			}]}
		return JsonResponse(data)

	def RegressionModelsSuez(request):
		madinet_Nasr_Housing=pd.read_csv(STOCKS_DATA_FOLDER + "Suez Cement.csv")
		madinet_Nasr_Housing=madinet_Nasr_Housing[-500:]
		dates=[]
		for date in madinet_Nasr_Housing['TRADE_DATE'].values:
			dates.append(str(date)[:10])
		madinet_Nasr_Housing['EWMA_12']=madinet_Nasr_Housing['CLOSE_PRICE'].ewm(span=12).mean()
		madinet_Nasr_Housing['HL_PCT']=(madinet_Nasr_Housing['HIGH_PRICE']-madinet_Nasr_Housing['LOW_PRICE'])/madinet_Nasr_Housing['LOW_PRICE']*100.0
		madinet_Nasr_Housing['PCT_change']=(madinet_Nasr_Housing['CLOSE_PRICE']-madinet_Nasr_Housing['OPEN_PRICE'])/madinet_Nasr_Housing['OPEN_PRICE']*100.0
		madinet_Nasr_Housing.dropna(inplace=True)
		madinet_Nasr_Housing['CLOSE_PRICE'].shift(-3).tail()
		X=madinet_Nasr_Housing[['EWMA_12','HL_PCT','PCT_change']].values
		y=madinet_Nasr_Housing['CLOSE_PRICE'].values
		X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,shuffle=False,random_state=1)
		sc=StandardScaler()
		sc.fit(X_train)
		X_train=sc.transform(X_train)
		X_test=sc.transform(X_test)
		lr=LinearRegression()
		KNN=KNeighborsRegressor()
		SVR=svm.SVR()
		lr.fit(X_train,y_train)
		KNN.fit(X_train,y_train)
		SVR.fit(X_train,y_train)
		y_pred_lr=lr.predict(X_test)
		y_pred_SVR = SVR.predict(X_test)
		y_pred_KNN = KNN.predict(X_test)
		madinet_Nasr_Housing['LR_pred']=None
		madinet_Nasr_Housing['KNN_pred']=None
		madinet_Nasr_Housing['SVR_pred']=None
		madinet_Nasr_Housing['LR_pred'].iloc[350:]=y_pred_lr
		madinet_Nasr_Housing['KNN_pred'].iloc[350:]=y_pred_KNN
		madinet_Nasr_Housing['SVR_pred'].iloc[350:]=y_pred_SVR
		data={
			'labels':dates,
			'datasets':[{
				'data':madinet_Nasr_Housing['SVR_pred'].values.tolist(),
				'label':"SVR_pred",
				'borderColor':"blue",
				'fill':'false'
			},
			{
				'data':madinet_Nasr_Housing['LR_pred'].values.tolist(),
				'label':"LR_pred",
				'borderColor':"green",
				'fill':'false'
			},
			{
				'data':madinet_Nasr_Housing['KNN_pred'].values.tolist(),
				'label':"KNN_pred",
				'borderColor':"orange",
				'fill':'false'
			},
			{
				'data':madinet_Nasr_Housing['CLOSE_PRICE'].values.tolist(),
				'label':"CLOSE_PRICE",
				'borderColor':"Red",
				'fill':'false'
			}]
		}
		return JsonResponse(data)