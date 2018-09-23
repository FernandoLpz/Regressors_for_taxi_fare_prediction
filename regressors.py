import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

Xtrain = np.load('Xtrain.npy')
Xtest = np.load('Xtest.npy')
Ytrain = np.load('Ytrain.npy')
Ytest = np.load('Ytest.npy')

class Regressors:
	def __init__(self, Xtrain, Xtest, Ytrain, Ytest):
		self.Xtrain = Xtrain
		self.Xtest = Xtest
		self.Ytrain = Ytrain
		self.Ytest = Ytest

	def def_xgboost(self, estimators):
		xgb_ = xgb.XGBRegressor(objective ='reg:linear', learning_rate=0.01, max_depth=3, n_estimators=estimators)
		xgb_.fit(self.Xtrain, self.Ytrain)
		pred = xgb_.predict(self.Xtest)

		return pred

	def def_RandomForestRegressor(self, estimators):
		rfr_ = RandomForestRegressor(n_estimators=estimators, max_depth=3)
		rfr_.fit(self.Xtrain, self.Ytrain)
		pred = rfr_.predict(self.Xtest)

		return pred

	def def_GradientBoostingRegressor(self, estimators):
		gbr_ = GradientBoostingRegressor(n_estimators=estimators, max_depth=3)
		gbr_.fit(self.Xtrain, self.Ytrain)
		pred = gbr_.predict(self.Xtest)

		return pred

	def def_AdaBoostRegressor(self, estimators):
		abr_ = AdaBoostRegressor(n_estimators=estimators)
		abr_.fit(self.Xtrain, self.Ytrain)
		pred = abr_.predict(self.Xtest)

		return pred

def def_metrics(ypred):
	mae = mean_absolute_error(Ytest, ypred)
	mse = mean_squared_error(Ytest, ypred)

	return mae, mse

def plot_performance(plot_name, loss_mae, loss_mse):
	steps = np.arange(50, 500, 50)
	plt.style.use('ggplot')
	plt.title(plot_name)
	plt.plot(steps, loss_mae, linewidth=3, label="MAE")
	plt.plot(steps, loss_mse, linewidth=3, label="MSE")
	plt.legend()
	plt.ylabel("Loss")
	plt.xlabel("Number of estimators")
	plt.show()

def main():
	model = Regressors(Xtrain, Xtest, Ytrain, Ytest)

	plot_name="XGBoosting Regressor"
	loss_mae, loss_mse = [], []
	print(plot_name)
	for est in range(50,500,50):
		print("Number of estimators: %d" % est)
		mae, mse = def_metrics(model.def_xgboost(estimators = est))
		print("MAE: ", mae)
		print("MSE: ", mse)
		loss_mae.append(mae)
		loss_mse.append(mse)
	plot_performance(plot_name, loss_mae, loss_mse)

	plot_name="Random Forest Regressor"
	loss_mae, loss_mse = [], []
	print(plot_name)
	for est in range(50,500,50):
		print("Number of estimators: %d" % est)
		mae, mse = def_metrics(model.def_xgboost(estimators = est))
		print("MAE: ", mae)
		print("MSE: ", mse)
		loss_mae.append(mae)
		loss_mse.append(mse)
	plot_performance(plot_name, loss_mae, loss_mse)

	plot_name="Gradient Boosting Regressor"
	loss_mae, loss_mse = [], []
	print(plot_name)
	for est in range(50,500,50):
		print("Number of estimators: %d" % est)
		mae, mse = def_metrics(model.def_xgboost(estimators = est))
		print("MAE: ", mae)
		print("MSE: ", mse)
		loss_mae.append(mae)
		loss_mse.append(mse)
	plot_performance(plot_name, loss_mae, loss_mse)

	plot_name="Ada Boost Regressor"
	loss_mae, loss_mse= [], []
	print(plot_name)
	for est in range(50,500,50):
		print("Number of estimators: %d" % est)
		mae, mse = def_metrics(model.def_xgboost(estimators = est))
		print("MAE: ", mae)
		print("MSE: ", mse)
		loss_mae.append(mae)
		loss_mse.append(mse)
	plot_performance(plot_name, loss_mae, loss_mse)

if __name__ == '__main__':
	main()




    