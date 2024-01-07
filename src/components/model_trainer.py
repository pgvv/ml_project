import os
import sys
from dataclasses import dataclass

#Algorithms
from catboost import CatBoostRegressor
from sklearn.ensemble import(AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#Logging & Exception Handling
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_filepath= os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            X_train, y_train, X_test, y_test= (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Booster": GradientBoostingRegressor(),
                "K-Neighbours Classifier": KNeighborsRegressor(),
                "CAT Booster": CatBoostRegressor(verbose= False),
                "ADA Booster": AdaBoostRegressor(),
                "XGB Classifier": XGBRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Booster":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGB Classifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CAT Booster":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "ADA Booster":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "K-Neighbours Classifier":{},
                
            }

            model_report: dict= evaluate_model(
                X_train= X_train, 
                y_train= y_train,
                X_test= X_test,
                y_test= y_test,
                models= models,
                param= params)
            
            #To get best model score
            best_model_score= max(sorted(model_report.values()))
            #To get corresponding best model name
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException("No best model found")
            
            logging.info("Solving model successfully found for both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_filepath,
                obj= best_model
            )

            #Results and Metrics
            predict_score= best_model.predict(X_test)
            r2_square= r2_score(y_test, predict_score)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys)