import lightgbm as lgb
import pandas as pd
import pickle
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

class LightGBMClassifier:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
    
    def objective(self, trial):
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        cv_scores = cross_val_score(
            lgb.LGBMClassifier(**params),
            self.X_train, self.y_train,
            cv=5,
            scoring='f1_macro'
        )
        return cv_scores.mean()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=100)
        best_params = study.best_trial.params
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(X_train, y_train)
        
        #モデル保存
        with open("results/model/LGBMcat_model.pkl","wb") as f:
             pickle.dump(self.model,f)
        print("#################################################")     
        print("model created!")
        print("#################################################")
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model is not trained. Fit the model first.")
        preds = self.model.predict(X)
        return pd.DataFrame(preds, index=X.index, columns=['prediction'])

    def load_model(self,x_test):
           # モデルの読み込み
          with open("results/model/LGBMcat_model.pkl", "rb") as f:
               model = pickle.load(f)
               
          if self.model is None:
              raise RuntimeError("Model is not trained. Fit the model first.")
          preds = model.predict(x_test)     
          return pd.DataFrame(preds, index=x_test.index, columns=['prediction'])