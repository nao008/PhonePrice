import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from functools import partial


class LightGBMClassifier():
    def __init__(self, depth=3, datadir=''):
        self.datadir = datadir
        self.depth = depth

    def fit(self, x_train, y_train):

        x_lgbtrain, x_lgbeval, y_lgbtrain, y_lgbeval = train_test_split(x_train, y_train, test_size=0.3, shuffle=True, random_state=314)
        lgb_train = lgb.Dataset(x_lgbtrain, y_lgbtrain, free_raw_data=False)
        lgb_eval = lgb.Dataset(x_lgbeval, y_lgbeval, reference=lgb_train, free_raw_data=False)
        model = lgb.LGBMClassifier(objective='multiclass')

        def f1_macro_score(y_true, y_pred):
            vec_dict={
                 0: [1,0,0,0],
                 1: [0,1,0,0],
                 2: [0,0,1,0],
                 3: [0,0,0,1],
                 }
            y_true_vec = [vec_dict[label] for label in y_true]
            y_pred_vec = [vec_dict[label] for label in y_pred]
            
            return f1_score(y_true_vec, y_pred_vec, average='macro')


        def bayes_objective(trial, dep):
            params = {
                'max_depth': dep,
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 2, 6),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 0, 10)
            }
            # モデルにパラメータ適用
            model.set_params(**params)
            # cross_val_scoreでクロスバリデーション
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            # 評価指標をf1_macroに変更
            score_funcs = {
                'f1_macro': make_scorer(f1_macro_score),
            }
            scores = cross_validate(model,
                                    pd.concat([x_lgbtrain, x_lgbeval], axis=0).values,
                                    pd.concat([y_lgbtrain, y_lgbeval], axis=0).values,
                                    cv=kf,
                                    scoring=score_funcs)

            return scores['test_f1_macro'].mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(partial(bayes_objective, dep=self.depth), n_trials=400)

        self.model = lgb.train(study.best_trial.params,
                               lgb_train,
                               num_boost_round=1000,
                               valid_names=['train', 'valid'],
                               valid_sets=[lgb_train, lgb_eval],
                               verbose_eval=-1)
        
        #モデル保存
        with open("results/model/LGBMmodel.pkl","wb") as f:
             pickle.dump(self.model,f)

    def predict(self, x_test):
             
        y_predict = pd.Series(np.argmax(self.model.predict(x_test.values), axis=1), index=x_test.index)
        return pd.DataFrame(y_predict, index=x_test.index, columns=['prediction'])

    def load_model(self,x_test):
          # モデルの読み込み
         with open("results/model/LGBMmodel.pkl", "rb") as f:
              model = pickle.load(f)
              
         y_predict = pd.Series(model.predict(x_test.values), index=x_test.index)
         y_predict= y_predict.round().astype(int)
         y_predict = y_predict.replace(4,3)
         return pd.DataFrame(y_predict, index=x_test.index, columns=['prediction'])