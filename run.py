import argparse
import os
import datetime
import pandas as pd
import configparser

from preprocessor.preprocess import DataCleansing
from predictor.LGBM_chatGPT import LightGBMClassifier as LGBM

from evaluator.evaluate import Output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="sample.ini")
    parser.add_argument("--name", type=str, default="lgbmsample")
    args = parser.parse_args()
    return args

def read_config(path,name):
    config = configparser.ConfigParser()
    config.read(path)
    config_dict = dict(config[name])
    type_dict = {"int":int,"float":float,"str":str}
    for key,value in config_dict.items():
        type_, value = value.split(" ")
        config_dict[key] = type_dict[type_](value)
    return config_dict

def main():
     
     config = parse_args()
     setting_dict = read_config(os.path.join("config",config.path),config.name)
     print("#######################################")
     models = {
          "lgbm": LGBM
     }
     model = models[setting_dict.pop("model_key")]
     model = model()
     
     #初期データの準備
     train_df = pd.read_csv(setting_dict.pop("data_dir"),index_col="id")
     test_df = pd.read_csv("./data/official_data/test.csv",index_col="id")
     
     #インスタンス生成
     pp = DataCleansing(train_df)
     
     X,y = pp.rt_train_data()
     
     #データ分割
     X_train, X_test, y_train, y_test = pp.train_split()

     # モデルの構築と学習
     model.fit(X_train,y_train)
     lgbm_submit = model.load_model(test_df)
     
     
     model_name = setting_dict.pop("model_name")
     
     #submitに出力
     Output.submit(lgbm_submit, f"{model_name}")


if __name__ == "__main__":
     main()
     
