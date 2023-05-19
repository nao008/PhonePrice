import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
from preprocess.preprocess import DataCleansing
import os


def mein():
     current_dir = os.getcwd()
     test_df = pd.read_csv(f"{current_dir}/data/test.csv",index_col=0)

     pp = DataCleansing()

     X_train, X_test, y_train, y_test = pp.train_split()

     # モデルの構築と学習
     model = LinearRegression()
     model.fit(X_train, y_train)
     # =============================================================================
     # # モデルの評価
     # y_pred = model.predict(X_test)
     # y_pred = pd.DataFrame(y_pred)
     # y_pred = y_pred.round().astype(int)
     # =============================================================================

     y_pred_test = model.predict(test_df)
     y_pred_test = pd.DataFrame(y_pred_test,index=test_df.index)
     y_pred_test = y_pred_test.round().astype(int)


     # y_pred_test.to_csv(f"submits/{datetime.datetime.today().strftime('%Y%m%d')}submit.csv",header=False)








if __name__ == "__mein__":
     mein()