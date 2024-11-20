import subprocess
import sys

# distutils 문제를 피하기 위한 예외 처리
try:
    # 'pip install pymongo' 명령을 실행
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymongo"])
except subprocess.CalledProcessError as e:
    print(f"Error occurred while installing pymongo: {e}")

# 필요한 라이브러리 목록
required_libraries = [
    'scikit-learn',
    'lightgbm',
    'xgboost',
    'prophet',
    'matplotlib',
    'seaborn',
    'pandas',
    'numpy',
    'tensorflow'
]

# 설치되지 않은 라이브러리만 설치하기
for lib in required_libraries:
    try:
        # 해당 라이브러리가 이미 설치되어 있으면 ImportError가 발생하지 않음
        __import__(lib)
    except ImportError:
        try:
            # 라이브러리가 없으면 pip로 설치
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while installing {lib}: {e}")

# 그 후, 원하는 라이브러리들을 import 할 수 있다.
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import json_normalize
import numpy as np

from prophet import Prophet
import holidays
import os

from datetime import datetime, timedelta
now = datetime.now()

import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# 현재 파이썬 파일이 있는 경로를 얻어 config.txt 파일 경로 설정
current_directory = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_directory, 'config.txt')

config = {}

# config.txt 파일을 읽어서 정보 가져오기
with open(config_file_path, 'r') as file:
    for line in file:
        # 줄에서 공백을 제거하고 '=' 기준으로 나누어 키-값 형태로 저장
        key, value = line.strip().split('=')
        config[key] = value

# config.txt에서 가져온 정보로 MongoDB 연결
mongo_uri = config.get('MONGO_URI')
db_pw = int(config.get('PW'))
collection_name = config.get('COLLECTION_NAME')

# 방문자 수 데이터 df 변환
from pymongo import MongoClient

# 데이터가 저장된 MongoDB의 주소
client = MongoClient(mongo_uri, db_pw)

# db를 저장하기
db = client.crawling

collection = db[collection_name]

# collection에 저장된 데이터를 데이터프레임으로 변환 및 저장
rows = collection.find()
history_stations = []
for row in rows:
    history_stations.append(row)

history_stations = pd.DataFrame(history_stations)

# 모든 충전소 정보를 하나로 합치기 df_all
# history_stations의 각 항목을 리스트로 저장
dfs = []

# 첫 번째 항목 처리
for i in range(len(history_stations)):
    df_temp = pd.DataFrame(history_stations['history'][i])  # history 항목을 DataFrame으로 변환
    df_temp['_id'] = history_stations['_id'][i]  # _id 값을 추가
    dfs.append(df_temp)  # 각 DataFrame을 리스트에 저장

# 모든 DataFrame을 한 번에 concat
df_all = pd.concat(dfs, ignore_index=True)

# 임의의 날짜 변수 추가
# 변수 생성 (주말, 월, 일, 시간, 분) #########################
df_all['weekday'] = df_all['time'].dt.weekday
df_all['month'] = df_all['time'].dt.month
df_all['day'] = df_all['time'].dt.day
df_all['hour'] = df_all['time'].dt.hour
df_all['minute'] = df_all['time'].dt.minute

# 변수 생성 (공휴일)
kr_holidays = holidays.KR()
df_all['holiday'] = df_all.time.apply(lambda x: 1 if x in kr_holidays else 0)


# ml 실행을 위해 날짜를 index로 설정하기
df_all.set_index(keys='time', inplace=True)

# 충전소 id값 인코딩하기
df_all['organization'] = df_all['_id'].astype(str).str[:2]

# 1. LabelEncoder 객체 생성
label_encoder = LabelEncoder()

# 2. 'organization' 컬럼을 라벨 인코딩하여 'org_encoded' 컬럼에 저장
df_all['org_encoded'] = label_encoder.fit_transform(df_all['organization'])

# 3. 'organization' 컬럼 삭제하기
df_all = df_all.drop('organization', axis=1)

# target encoding 적용
# 1. 각 '_id'에 대해 target 값(visitNum)의 평균을 구하기
target_means = df_all.groupby('_id')['visitNum'].mean()

# 2. '_id' 컬럼을 해당 평균 값으로 대체하기
df_all['id_encoded'] = df_all['_id'].map(target_means)

# _id 컬럼 삭제하기
df_all = df_all.drop('_id', axis=1)

# 모델 학습 및 예측
# count 변수 저장하기
count = df_all.iloc[0, 0]

# target(예측할 열) 설정하기
target = 'visitNum'

# x, y 값 설정하기
x = df_all.drop(target, axis=1)
y = df_all[target]


# 데이터가 10개 이상일 경우, 기존처럼 훈련/테스트 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

model = RandomForestRegressor(random_state=0, n_jobs=-1)

model.fit(x_train, y_train) # 모델 학습

# 특성 중요도 추출
importances = model.feature_importances_

y_pred = model.predict(x_test) # 모델 예측
y_pred = np.round(y_pred)

print(r2_score(y_test, y_pred))  # 모델 정확도 출력
acc = model.score(x_test, y_test)

# 시각화를 위함 - df 형식으로 변환
y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(y_test)

# 하나의 df로 합치기
y_test.reset_index(inplace=True)
df = pd.concat([y_test, y_pred], axis=1)
df.columns = ['time', 'y_test', 'y_pred']

# 모델 성능 평가
# 'y_test'와 'y_pred' 열을 비교하여 정확도 평가

# R² (R-squared) 평가
r2 = r2_score(df['y_test'], df['y_pred'])
print(f"R-squared: {r2:.4f}")

# MSE (Mean Squared Error) 평가
mse = mean_squared_error(df['y_test'], df['y_pred'])
print(f"Mean Squared Error: {mse:.4f}")

# RMSE (Root Mean Squared Error) 평가
rmse = mse ** 0.5
print(f"Root Mean Squared Error: {rmse:.4f}")

# MAE (Mean Absolute Error) 평가
mae = mean_absolute_error(df['y_test'], df['y_pred'])
print(f"Mean Absolute Error: {mae:.4f}")

# 실제 데이터로 예측 후 MongoDB에 반영하기
classes = label_encoder.classes_

def run_ml(i):
    # 시간 단위별 예측 df 생성
    pre_df = pd.date_range(now.date()+ timedelta(days=1) , periods=24 , freq="30min") # 30분 단위로 예측 df 만들기

    pre_df = pd.DataFrame(pre_df) # 데이터 프레임 형태로 변환
    pre_df.columns=['time'] # 열 이름 변경

    id = history_stations['_id'][i]

    pre_df['count'] = count

    # 변수 추가하기
    pre_df['time'] = pd.to_datetime(pre_df['time'])
    pre_df['weekday'] = pre_df['time'].dt.weekday
    pre_df['month'] = pre_df['time'].dt.month
    pre_df['day'] = pre_df['time'].dt.day
    pre_df['hour'] = pre_df['time'].dt.hour
    pre_df['minute'] = pre_df['time'].dt.minute

    kr_holidays = holidays.KR()
    pre_df['holiday'] = pre_df.time.apply(lambda x: 1 if x in kr_holidays else 0)

    pre_df.set_index(keys='time', inplace=True)

    # #### 충전소 종류 추가
    # 'id'의 앞 두 글자만 추출
    id_prefix = id[:2]  # 예: 'AB'

    # 'id_prefix'와 'classes_'에서 동일한 값이 있는 인덱스 찾기
    org_value = np.where(classes == id_prefix)[0]
    pre_df['org_encoded'] = org_value[0]

    #### target encoding을 사용하려고 하는데 각각 target을 i값 별로 다르기 때문에
    pre_df['id_encoded'] = target_means.iloc[i]


    pre_predict = model.predict(pre_df)
    pre_predict= np.round(pre_predict)

    pre_predict = pre_predict.tolist()
    collection = db['demand-info']

    statId = history_stations['_id'][i]
    # 해당 statId를 가진 문서 조회

    # print(id, statId)
    # print(pre_predict)
    existing_doc = collection.find_one({"statId": statId })

    # 문서가 존재하지 않으면 새로운 문서를 추가하고 업데이트
    if existing_doc is None:
        new_doc = { "statId": statId, "demandInfo": { "viewNum": 0, "departsIn30m": [], "hourlyVisitNum": [] } }
        result = collection.insert_one(new_doc)
        print("Added new document")
        #time.sleep(0.1)

    x = collection.update_one(
        {"statId":statId},
        {"$set" : {
            'demandInfo.hourlyVisitNum' : pre_predict
        }
        })


for i in range(len(history_stations)):
    run_ml(i)

# 끝 ,
