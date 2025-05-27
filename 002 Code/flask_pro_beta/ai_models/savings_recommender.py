# ai_models/savings_recommender.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import traceback

# settings 객체가 외부(app.py)에서 주입될 것을 가정
settings = None

def generate_sample_data(num_records=50):
    """
    절약 추천 시스템을 위한 샘플 데이터셋 생성 함수
    """
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=random.randint(0, 29)) for _ in range(num_records)]
    dates_str = [d.strftime('%Y-%m-%d') for d in dates] # 문자열로 변환

    categories = ['카페', '식비', '편의점', '마트', '술/유흥', '간식', '배달음식', '교통비', '기타']

    food_by_category = {
        '카페': ['아메리카노', '카페라떼', '아이스티', '아이스아메리카노', '카페모카', '바닐라라떼', '에스프레소'],
        '식비': ['돈까스정식', '김치찌개', '제육볶음', '비빔밥', '된장찌개', '칼국수', '짜장면', '삼겹살'],
        '편의점': ['도시락', '삼각김밥', '샌드위치', '라면', '김밥', '샐러드', '즉석식품'],
        '마트': ['장보기', '과일', '채소', '고기', '생필품', '음료수', '반찬'],
        '술/유흥': ['맥주', '소주', '양주', '와인', '안주', '치킨', '피자'],
        '간식': ['과자', '아이스크림', '빵', '초콜릿', '스낵', '떡', '사탕'],
        '배달음식': ['치킨', '피자', '족발', '보쌈', '중국집', '햄버거', '분식'],
        '교통비': ['지하철', '버스', '택시', '기차'],
        '기타': ['문구류', '취미용품', '문화생활', '옷', '미용', '생활용품'],
    }

    price_range = {
        '카페': (2500, 7000),
        '식비': (6000, 15000),
        '편의점': (2000, 8000),
        '마트': (5000, 50000),
        '술/유흥': (10000, 50000),
        '간식': (1000, 5000),
        '배달음식': (15000, 35000),
        '교통비': (1300, 50000),
        '기타': (1000, 100000),
    }

    weekdays = [d.weekday() for d in dates]

    data = []
    for i in range(num_records):
        category = random.choice(categories)
        food = random.choice(food_by_category.get(category, ['기타 소비']))
        price_min, price_max = price_range[category]
        price = random.randint(price_min, price_max)

        need_recommendation = 1 if price > (price_min + price_max) / 2 else 0

        if need_recommendation:
            satisfaction = random.choices([1, 2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]
        else:
            satisfaction = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.2, 0.3, 0.35])[0]

        data.append({
            '날짜': dates_str[i],
            '요일': weekdays[i],
            '카테고리': category,
            '음식명': food,
            '금액': price,
            '추천필요': need_recommendation,
            '만족도': satisfaction
        })
    df = pd.DataFrame(data)
    return df

def generate_recommendation(category, food_name, price):
    """
    카테고리, 음식명, 가격을 기반으로 맞춤형 절약 추천 메시지 생성
    """
    recommendations = {
        '카페': [
            f"'{food_name}' 대신 집에서 직접 커피를 내려 보세요. 한 달이면 약 {int(price * 20 * 0.8):,}원을 절약할 수 있어요.",
            f"'{food_name}'를 매일 마시는 대신 일주일에 3번으로 줄이면 월 {int(price * 12):,}원을 절약할 수 있어요.",
            f"'{food_name}' 대신 텀블러를 사용하면 할인을 받을 수 있어 매달 {int(price * 20 * 0.1):,}원 정도 절약 가능해요."
        ],
        '식비': [
            f"'{food_name}' 대신 도시락을 싸서 다니면 끼니당 약 {int(price * 0.6):,}원을 절약할 수 있어요.",
            f"'{food_name}' 같은 메뉴는 집에서 만들면 재료비가 {int(price * 0.4):,}원 정도로 절반 이하예요.",
            f"학교나 회사 구내식당을 이용하면 '{food_name}'보다 {int(price * 0.3):,}원 정도 저렴해요."
        ],
        '편의점': [
            f"'{food_name}' 대신 집에서 미리 준비해 오면 약 {int(price * 0.7):,}원을 절약할 수 있어요.",
            f"'{food_name}'보다는 대용량 제품을 구매해서 소분해 먹으면 끼니당 {int(price * 0.5):,}원 정도 절약돼요.",
            f"'{food_name}'은 마트에서 사면 편의점보다 약 {int(price * 0.3):,}원 저렴해요."
        ],
        '마트': [
            f"'{food_name}'은 장보기 전 미리 계획을 세우고 목록을 작성하면 약 {int(price * 0.2):,}원을 절약할 수 있어요.",
            f"'{food_name}'은 대형마트 할인일에 구매하면 {int(price * 0.1):,}원에서 {int(price * 0.3):,}원까지 절약할 수 있어요.",
            f"'{food_name}'은 온라인 마트를 이용하면 오프라인보다 약 {int(price * 0.15):,}원 저렴해요."
        ],
        '술/유흥': [
            f"'{food_name}' 대신 집에서 홈파티를 즐기면 약 {int(price * 0.7):,}원을 절약할 수 있어요.",
            f"'{food_name}'은 주 1회로 제한하면 월 {int(price * 3):,}원을 절약할 수 있어요.",
            f"'{food_name}' 대신 저렴한 취미 활동으로 대체하면 매달 {int(price * 4):,}원의 비용을 줄일 수 있어요."
        ],
        '간식': [
            f"'{food_name}' 대신 대용량으로 구매해서 소분해 먹으면 약 {int(price * 0.3):,}원을 절약할 수 있어요.",
            f"'{food_name}' 대신 집에서 만든 간식을 챙겨 다니면 하루 {int(price * 0.7):,}원을 절약할 수 있어요.",
            f"'{food_name}'은 정해진 간식 시간을 정해놓고 계획적으로 구매하면 월 {int(price * 15):,}원을 절약할 수 있어요."
        ],
        '배달음식': [
            f"'{food_name}' 대신 직접 요리하면 한 끼당 약 {int(price * 0.6):,}원을 절약할 수 있어요.",
            f"'{food_name}'보다 포장해서 픽업하면 배달비 {int(price * 0.1):,}원을 절약할 수 있어요.",
            f"'{food_name}'은 여러 명이 함께 시켜 먹으면 1인당 {int(price * 0.3):,}원을 절약할 수 있어요."
        ],
        '교통비': [
            f"'{food_name}' 대신 대중교통을 이용하면 택시비의 {int(price * 0.8):,}원을 절약할 수 있어요.",
            f"걷거나 자전거를 이용하면 건강도 챙기고 '{food_name}' 소비도 줄일 수 있어요.",
            f"대중교통 정기권을 이용하면 '{food_name}'로 인한 월 {int(price * 0.1):,}원 정도를 절약할 수 있어요."
        ],
        '기타': [
            f"'{food_name}' 구매 전 정말 필요한지 한 번 더 생각해 보세요. 충동구매를 줄일 수 있어요.",
            f"'{food_name}' 대신 중고거래나 공유 서비스를 활용하면 {int(price * 0.5):,}원 이상 절약 가능해요.",
            f"불필요한 구독 서비스 해지 등 고정 지출을 점검해 보세요."
        ]
    }
    default_recommendations = [
        f"'{food_name}'의 소비를 줄이고 식비 계획을 세워보세요.",
        f"더 저렴한 대안을 찾아보는 것이 좋겠어요.",
        f"비슷한 만족감을 주는 더 경제적인 선택이 있을지 고민해보세요."
    ]
    if category in recommendations:
        return random.choice(recommendations[category])
    else:
        return random.choice(default_recommendations)

def save_sample_data_to_csv(df, file_path):
    """
    생성된 샘플 데이터를 CSV 파일로 저장
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

def train_recommendation_model(data, model_dir):
    """
    추천 필요 여부를 예측하는 머신러닝 모델 학습
    """
    train_data = data.copy()

    if '날짜' in train_data.columns and not pd.api.types.is_datetime64_any_dtype(train_data['날짜']):
        train_data['날짜'] = pd.to_datetime(train_data['날짜'], errors='coerce')
    if '요일' not in train_data.columns and '날짜' in train_data.columns:
        train_data['요일'] = train_data['날짜'].dt.weekday
    elif '요일' not in train_data.columns:
        train_data['요일'] = np.random.randint(0, 7, size=len(train_data))

    all_categories = ['카페', '식비', '편의점', '마트', '술/유흥', '간식', '배달음식', '교통비', '기타']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(pd.DataFrame(all_categories, columns=['카테고리']))

    X = train_data[['요일', '카테고리', '금액']].copy()
    y = train_data['추천필요']

    try:
        category_encoded = encoder.transform(X[['카테고리']])
        weekday = X['요일'].values.reshape(-1, 1)
        price = X['금액'].values.reshape(-1, 1)

        scaler = StandardScaler()
        price_scaled = scaler.fit_transform(price)

        X_processed = np.hstack([weekday, category_encoded, price_scaled])

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"모델 정확도: {accuracy:.4f}")
        # print("\n분류 보고서:")
        # print(classification_report(y_test, y_pred))

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, 'savings_recommendation_model.pkl'))
        joblib.dump(encoder, os.path.join(model_dir, 'category_encoder.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'price_scaler.pkl'))
        return model, encoder, scaler
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {e}")
        traceback.print_exc()
        model = RandomForestClassifier(n_estimators=10)
        encoder = OneHotEncoder(sparse_output=False)
        scaler = StandardScaler()
        dummy_X = np.random.rand(10, len(all_categories) + 2)
        dummy_y = np.random.randint(0, 2, 10)
        model.fit(dummy_X, dummy_y)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, 'savings_recommendation_model.pkl'))
        joblib.dump(encoder, os.path.join(model_dir, 'category_encoder.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'price_scaler.pkl'))
        return model, encoder, scaler

class SavingsRecommender:
    def __init__(self):
        if settings is None:
            raise RuntimeError("FlaskSettings object not injected into savings_recommender module.")

        self.model_path = os.path.join(settings.MODEL_DIR, 'savings_recommendation_model.pkl')
        self.encoder_path = os.path.join(settings.MODEL_DIR, 'category_encoder.pkl')
        self.scaler_path = os.path.join(settings.MODEL_DIR, 'price_scaler.pkl')
        self.spending_data_csv_path = os.path.join(settings.DATA_DIR, 'spending_data.csv')

        self.model = None
        self.encoder = None
        self.scaler = None

        try:
            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)
            self.scaler = joblib.load(self.scaler_path)
            # print("AI 모델 로드 성공.")
        except FileNotFoundError:
            print(f"경고: 모델 파일이 없습니다. '{self.spending_data_csv_path}'를 기반으로 모델을 학습합니다.")
            sample_data = generate_sample_data(num_records=500)
            save_sample_data_to_csv(sample_data, file_path=self.spending_data_csv_path)
            self.model, self.encoder, self.scaler = train_recommendation_model(
                sample_data, model_dir=settings.MODEL_DIR
            )
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            traceback.print_exc()

    def preprocess_data(self, data):
        processed_data = data.copy()
        if '요일' not in processed_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(processed_data['날짜']):
                processed_data['날짜'] = pd.to_datetime(processed_data['날짜'], errors='coerce')
            processed_data['요일'] = processed_data['날짜'].dt.weekday

        X = processed_data[['요일', '카테고리', '금액']].copy()

        category_encoded = self.encoder.transform(X[['카테고리']])

        weekday = X['요일'].values.reshape(-1, 1)
        price = X['금액'].values.reshape(-1, 1)

        price_scaled = self.scaler.transform(price)
        X_processed = np.hstack([weekday, category_encoded, price_scaled])
        return X_processed

    def predict_recommendation_need(self, data):
        X_processed = self.preprocess_data(data)
        predictions = self.model.predict(X_processed)
        result_data = data.copy()
        result_data['추천필요_예측'] = predictions
        return result_data

    def generate_recommendations(self, data):
        processed_data = data.copy()
        predicted_data = self.predict_recommendation_need(processed_data)

        if '추천' not in predicted_data.columns:
            predicted_data['추천'] = None

        for idx, row in predicted_data.iterrows():
            if row['추천필요_예측'] == 1:
                recommendation = generate_recommendation(
                    row['카테고리'],
                    row['음식명'],
                    row['금액']
                )
                predicted_data.at[idx, '추천'] = recommendation
            else:
                predicted_data.at[idx, '추천'] = "절약이 필요하지 않습니다."
        return predicted_data

def analyze_spending(data, daily_budget=15000):
    """
    소비 데이터를 분석하고 예산 초과 여부 확인
    """
    analysis_data = data.copy()

    if not pd.api.types.is_datetime64_any_dtype(analysis_data['날짜']):
        analysis_data['날짜'] = pd.to_datetime(analysis_data['날짜'], errors='coerce')
    analysis_data = analysis_data.dropna(subset=['날짜'])

    if analysis_data.empty:
        return {
            'total_expense': 0,
            'avg_daily_expense': 0,
            'daily_budget': daily_budget,
            'overspent_days': [],
            'category_total': [],
            'weekday_avg': []
        }

    daily_total = analysis_data.groupby('날짜')['금액'].sum()
    category_total = analysis_data.groupby('카테고리')['금액'].sum().sort_values(ascending=False)

    weekday_avg = pd.Series(dtype='float64')
    if not analysis_data['날짜'].empty:
        try:
            analysis_data['요일이름'] = analysis_data['날짜'].dt.day_name(locale='ko_KR')
            if analysis_data['요일이름'].isnull().any():
                 weekday_map = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}
                 analysis_data['요일이름'] = analysis_data['날짜'].dt.weekday.map(weekday_map)
            weekday_avg = analysis_data.groupby('요일이름')['금액'].mean().sort_values(ascending=False)
        except Exception as e:
            analysis_data['요일이름'] = analysis_data['날짜'].dt.day_name()
            weekday_avg = analysis_data.groupby('요일이름')['금액'].mean().sort_values(ascending=False)

    overspent_days = daily_total[daily_total > daily_budget]

    return {
        'total_expense': int(analysis_data['금액'].sum()),
        'avg_daily_expense': int(daily_total.mean()) if not daily_total.empty else 0,
        'daily_budget': daily_budget,
        'overspent_days': [{
            'date': date.strftime('%Y-%m-%d'),
            'amount': int(amount),
            'over_amount': int(amount - daily_budget)
        } for date, amount in overspent_days.items()],
        'category_total': [{
            'category': cat,
            'amount': int(amt),
            'percentage': round(amt / analysis_data['금액'].sum() * 100, 1)
        } for cat, amt in category_total.items()],
        'weekday_avg': [{
            'day': day,
            'avg_amount': int(amt)
        } for day, amt in weekday_avg.items()] if not weekday_avg.empty else []
    }