# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from ai_models import savings_recommender

# AI 모델 경로를 sys.path에 추가하여 임포트 가능하게 함
# Flask 앱의 루트 디렉토리 기준으로 ai_models 폴더를 찾도록 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_models_path = os.path.join(current_dir, 'ai_models')
sys.path.append(ai_models_path)

# AI 모델 및 데이터 경로 설정
MODEL_DIR = os.path.join(current_dir, 'models')
DATA_DIR = os.path.join(current_dir, 'data')

# settings.py 역할을 하는 클래스 (Django의 settings를 모방)
class FlaskSettings:
    MODEL_DIR = MODEL_DIR
    DATA_DIR = DATA_DIR

# Flask 앱 생성
app = Flask(__name__, static_folder='static', template_folder='templates')

# AI 코드에 FlaskSettings 객체 주입

savings_recommender.settings = FlaskSettings() # AI 모듈에 settings 객체 주입

# Recommender 객체 초기화 (앱 시작 시 한 번만 로드)
recommender_instance = savings_recommender.SavingsRecommender()


# ===== 웹 페이지 라우트 =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_expense')
def add_expense_page():
    return render_template('add_expense.html')

@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')


# ===== API 엔드포인트 =====

# 지출 추가 API
@app.route('/api/add_expense', methods=['POST'])
def api_add_expense():
    data = request.json
    expense_date_str = data.get('date')
    category = data.get('category')
    item_name = data.get('itemName')
    amount = int(data.get('amount'))
    satisfaction = int(data.get('satisfaction'))

    # 날짜 문자열을 datetime 객체로 변환
    expense_date = datetime.strptime(expense_date_str, '%Y-%m-%d')
    # 요일 계산 (0: 월요일, 6: 일요일)
    weekday = expense_date.weekday()

    # AI 모델 예측을 위해 데이터프레임 형식으로 준비
    new_expense_df = pd.DataFrame([{
        '날짜': expense_date_str, # AI 코드에서 문자열 날짜를 다시 datetime으로 변환
        '요일': weekday,
        '카테고리': category,
        '음식명': item_name,
        '금액': amount,
        '만족도': satisfaction,
    }])

    # AI 모델을 사용하여 '추천필요_예측' 값 얻기
    predicted_expense_df = recommender_instance.predict_recommendation_need(new_expense_df.copy())
    recommend_need = int(predicted_expense_df['추천필요_예측'].iloc[0]) # 0 또는 1

    # CSV 파일에 저장
    csv_path = os.path.join(DATA_DIR, 'spending_data.csv')
    # 파일이 없으면 헤더와 함께 생성
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        df = pd.DataFrame(columns=['날짜', '요일', '카테고리', '음식명', '금액', '추천필요', '만족도'])
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 기존 데이터 로드 후 추가
    existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 새로운 지출 데이터를 기존 데이터프레임 형식에 맞게 구성
    new_row = {
        '날짜': expense_date_str,
        '요일': weekday,
        '카테고리': category,
        '음식명': item_name,
        '금액': amount,
        '추천필요': recommend_need, # AI 예측 결과 반영
        '만족도': satisfaction
    }
    # pandas.concat을 사용하여 데이터프레임에 행 추가 (더 안정적)
    updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 추천 메시지 생성 (추천 필요 여부에 따라)
    recommendation_message = ""
    if recommend_need == 1:
        recommendation_message = savings_recommender.generate_recommendation(
            category, item_name, amount
        )
    else:
        recommendation_message = "절약이 필요하지 않습니다."

    return jsonify({"message": "Expense added successfully!", "recommendation": recommendation_message})


# 지출 내역 조회 API
@app.route('/api/get_expenses')
def api_get_expenses():
    csv_path = os.path.join(DATA_DIR, 'spending_data.csv')
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return jsonify([]) # 파일 없거나 비어있으면 빈 배열 반환
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # 날짜를 datetime 객체로 변환하여 정렬
        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.sort_values(by='날짜', ascending=False)
        df['날짜'] = df['날짜'].dt.strftime('%Y-%m-%d') # 다시 문자열로 변환
        return jsonify(df.to_dict(orient='records'))
    except pd.errors.EmptyDataError:
        return jsonify([]) # 빈 CSV 파일이면 빈 배열 반환
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 지출 분석 API
@app.route('/api/analysis')
def api_analysis():
    csv_path = os.path.join(DATA_DIR, 'spending_data.csv')
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return jsonify({"error": "No spending data available for analysis. Please add some expenses first."}), 404

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # '날짜' 컬럼이 문자열이면 datetime으로 변환 (analyze_spending 함수 내부에서 처리되지만 명시적)
        if not pd.api.types.is_datetime64_any_dtype(df['날짜']):
             df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        df = df.dropna(subset=['날짜']) # 변환 실패한 행 제거
        if df.empty:
             return jsonify({"error": "No valid spending data after date parsing."}), 404

        analysis_results = savings_recommender.analyze_spending(df.copy())
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 데이터 및 모델 폴더가 없으면 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 앱 실행
    app.run(host='0.0.0.0', port=5000, debug=True) # 개발 시 debug=True, 배포 시 False. host='0.0.0.0'은 외부 접속 허용