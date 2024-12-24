import requests
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Attention
from bs4 import BeautifulSoup

category_names = {
    1: "PRICING",
    2: "ORDER",
    3: "BOOKING",
    4: "SCHEDULE INFORM",
    5: "MRN",
    6: "SECURITY(BA)",
    7: "SECURITY(XRAY)",
    8: "SECURITY(RAS)",
    9: "PICK UP",
    10: "DOCS",
    11: "CS PROBLEM(SCHEDULE)",
    12: "CS PROBLEM(DAMAGE)",
    13: "CS PROBLEM(MISSING)",
    14: "CS PROBLEM(CROSS)",
    15: "CS PROBLEM(MISS)",
    16: "CS PROBLEM(PRICING)",
    17: "SCHEDULE CNFM",
    18: "ETC"
}

# 1. 엘라스틱서치 데이터 로드
url = "http://110.234.28.58:9200/air.france1@atlanticif.com/_search?pretty=true"
headers = {"Content-Type": "application/json"}

today = datetime.now().strftime("%Y%m%d%H%M%S")
data = {
    "query": {
        "bool": {
            "must": [
                {"exists": {"field": "cd_classify"}},
                {"range": {"tm_rcv": {"gte": "20241212000000", "lte": today}}}
            ]
        }
    },
    "size": 1000
}

response = requests.post(url, headers=headers, json=data)

if response.status_code != 200:
    print(f"Error fetching data: {response.status_code}, {response.text}")
    exit()

# 2. 데이터 준비
results = response.json()
documents = [hit["_source"] for hit in results["hits"]["hits"]]
df = pd.DataFrame(documents)

print("사용 가능한 컬럼들:", df.columns.tolist())

# 필요한 컬럼만 선택
df = df[["dc_body", "cd_classify"]]

# 컬럼 이름 변경 후 확인
df = df.rename(columns={"dc_body": "text", "cd_classify": "label"})
print("\n변경 후 컬럼명:", df.columns.tolist())

# 데이터 샘플 확인
print("\n데이터 샘플:")
print(df.head())

# 결측치 확인
print("\n결측치 개수:")
print(df.isnull().sum())

# 3. 라벨 인코딩

# 라벨 전처리 ('001' -> 1 변환)
df['label'] = df['label'].str.lstrip('0').astype(int)

# 유효한 라벨 확인
valid_labels = list(category_names.keys())
print("\n유효한 라벨:", valid_labels)

# 현재 데이터의 라벨 분포 확인
print("\n현재 라벨 분포:")
for label, count in df['label'].value_counts().items():
   print(f"라벨 {label} ({category_names[label]}) : {count}개")

# 유효하지 않은 라벨 확인
invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique()
print("\n유효하지 않은 라벨:", invalid_labels)

# 유효한 데이터만 필터링
df = df[df['label'].isin(valid_labels)]

# 라벨 인코딩
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df['label'])

# 4. 텍스트 전처리
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # HTML 태그 및 구조 완전히 제거
    soup = BeautifulSoup(text, 'html.parser')
    
    # 모든 스크립트, 스타일 태그 제거
    for element in soup(['script', 'style', 'o:p']):
        element.decompose()
    
    # 실제 텍스트 내용만 추출 (줄바꿈 보존)
    lines = soup.get_text(separator=' ', strip=True).splitlines()
    lines = [line.strip() for line in lines if line.strip()]  # 빈 줄 제거
    text = ' '.join(lines)
    
    # 텍스트 정제
    text = text.lower()  # 소문자 변환
    text = re.sub(r'&\w+;', ' ', text)  # HTML 엔티티 제거
    text = re.sub(r'[^\w\s]', ' ', text)  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
    text = text.strip()  # 앞뒤 공백 제거
    
    return text

# 텍스트 전처리 적용
df["text"] = df["text"].apply(preprocess_text)

# 전처리된 데이터 확인
print("\n전처리된 데이터 샘플:")
print(df.head())

# 각 카테고리별 예시 출력
print("\n각 카테고리별 예시:")
for label, category in category_names.items():
    examples = df[df['label'] == label]['text']
    if not examples.empty:
        print(f"\n{category} (라벨: {label})")
        print(f"데이터 수: {len(examples)}개")
        print(f"예시:")
        print(examples.iloc[0])  # 각 카테고리의 첫 번째 예시 전체 출력
        print("-" * 80)

# 빈 텍스트 확인
empty_texts = df["text"].str.strip() == ""
print(f"\n빈 텍스트 수: {empty_texts.sum()}")

# 5. 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], 
    df["encoded_label"],
    test_size=0.2, 
    random_state=42
)

# 6. 토크나이저 및 시퀀스 변환
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# 7. LSTM + Attention 모델 정의 (Addons 제거)
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=max_words, output_dim=128)(input_layer)
lstm_layer = LSTM(128, return_sequences=True)(embedding_layer)

# 기본 Attention 적용
attention_layer = Attention()([lstm_layer, lstm_layer])

# 글로벌 평균 풀링
global_pooling = GlobalAveragePooling1D()(attention_layer)

# 출력 레이어
output_layer = Dense(len(label_encoder.classes_), activation="softmax")(global_pooling)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 모델 요약
model.summary()

# 8. 모델 학습
history = model.fit(
    X_train_pad, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 9. 모델 평가
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
print(f"테스트 정확도: {test_accuracy:.2f}")

# 10. 새로운 데이터 예측
new_texts = ["The pricing information is needed for the shipment."]
new_texts_seq = tokenizer.texts_to_sequences(new_texts)
new_texts_pad = pad_sequences(new_texts_seq, maxlen=max_len)

predictions = model.predict(new_texts_pad)
predicted_encoded_labels = predictions.argmax(axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_encoded_labels)

for text, pred_label in zip(new_texts, predicted_labels):
    category = category_names.get(pred_label, "Unknown")
    print(f"\n텍스트: {text}")
    print(f"예측 카테고리: {category} (라벨: {pred_label})")