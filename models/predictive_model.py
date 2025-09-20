"""
TensorFlow 2.0 기반 예측 유지보수 모델
LSTM과 Dense Layer를 활용한 시계열 예측 모델
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class IoTPredictiveMaintenanceModel:
    """IoT 예측 유지보수를 위한 TensorFlow 2.0 모델"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 10):
        """
        Args:
            sequence_length: 입력 시퀀스 길이 (예: 60분)
            prediction_horizon: 예측 기간 (예: 10분 후)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.model_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 엔지니어링"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['device_id', 'timestamp']).reset_index(drop=True)
        
        # 시간 기반 특성
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # 롤링 통계 특성
        numeric_cols = ['temperature', 'vibration_x', 'vibration_y', 'vibration_z', 
                       'pressure', 'rotation_speed', 'current', 'voltage', 'power_factor', 'noise_level']
        
        for col in numeric_cols:
            df[f'{col}_ma_5'] = df.groupby('device_id')[col].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{col}_ma_15'] = df.groupby('device_id')[col].rolling(15, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{col}_std_5'] = df.groupby('device_id')[col].rolling(5, min_periods=1).std().fillna(0).reset_index(0, drop=True)
            df[f'{col}_trend'] = df.groupby('device_id')[col].diff().fillna(0).reset_index(0, drop=True)
        
        # 복합 특성
        df['vibration_total'] = np.sqrt(df['vibration_x']**2 + df['vibration_y']**2 + df['vibration_z']**2)
        df['power_efficiency'] = df['power_consumption'] / (df['rotation_speed'] / 1000)
        df['thermal_ratio'] = df['temperature'] / df['current']
        
        # 이상 점수 기반 라벨 생성
        df['maintenance_needed'] = (df['anomaly_score'] > 0.7).astype(int)
        df['failure_risk'] = pd.cut(df['anomaly_score'], 
                                   bins=[0, 0.3, 0.7, 1.0], 
                                   labels=['low', 'medium', 'high'])
        
        return df
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            y.append(targets[i + self.prediction_horizon - 1])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int], num_classes: int = 2) -> keras.Model:
        """LSTM 기반 예측 모델 구축"""
        model = keras.Sequential([
            # LSTM 레이어들
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense 레이어들
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # 출력 레이어
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        # 모델 컴파일
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        if num_classes > 2:
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
        return model
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2, epochs: int = 50, batch_size: int = 32):
        """모델 훈련"""
        print("특성 엔지니어링 중...")
        df_processed = self.prepare_features(df)
        
        # 특성 선택
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['device_id', 'timestamp', 'status', 'maintenance_needed', 'failure_risk']]
        
        self.feature_columns = feature_cols
        
        # 디바이스별로 데이터 처리
        X_all, y_all = [], []
        
        for device_id in df_processed['device_id'].unique():
            device_data = df_processed[df_processed['device_id'] == device_id].reset_index(drop=True)
            
            if len(device_data) < self.sequence_length + self.prediction_horizon:
                continue
                
            # 특성 스케일링
            features = device_data[feature_cols].values
            features_scaled = self.scaler.fit_transform(features)
            
            # 타겟 준비
            targets = device_data['maintenance_needed'].values
            
            # 시퀀스 생성
            X_device, y_device = self.create_sequences(features_scaled, targets)
            
            if len(X_device) > 0:
                X_all.append(X_device)
                y_all.append(y_device)
        
        # 모든 디바이스 데이터 결합
        X = np.vstack(X_all)
        y = np.hstack(y_all)
        
        print(f"훈련 데이터 형태: X={X.shape}, y={y.shape}")
        
        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # 모델 구축
        self.model = self.build_model((self.sequence_length, len(feature_cols)), num_classes=2)
        
        print("모델 구조:")
        self.model.summary()
        
        # 콜백 설정
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            )
        ]
        
        # 모델 훈련
        print("모델 훈련 시작...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model_trained = True
        
        # 훈련 결과 평가
        y_pred = (self.model.predict(X_val) > 0.5).astype(int).flatten()
        print("\n검증 세트 성능:")
        print(classification_report(y_val, y_pred))
        
        return history
    
    def predict(self, df: pd.DataFrame, device_id: str = None) -> Dict:
        """예측 수행"""
        if not self.model_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        df_processed = self.prepare_features(df)
        
        if device_id:
            device_data = df_processed[df_processed['device_id'] == device_id]
        else:
            device_data = df_processed
            
        if len(device_data) < self.sequence_length:
            return {"error": "충분한 데이터가 없습니다."}
        
        # 최근 데이터로 예측
        recent_data = device_data.tail(self.sequence_length)
        features = recent_data[self.feature_columns].values
        features_scaled = self.scaler.transform(features)
        
        # 예측 수행
        X_pred = features_scaled.reshape(1, self.sequence_length, -1)
        prediction_prob = self.model.predict(X_pred)[0]
        
        # 예측 결과
        maintenance_prob = prediction_prob[1] if len(prediction_prob) > 1 else prediction_prob[0]
        maintenance_needed = maintenance_prob > 0.5
        
        # 위험도 계산
        if maintenance_prob < 0.3:
            risk_level = "low"
        elif maintenance_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "device_id": device_id or "unknown",
            "maintenance_probability": float(maintenance_prob),
            "maintenance_needed": bool(maintenance_needed),
            "risk_level": risk_level,
            "prediction_horizon_minutes": self.prediction_horizon,
            "timestamp": df_processed['timestamp'].iloc[-1].isoformat()
        }
    
    def save_model(self, model_path: str = "iot_predictive_model"):
        """모델 저장"""
        if not self.model_trained:
            raise ValueError("훈련된 모델이 없습니다.")
        
        # TensorFlow 모델 저장
        self.model.save(f"{model_path}.h5")
        
        # 스케일러 저장
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        
        # 메타데이터 저장
        metadata = {
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "feature_columns": self.feature_columns
        }
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"모델이 저장되었습니다: {model_path}")
    
    def load_model(self, model_path: str = "iot_predictive_model"):
        """모델 로드"""
        # TensorFlow 모델 로드
        self.model = keras.models.load_model(f"{model_path}.h5")
        
        # 스케일러 로드
        self.scaler = joblib.load(f"{model_path}_scaler.pkl")
        
        # 메타데이터 로드
        with open(f"{model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.sequence_length = metadata["sequence_length"]
        self.prediction_horizon = metadata["prediction_horizon"]
        self.feature_columns = metadata["feature_columns"]
        self.model_trained = True
        
        print(f"모델이 로드되었습니다: {model_path}")
    
    def plot_training_history(self, history):
        """훈련 히스토리 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 손실 그래프
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 정확도 그래프
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


def demo_training():
    """모델 훈련 데모"""
    from data_generator import generate_sample_dataset
    
    print("샘플 데이터 생성 중...")
    data, _ = generate_sample_dataset()
    
    # 모델 초기화 및 훈련
    model = IoTPredictiveMaintenanceModel(sequence_length=30, prediction_horizon=5)
    
    print("모델 훈련 시작...")
    history = model.train(data, epochs=20, batch_size=16)
    
    # 모델 저장
    model.save_model("iot_predictive_model")
    
    # 훈련 히스토리 플롯
    model.plot_training_history(history)
    
    # 샘플 예측
    sample_device = data['device_id'].iloc[0]
    sample_data = data[data['device_id'] == sample_device].tail(100)
    
    prediction = model.predict(sample_data, sample_device)
    print(f"\n샘플 예측 결과:")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    demo_training()
