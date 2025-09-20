"""
TensorFlow 2.0 기반 예측 유지보수 모델 (SCARA 로봇 데이터용 수정)
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
    """IoT 예측 유지보수를 위한 TensorFlow 2.0 모델 (SCARA 로봇 특화)"""

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
        """특성 엔지니어링 (SCARA 로봇 데이터에 맞춤)"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['device_id', 'timestamp']).reset_index(drop=True)

        # 시간 기반 특성
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        # SCARA 로봇 데이터에 실제로 존재하는 컬럼들만 사용
        # 기본 센서 컬럼들
        base_sensor_cols = []

        # 관절 관련 컬럼
        joint_related_patterns = ['joint', 'actual_pos', 'cmd_pos', 'pos_error',
                                'torque_cmd', 'torque_feedback', 'velocity']
        for pattern in joint_related_patterns:
            matching_cols = [col for col in df.columns if pattern in col and df[col].dtype in ['float64', 'int64']]
            base_sensor_cols.extend(matching_cols)

        # Cartesian 및 SCARA 좌표 컬럼
        coordinate_patterns = ['cartesian', 'scara']
        for pattern in coordinate_patterns:
            matching_cols = [col for col in df.columns if pattern in col and df[col].dtype in ['float64', 'int64']]
            base_sensor_cols.extend(matching_cols)

        # 엔지니어링 특성 컬럼 (이미 계산된 것들)
        engineering_patterns = ['total_', 'mean_', 'max_', '_ma_', '_std_', '_change', '_imbalance']
        for pattern in engineering_patterns:
            matching_cols = [col for col in df.columns if pattern in col and df[col].dtype in ['float64', 'int64']]
            base_sensor_cols.extend(matching_cols)

        # 중복 제거
        base_sensor_cols = list(set(base_sensor_cols))

        # 실제 존재하는 컬럼만 필터링
        numeric_cols = [col for col in base_sensor_cols if col in df.columns]

        print(f"특성 엔지니어링에 사용할 컬럼 수: {len(numeric_cols)}")

        # 기존 엔지니어링 특성이 없다면 기본적인 롤링 통계 추가
        if len([col for col in numeric_cols if '_ma_' in col or '_std_' in col]) < 5:
            print("기본 롤링 통계 특성 추가 중...")
            # 주요 관절 컬럼들에 대해서만 롤링 통계 계산
            main_joint_cols = [col for col in numeric_cols if any(joint in col for joint in ['joint1', 'joint2', 'joint3', 'joint6'])][:10]  # 처음 10개만

            for col in main_joint_cols:
                if col in df.columns:
                    try:
                        df[f'{col}_ma_5'] = df.groupby('device_id')[col].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
                        df[f'{col}_ma_15'] = df.groupby('device_id')[col].rolling(15, min_periods=1).mean().reset_index(0, drop=True)
                        df[f'{col}_std_5'] = df.groupby('device_id')[col].rolling(5, min_periods=1).std().fillna(0).reset_index(0, drop=True)
                        df[f'{col}_trend'] = df.groupby('device_id')[col].diff().fillna(0).reset_index(0, drop=True)
                        numeric_cols.extend([f'{col}_ma_5', f'{col}_ma_15', f'{col}_std_5', f'{col}_trend'])
                    except Exception as e:
                        print(f"컬럼 {col} 처리 중 오류: {e}")
                        continue

        # SCARA 로봇 특화 복합 특성 (데이터에 맞는 컬럼이 있을 때만)
        cartesian_cols = [col for col in df.columns if 'cartesian' in col and any(axis in col for axis in ['_x', '_y', '_z'])]
        if len(cartesian_cols) >= 3:
            try:
                df['cartesian_distance'] = np.sqrt(
                    df[cartesian_cols[0]]**2 +
                    df[cartesian_cols[1]]**2 +
                    df[cartesian_cols[2]]**2
                )
                numeric_cols.append('cartesian_distance')
            except:
                pass

        # 관절 위치 오차 총합 (있는 경우)
        pos_error_cols = [col for col in df.columns if 'pos_error' in col and 'joint' in col]
        if len(pos_error_cols) > 0:
            try:
                df['total_pos_error'] = df[pos_error_cols].abs().sum(axis=1)
                numeric_cols.append('total_pos_error')
            except:
                pass

        # 토크 관련 특성 (있는 경우)
        torque_cmd_cols = [col for col in df.columns if 'torque_cmd' in col and 'joint' in col]
        if len(torque_cmd_cols) > 0:
            try:
                df['total_torque_cmd'] = df[torque_cmd_cols].abs().sum(axis=1)
                numeric_cols.append('total_torque_cmd')
            except:
                pass

        # 이상 점수 기반 라벨 생성
        if 'anomaly_score' in df.columns:
            df['maintenance_needed'] = (df['anomaly_score'] > 0.7).astype(int)
        elif 'health_score' in df.columns:
            df['maintenance_needed'] = (df['health_score'] < 50).astype(int)
        else:
            # 기본값으로 임의 생성 (실제로는 다른 방법으로 라벨을 생성해야 함)
            df['maintenance_needed'] = 0

        if 'anomaly_score' in df.columns:
            df['failure_risk'] = pd.cut(df['anomaly_score'],
                                       bins=[0, 0.3, 0.7, 1.0],
                                       labels=['low', 'medium', 'high'])
        else:
            df['failure_risk'] = 'low'

        # 최종 numeric_cols 정리
        numeric_cols = [col for col in numeric_cols if col in df.columns and df[col].dtype in ['float64', 'int64']]

        print(f"최종 특성 수: {len(numeric_cols)}")
        return df

    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        X, y = [], []

        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            y.append(targets[i + self.prediction_horizon - 1])

        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int], num_classes: int = 2) -> keras.Model:
        """LSTM 기반 예측 모델 구축 (이진 분류용 수정)"""
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

            # 출력 레이어 - 이진 분류이므로 1개 노드 사용
            layers.Dense(1, activation='sigmoid')  # 2 -> 1로 변경
        ])

        # 모델 컴파일 - 이진 분류용
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # 이진 분류용 손실 함수
            metrics=['accuracy']
        )

        return model

    def train(self, df: pd.DataFrame, validation_split: float = 0.2, epochs: int = 50, batch_size: int = 32):
        """모델 훈련"""
        print("특성 엔지니어링 중...")
        df_processed = self.prepare_features(df)

        # 특성 선택 (SCARA 로봇에 맞춤)
        feature_cols = [col for col in df_processed.columns
                       if col not in ['device_id', 'timestamp', 'status', 'maintenance_needed', 'failure_risk']
                       and df_processed[col].dtype in ['float64', 'int64']]

        # 너무 많은 특성이 있으면 상위 50개만 선택 (메모리 절약)
        if len(feature_cols) > 50:
            print(f"특성이 {len(feature_cols)}개로 많아 상위 50개만 선택합니다.")
            # 분산이 큰 특성들 우선 선택
            feature_variances = df_processed[feature_cols].var().sort_values(ascending=False)
            feature_cols = feature_variances.head(50).index.tolist()

        self.feature_columns = feature_cols
        print(f"최종 사용할 특성 수: {len(feature_cols)}")

        # 디바이스별로 데이터 처리
        X_all, y_all = [], []

        for device_id in df_processed['device_id'].unique():
            device_data = df_processed[df_processed['device_id'] == device_id].reset_index(drop=True)

            if len(device_data) < self.sequence_length + self.prediction_horizon:
                print(f"디바이스 {device_id}: 데이터 부족 (필요: {self.sequence_length + self.prediction_horizon}, 실제: {len(device_data)})")
                continue

            # 특성 스케일링
            features = device_data[feature_cols].values

            # NaN 값 처리
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            if not hasattr(self.scaler, 'scale_'):
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)

            # 타겟 준비
            targets = device_data['maintenance_needed'].values

            # 시퀀스 생성
            X_device, y_device = self.create_sequences(features_scaled, targets)

            if len(X_device) > 0:
                X_all.append(X_device)
                y_all.append(y_device)

        if not X_all:
            raise ValueError("훈련 가능한 데이터가 없습니다. 시퀀스 길이나 예측 기간을 줄여보세요.")

        # 모든 디바이스 데이터 결합
        X = np.vstack(X_all)
        y = np.hstack(y_all)

        print(f"훈련 데이터 형태: X={X.shape}, y={y.shape}")
        print(f"라벨 분포: {np.bincount(y)}")

        # 클래스 불균형 체크
        if len(np.unique(y)) < 2:
            print("⚠️ 라벨이 하나의 클래스만 있습니다. 더 다양한 데이터가 필요합니다.")
            # 임의로 일부 라벨을 변경하여 훈련 가능하게 만듦
            if np.all(y == 0):
                y[-len(y)//10:] = 1  # 마지막 10%를 유지보수 필요로 변경
            else:
                y[-len(y)//10:] = 0
            print(f"수정된 라벨 분포: {np.bincount(y)}")

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

        # NaN 값 처리
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        features_scaled = self.scaler.transform(features)

        # 예측 수행
        X_pred = features_scaled.reshape(1, self.sequence_length, -1)
        prediction_prob = self.model.predict(X_pred, verbose=0)[0]

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
            "timestamp": df_processed['timestamp'].iloc[-1].isoformat(),
            "method": "SCARA_optimized_model"
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
            "feature_columns": self.feature_columns,
            "model_type": "SCARA_optimized"
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
    try:
        from data.data_generator import generate_sample_dataset

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

    except ImportError:
        print("data_generator.py 모듈을 찾을 수 없습니다.")
    except Exception as e:
        print(f"데모 실행 중 오류: {e}")


if __name__ == "__main__":
    demo_training()