"""
SCARA 로봇 데이터로 AI 예측 모델 훈련
processed_scara_data_*.csv 파일을 사용하여 예측 모델을 훈련합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 안전한 import
try:
    from predictive_model import IoTPredictiveMaintenanceModel
    MODEL_AVAILABLE = True
except ImportError:
    print("⚠️  predictive_model.py를 찾을 수 없습니다.")
    MODEL_AVAILABLE = False

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} 로드됨")
    TF_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow를 찾을 수 없습니다. pip install tensorflow")
    TF_AVAILABLE = False


def load_scara_data(csv_file_path):
    """SCARA 로봇 데이터 로드 및 검증"""
    
    print(f"📁 데이터 로드: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file_path}")
        return None
    
    try:
        # CSV 파일 로드
        df = pd.read_csv(csv_file_path)
        print(f"✅ 데이터 로드 성공: {len(df):,}행 x {len(df.columns)}열")
        
        # 기본 정보 출력
        file_size = os.path.getsize(csv_file_path) / (1024*1024)  # MB
        print(f"📏 파일 크기: {file_size:.1f} MB")
        
        # 타임스탬프 변환
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_range = df['timestamp'].max() - df['timestamp'].min()
            print(f"⏰ 시간 범위: {time_range}")
            print(f"   시작: {df['timestamp'].min()}")
            print(f"   종료: {df['timestamp'].max()}")
        
        # 데이터 품질 확인
        print(f"\n📊 데이터 품질 체크:")
        
        # 결측값 확인
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        
        if len(missing_cols) > 0:
            print(f"   ⚠️  결측값: {len(missing_cols)}개 컬럼")
            for col, count in missing_cols.head(5).items():
                print(f"     - {col}: {count}개 ({count/len(df)*100:.1f}%)")
        else:
            print(f"   ✅ 결측값 없음")
        
        # 주요 통계
        if 'health_score' in df.columns:
            print(f"   건강도: 평균 {df['health_score'].mean():.1f}% (범위: {df['health_score'].min():.1f}% ~ {df['health_score'].max():.1f}%)")
        
        if 'anomaly_score' in df.columns:
            print(f"   이상점수: 평균 {df['anomaly_score'].mean():.3f} (범위: {df['anomaly_score'].min():.3f} ~ {df['anomaly_score'].max():.3f})")
        
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            print(f"   상태 분포: {dict(status_counts)}")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None


def analyze_scara_features(df):
    """SCARA 로봇 특성 분석"""
    
    print(f"\n🔍 SCARA 로봇 특성 분석")
    print("="*50)
    
    # 특성 카테고리별 분류
    feature_categories = {
        '시간': [col for col in df.columns if col in ['timestamp', 'hour', 'day_of_week', 'minute']],
        '관절 위치': [col for col in df.columns if 'actual_pos' in col and 'joint' in col],
        '위치 명령': [col for col in df.columns if 'cmd_pos' in col and 'joint' in col],
        '위치 오차': [col for col in df.columns if 'pos_error' in col and 'joint' in col],
        '토크 명령': [col for col in df.columns if 'torque_cmd' in col and 'joint' in col],
        '토크 피드백': [col for col in df.columns if 'torque_feedback' in col and 'joint' in col],
        'Cartesian': [col for col in df.columns if 'cartesian' in col],
        'SCARA 좌표': [col for col in df.columns if 'scara' in col],
        '엔지니어링': [col for col in df.columns if any(x in col for x in ['total_', 'mean_', 'max_', '_ma_', '_std_', '_change'])],
        '타겟 변수': [col for col in df.columns if col in ['health_score', 'anomaly_score', 'status', 'device_id']]
    }
    
    print(f"📋 특성 카테고리:")
    total_features = 0
    for category, features in feature_categories.items():
        if features:
            print(f"   {category}: {len(features)}개")
            total_features += len(features)
            # 샘플 특성 출력
            if len(features) <= 5:
                for feature in features:
                    print(f"     - {feature}")
            else:
                for feature in features[:3]:
                    print(f"     - {feature}")
                print(f"     ... 및 {len(features)-3}개 추가")
    
    print(f"\n📊 총 특성 수: {total_features}개")
    
    # 상관관계가 높은 주요 특성들 찾기
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 10:  # 너무 많으면 샘플링
        sample_cols = np.random.choice(numeric_cols, 10, replace=False)
        corr_matrix = df[sample_cols].corr()
        
        print(f"\n🔗 주요 특성 상관관계 (샘플 10개):")
        # 높은 상관관계 찾기
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # 높은 상관관계
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            for col1, col2, corr_val in high_corr_pairs[:5]:
                print(f"   {col1} ↔ {col2}: {corr_val:.3f}")
        else:
            print(f"   높은 상관관계 (>0.7) 없음")
    
    return feature_categories


def prepare_model_data(df):
    """모델 훈련용 데이터 준비"""
    
    print(f"\n🔧 모델 훈련용 데이터 준비")
    print("="*40)
    
    # 필요한 컬럼 확인
    required_cols = ['timestamp', 'health_score', 'anomaly_score', 'status']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 필수 컬럼 누락: {missing_cols}")
        return None
    
    # 디바이스 ID 확인/생성
    if 'device_id' not in df.columns:
        df['device_id'] = 'SCARA_ROBOT_001'
        print(f"📝 device_id 컬럼 생성: SCARA_ROBOT_001")
    
    # 타겟 변수 준비
    print(f"🎯 타겟 변수 분포:")
    
    # 이진 분류를 위한 maintenance_needed 생성
    if 'maintenance_needed' not in df.columns:
        # 건강도 70% 미만 또는 이상점수 0.5 이상이면 유지보수 필요
        df['maintenance_needed'] = (
            (df['health_score'] < 70) | (df['anomaly_score'] > 0.5)
        ).astype(int)
    
    maintenance_dist = df['maintenance_needed'].value_counts()
    print(f"   유지보수 필요: {maintenance_dist}")
    print(f"   비율: {maintenance_dist[1] / len(df) * 100:.1f}% 필요")
    
    # 특성 선택 (모델 성능을 위해)
    feature_cols = []
    
    # 주요 센서 데이터
    sensor_cols = [col for col in df.columns if any(keyword in col for keyword in 
                   ['actual_pos', 'pos_error', 'torque_cmd', 'cartesian'])]
    feature_cols.extend(sensor_cols)
    
    # 엔지니어링 특성 (주요한 것들만)
    engineering_cols = [col for col in df.columns if any(keyword in col for keyword in 
                       ['total_', 'mean_', 'max_', '_imbalance', '_std_5', '_ma_5'])]
    feature_cols.extend(engineering_cols)
    
    # 시간 특성
    time_cols = [col for col in df.columns if col in ['hour', 'day_of_week', 'minute']]
    feature_cols.extend(time_cols)
    
    # 중복 제거
    feature_cols = list(set(feature_cols))
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"📊 선택된 특성: {len(feature_cols)}개")
    print(f"   센서 데이터: {len(sensor_cols)}개")
    print(f"   엔지니어링: {len(engineering_cols)}개")
    print(f"   시간 특성: {len(time_cols)}개")
    
    # 데이터 정리
    model_df = df[['timestamp', 'device_id', 'health_score', 'anomaly_score', 
                   'status', 'maintenance_needed'] + feature_cols].copy()
    
    # 결측값 처리
    numeric_cols = model_df.select_dtypes(include=[np.number]).columns
    model_df[numeric_cols] = model_df[numeric_cols].fillna(method='ffill').fillna(0)
    
    print(f"✅ 모델용 데이터 준비 완료: {len(model_df)}행 x {len(model_df.columns)}열")
    
    return model_df


def train_scara_model(model_df):
    """SCARA 로봇 예측 모델 훈련"""
    
    if not MODEL_AVAILABLE or not TF_AVAILABLE:
        print("❌ 모델 훈련을 위한 라이브러리가 없습니다.")
        return None
    
    print(f"\n🤖 SCARA 로봇 AI 모델 훈련 시작")
    print("="*50)
    
    try:
        # 모델 초기화 (SCARA 로봇에 최적화된 설정)
        model = IoTPredictiveMaintenanceModel(
            sequence_length=30,  # 5분 시퀀스 (30 x 10초)
            prediction_horizon=18  # 3분 후 예측 (18 x 10초)
        )
        
        print(f"🏗️  모델 구성:")
        print(f"   시퀀스 길이: {model.sequence_length} (5분)")
        print(f"   예측 기간: {model.prediction_horizon} (3분 후)")
        
        # 훈련 실행
        print(f"\n🏋️  모델 훈련 중...")
        print(f"   데이터: {len(model_df):,}개 시점")
        print(f"   특성: {len(model_df.columns)}개")
        
        history = model.train(
            model_df,
            epochs=50,  # SCARA 데이터에 적합한 에포크
            batch_size=32,
            validation_split=0.2
        )
        
        print(f"✅ 훈련 완료!")
        
        # 모델 저장
        model_name = f"scara_robot_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save_model(model_name)
        print(f"💾 모델 저장: {model_name}")
        
        # 훈련 히스토리 시각화
        if hasattr(history, 'history'):
            plot_training_history(history, model_name)
        
        # 샘플 예측 수행
        sample_prediction(model, model_df)
        
        return model, model_name
        
    except Exception as e:
        print(f"❌ 모델 훈련 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def plot_training_history(history, model_name):
    """훈련 히스토리 시각화"""
    
    try:
        print(f"\n📊 훈련 결과 시각화...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'SCARA 로봇 모델 훈련 결과 - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. 손실 그래프
        ax1 = axes[0, 0]
        ax1.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('모델 손실 (Loss)', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 정확도 그래프
        ax2 = axes[0, 1]
        if 'accuracy' in history.history:
            ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
            ax2.set_title('모델 정확도 (Accuracy)', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Accuracy 데이터 없음', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('모델 정확도', fontweight='bold')
        
        # 3. 학습률 변화 (있는 경우)
        ax3 = axes[1, 0]
        if 'lr' in history.history:
            ax3.plot(history.history['lr'], color='green', linewidth=2)
            ax3.set_title('학습률 변화', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '학습률 데이터 없음', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('학습률 변화', fontweight='bold')
        
        # 4. 훈련 요약
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 최종 성능 메트릭
        final_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 'N/A'
        final_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 'N/A'
        total_epochs = len(history.history['loss'])
        
        summary_text = f"""
훈련 요약:
━━━━━━━━━━━━━━━━
총 에포크: {total_epochs}
최종 검증 손실: {final_loss:.4f if final_loss != 'N/A' else 'N/A'}
최종 검증 정확도: {final_acc:.4f if final_acc != 'N/A' else 'N/A'}

모델 구성:
• 아키텍처: LSTM + Dense
• 시퀀스 길이: 30 (5분)
• 예측 기간: 18 (3분 후)
• 배치 크기: 32

데이터셋:
• SCARA 로봇 센서 데이터
• 4개 관절 (J1, J2, J3, J6)
• 위치, 토크, 오차 정보
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 저장
        plot_filename = f"{model_name}_training_history.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"💾 훈련 그래프 저장: {plot_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️  시각화 오류: {e}")


def sample_prediction(model, model_df):
    """샘플 예측 수행"""
    
    try:
        print(f"\n🔮 샘플 예측 수행...")
        
        # 최근 데이터로 예측
        device_id = model_df['device_id'].iloc[0]
        recent_data = model_df.tail(500)  # 최근 500개 시점
        
        prediction = model.predict(recent_data, device_id)
        
        print(f"📱 SCARA 로봇 예측 결과:")
        print(f"   디바이스: {prediction['device_id']}")
        print(f"   유지보수 확률: {prediction['maintenance_probability']:.1%}")
        print(f"   유지보수 필요: {'✅ 예' if prediction['maintenance_needed'] else '❌ 아니오'}")
        print(f"   위험 수준: {prediction['risk_level']}")
        print(f"   예측 시점: {prediction.get('timestamp', 'N/A')}")
        
        # 위험 수준별 권장사항
        risk_recommendations = {
            'low': "정상 운영 상태입니다. 정기 점검만 수행하세요.",
            'medium': "주의가 필요합니다. 센서 상태를 모니터링하고 예방 정비를 계획하세요.",
            'high': "즉시 점검이 필요합니다. 운영을 중단하고 전문가의 진단을 받으세요."
        }
        
        recommendation = risk_recommendations.get(prediction['risk_level'], "상태를 확인하세요.")
        print(f"   💡 권장사항: {recommendation}")
        
        return prediction
        
    except Exception as e:
        print(f"⚠️  예측 오류: {e}")
        return None


def main():
    """메인 실행 함수"""
    
    print("🤖 SCARA 로봇 AI 모델 훈련 시스템")
    print("="*60)
    
    # 1. CSV 파일 찾기
    csv_files = [f for f in os.listdir('.') if f.startswith('processed_scara_data_') and f.endswith('.csv')]
    
    if not csv_files:
        print("❌ processed_scara_data_*.csv 파일을 찾을 수 없습니다.")
        print("먼저 fixed_xml_data_processor.py를 실행하여 데이터를 처리하세요.")
        return
    
    # 가장 최근 파일 선택
    latest_file = max(csv_files, key=lambda x: os.path.getmtime(x))
    print(f"📁 최신 데이터 파일: {latest_file}")
    
    # 2. 데이터 로드
    df = load_scara_data(latest_file)
    if df is None:
        return
    
    # 3. 특성 분석
    feature_categories = analyze_scara_features(df)
    
    # 4. 모델 데이터 준비
    model_df = prepare_model_data(df)
    if model_df is None:
        return
    
    # 5. 사용자 확인
    print(f"\n🚀 모델 훈련을 시작하시겠습니까?")
    print(f"   데이터: {len(model_df):,}개 시점")
    print(f"   특성: {len(model_df.columns)}개")
    print(f"   예상 시간: 5-15분")
    
    response = input(f"\n훈련 시작? (y/N): ")
    if response.lower() != 'y':
        print("훈련이 취소되었습니다.")
        return
    
    # 6. 모델 훈련
    model, model_name = train_scara_model(model_df)
    
    if model and model_name:
        print(f"\n🎉 SCARA 로봇 AI 모델 훈련 완료!")
        print(f"📁 저장된 파일들:")
        print(f"   🤖 모델: {model_name}.h5")
        print(f"   🔧 스케일러: {model_name}_scaler.pkl")
        print(f"   📄 메타데이터: {model_name}_metadata.json")
        print(f"   📊 훈련 그래프: {model_name}_training_history.png")
        
        print(f"\n💡 다음 단계:")
        print(f"1. 실시간 예측 시스템에 모델 통합")
        print(f"2. 대시보드에서 실시간 모니터링")
        print(f"3. 알림 시스템 설정")
        
        return model_name
    else:
        print(f"\n❌ 모델 훈련 실패")
        return None


if __name__ == "__main__":
    model_name = main()
