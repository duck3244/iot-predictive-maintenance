"""
IoT 예측 유지보수 시스템 설정 파일
모든 시스템 설정을 중앙에서 관리합니다.
"""

import os
from datetime import timedelta


class Config:
    """기본 설정"""
    
    # 프로젝트 정보
    PROJECT_NAME = "IoT 예측 유지보수 시스템"
    VERSION = "2.0.0"
    DESCRIPTION = "TensorFlow 2.0 기반 IoT 예측 유지보수 시스템"
    
    # 파일 경로
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # 디렉토리 생성
    for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # 데이터베이스 설정
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///iot_data.db')
    
    # API 설정
    SECRET_KEY = os.environ.get('SECRET_KEY', 'iot-predictive-maintenance-secret-key-2024')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    API_HOST = os.environ.get('API_HOST', '0.0.0.0')
    API_PORT = int(os.environ.get('API_PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # Streamlit 설정
    STREAMLIT_HOST = os.environ.get('STREAMLIT_HOST', '0.0.0.0')
    STREAMLIT_PORT = int(os.environ.get('STREAMLIT_PORT', 8501))
    
    # 로깅 설정
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class IoTSensorConfig:
    """IoT 센서 관련 설정"""
    
    # 기본 센서 설정
    SENSOR_BASELINES = {
        'temperature': 65.0,      # 온도 (°C)
        'vibration_x': 0.5,       # X축 진동 (mm/s)
        'vibration_y': 0.5,       # Y축 진동 (mm/s)
        'vibration_z': 0.3,       # Z축 진동 (mm/s)
        'pressure': 2.5,          # 압력 (bar)
        'rotation_speed': 1800,   # 회전속도 (RPM)
        'current': 15.0,          # 전류 (A)
        'voltage': 220.0,         # 전압 (V)
        'power_factor': 0.95,     # 역률
        'noise_level': 45.0       # 소음 (dB)
    }
    
    # 센서 허용 범위 (정상 범위)
    SENSOR_RANGES = {
        'temperature': (40, 90),
        'vibration_x': (0.1, 2.0),
        'vibration_y': (0.1, 2.0),
        'vibration_z': (0.1, 1.5),
        'pressure': (1.0, 4.0),
        'rotation_speed': (1500, 2100),
        'current': (10, 25),
        'voltage': (200, 240),
        'power_factor': (0.8, 1.0),
        'noise_level': (35, 65)
    }
    
    # 데이터 생성 설정
    DEFAULT_FAILURE_PROBABILITY = 0.02
    DATA_GENERATION_INTERVAL = 1  # 초
    HISTORICAL_DATA_DAYS = 30
    HISTORICAL_DATA_INTERVAL = 5  # 분
    
    # 건강도 임계값
    HEALTH_THRESHOLDS = {
        'normal': 70,    # 70% 이상
        'warning': 30,   # 30-70%
        'critical': 0    # 30% 미만
    }
    
    # 이상 점수 임계값
    ANOMALY_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.7,
        'high': 1.0
    }


class ModelConfig:
    """머신러닝 모델 관련 설정"""
    
    # 모델 아키텍처
    SEQUENCE_LENGTH = 60        # 입력 시퀀스 길이 (분)
    PREDICTION_HORIZON = 10     # 예측 기간 (분)
    
    # LSTM 모델 설정
    LSTM_UNITS = [128, 64, 32]  # LSTM 레이어 유닛 수
    DROPOUT_RATE = 0.2          # 드롭아웃 비율
    DENSE_UNITS = [64, 32]      # Dense 레이어 유닛 수
    
    # 훈련 설정
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001
    
    # 조기 종료 설정
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # 모델 파일명
    DEFAULT_MODEL_NAME = "iot_predictive_model"
    MODEL_FILE_EXTENSION = ".h5"
    SCALER_FILE_EXTENSION = "_scaler.pkl"
    METADATA_FILE_EXTENSION = "_metadata.json"
    
    # 특성 엔지니어링
    ROLLING_WINDOWS = [5, 15]   # 롤링 윈도우 크기
    
    # 예측 임계값
    MAINTENANCE_THRESHOLD = 0.5  # 유지보수 필요 임계값


class DashboardConfig:
    """대시보드 관련 설정"""
    
    # 페이지 설정
    PAGE_TITLE = "IoT 예측 유지보수 대시보드"
    PAGE_ICON = "🏭"
    LAYOUT = "wide"
    
    # 데이터 설정
    MAX_DATA_POINTS = 1000      # 메모리에 저장할 최대 데이터 포인트
    REFRESH_INTERVAL = 5        # 자동 새로고침 간격 (초)
    
    # 차트 설정
    CHART_HEIGHT = 400
    TIME_SERIES_HEIGHT = 200
    HEATMAP_HEIGHT = 400
    
    # 색상 설정
    STATUS_COLORS = {
        'normal': '#4CAF50',
        'warning': '#FF9800',
        'critical': '#F44336'
    }
    
    # 디바이스 설정
    DEFAULT_DEVICE_COUNT = 5
    
    # 알림 설정
    ENABLE_ALERTS = True
    ALERT_SOUND = False


class APIConfig:
    """API 관련 설정"""
    
    # CORS 설정
    CORS_ORIGINS = ["*"]
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS = ["Content-Type", "Authorization"]
    
    # API 버전
    API_VERSION = "v1"
    API_PREFIX = f"/api/{API_VERSION}"
    
    # 인증 설정
    DEFAULT_USERS = {
        "admin": "password123",
        "operator": "operator123",
        "viewer": "viewer123"
    }
    
    # 속도 제한
    RATE_LIMIT_PER_MINUTE = 60
    
    # 배치 처리 설정
    MAX_BATCH_SIZE = 100
    
    # 페이지네이션
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE = 200


class AlertConfig:
    """알림 관련 설정"""
    
    # 알림 유형
    ALERT_TYPES = {
        'health_low': "건강도 저하",
        'anomaly_high': "이상 점수 높음",
        'prediction_high': "고장 위험 높음",
        'sensor_fault': "센서 오류",
        'maintenance_due': "유지보수 필요"
    }
    
    # 알림 우선순위
    ALERT_PRIORITIES = {
        'low': 1,
        'medium': 2,
        'high': 3,
        'critical': 4
    }
    
    # 알림 임계값
    HEALTH_ALERT_THRESHOLD = 70
    ANOMALY_ALERT_THRESHOLD = 0.7
    PREDICTION_ALERT_THRESHOLD = 0.8
    
    # 알림 쿨다운 (같은 디바이스의 같은 알림 재전송 방지)
    ALERT_COOLDOWN_MINUTES = 15
    
    # 이메일 설정 (선택사항)
    EMAIL_ENABLED = False
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', '')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
    EMAIL_FROM = os.environ.get('EMAIL_FROM', 'iot-system@company.com')
    EMAIL_TO = os.environ.get('EMAIL_TO', '').split(',')


class TestConfig(Config):
    """테스트 환경 설정"""
    
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    
    # 테스트용 축소된 설정
    EPOCHS = 5
    BATCH_SIZE = 16
    SEQUENCE_LENGTH = 30
    HISTORICAL_DATA_DAYS = 7


def get_config(env='development'):
    """환경별 설정 반환"""
    configs = {
        'development': Config,
        'production': Config,
        'testing': TestConfig
    }
    
    return configs.get(env, Config)


# 전역 설정 인스턴스
config = get_config(os.environ.get('ENVIRONMENT', 'development'))
sensor_config = IoTSensorConfig()
model_config = ModelConfig()
dashboard_config = DashboardConfig()
api_config = APIConfig()
alert_config = AlertConfig()


if __name__ == "__main__":
    print(f"프로젝트: {config.PROJECT_NAME}")
    print(f"버전: {config.VERSION}")
    print(f"디버그 모드: {config.DEBUG}")
    print(f"데이터 디렉토리: {config.DATA_DIR}")
    print(f"모델 디렉토리: {config.MODEL_DIR}")
