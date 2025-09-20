"""
IoT 예측 유지보수 시스템 유틸리티 함수
공통으로 사용되는 함수들을 모아놓은 모듈
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import uuid
from functools import wraps
import time

from config import config, sensor_config, alert_config


def setup_logging(name: str = None, level: str = None) -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger(name or __name__)
    
    if not logger.handlers:
        level = level or config.LOG_LEVEL
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 파일 핸들러
        log_file = os.path.join(config.LOG_DIR, f"{name or 'app'}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # 포맷터
        formatter = logging.Formatter(config.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)
    
    return logger


def timer(func):
    """함수 실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"{func.__name__} 실패 (시도 {attempt + 1}/{max_attempts}): {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def validate_sensor_data(data: Dict) -> bool:
    """센서 데이터 유효성 검증"""
    try:
        # 필수 필드 확인
        required_fields = ['device_id', 'timestamp', 'sensors']
        for field in required_fields:
            if field not in data:
                return False
        
        # 센서 데이터 범위 확인
        sensors = data.get('sensors', {})
        for sensor_name, value in sensors.items():
            if sensor_name in sensor_config.SENSOR_RANGES:
                min_val, max_val = sensor_config.SENSOR_RANGES[sensor_name]
                if not (min_val <= value <= max_val * 2):  # 2배까지 허용 (이상값 포함)
                    print(f"센서 {sensor_name} 값 {value}이 허용 범위를 벗어남")
                    return False
        
        return True
    
    except Exception as e:
        print(f"센서 데이터 검증 오류: {e}")
        return False


def normalize_sensor_data(data: Dict) -> Dict:
    """센서 데이터 정규화"""
    normalized = data.copy()
    
    try:
        sensors = normalized.get('sensors', {})
        for sensor_name, value in sensors.items():
            if sensor_name in sensor_config.SENSOR_BASELINES:
                baseline = sensor_config.SENSOR_BASELINES[sensor_name]
                # 0-1 범위로 정규화 (기준값 대비)
                normalized_value = value / (baseline * 2)  # 기준값의 2배를 최대값으로
                sensors[sensor_name] = max(0, min(1, normalized_value))
        
        normalized['sensors'] = sensors
        
    except Exception as e:
        print(f"데이터 정규화 오류: {e}")
    
    return normalized


def calculate_health_score(sensors: Dict) -> float:
    """센서 데이터로부터 건강도 계산"""
    try:
        if not sensors:
            return 100.0
        
        deviations = []
        
        for sensor_name, value in sensors.items():
            if sensor_name in sensor_config.SENSOR_BASELINES:
                baseline = sensor_config.SENSOR_BASELINES[sensor_name]
                deviation = abs(value - baseline) / baseline
                deviations.append(deviation)
        
        if not deviations:
            return 100.0
        
        # 평균 편차를 건강도로 변환 (편차가 클수록 건강도 낮음)
        avg_deviation = np.mean(deviations)
        health_score = max(0, 100 - (avg_deviation * 100))
        
        return health_score
    
    except Exception as e:
        print(f"건강도 계산 오류: {e}")
        return 50.0  # 기본값


def calculate_anomaly_score(sensors: Dict, health_score: float) -> float:
    """이상 점수 계산"""
    try:
        # 건강도 기반 이상 점수
        health_anomaly = (100 - health_score) / 100
        
        # 센서값 기반 이상 점수
        sensor_anomalies = []
        
        for sensor_name, value in sensors.items():
            if sensor_name in sensor_config.SENSOR_RANGES:
                min_val, max_val = sensor_config.SENSOR_RANGES[sensor_name]
                
                if value < min_val:
                    anomaly = (min_val - value) / min_val
                elif value > max_val:
                    anomaly = (value - max_val) / max_val
                else:
                    anomaly = 0
                
                sensor_anomalies.append(anomaly)
        
        sensor_anomaly = np.mean(sensor_anomalies) if sensor_anomalies else 0
        
        # 건강도와 센서 이상을 결합
        final_anomaly = (health_anomaly * 0.7) + (sensor_anomaly * 0.3)
        
        return min(1.0, final_anomaly)
    
    except Exception as e:
        print(f"이상 점수 계산 오류: {e}")
        return 0.0


def determine_status(health_score: float, anomaly_score: float) -> str:
    """건강도와 이상 점수로부터 상태 결정"""
    if health_score >= sensor_config.HEALTH_THRESHOLDS['normal']:
        return 'normal'
    elif health_score >= sensor_config.HEALTH_THRESHOLDS['warning']:
        return 'warning'
    else:
        return 'critical'


def generate_device_id(prefix: str = "DEVICE") -> str:
    """고유한 디바이스 ID 생성"""
    timestamp = datetime.now().strftime("%Y%m%d")
    random_part = str(uuid.uuid4())[:8].upper()
    return f"{prefix}_{timestamp}_{random_part}"


def create_alert(device_id: str, alert_type: str, message: str, 
                priority: str = 'medium', metadata: Dict = None) -> Dict:
    """알림 생성"""
    alert = {
        'id': str(uuid.uuid4()),
        'device_id': device_id,
        'type': alert_type,
        'message': message,
        'priority': priority,
        'timestamp': datetime.now().isoformat(),
        'status': 'active',
        'metadata': metadata or {}
    }
    
    return alert


def save_json(data: Any, filepath: str) -> bool:
    """JSON 파일 저장"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        print(f"JSON 저장 오류: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """JSON 파일 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없음: {filepath}")
        return None
    except Exception as e:
        print(f"JSON 로드 오류: {e}")
        return None


def save_pickle(data: Any, filepath: str) -> bool:
    """Pickle 파일 저장"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Pickle 저장 오류: {e}")
        return False


def load_pickle(filepath: str) -> Optional[Any]:
    """Pickle 파일 로드"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없음: {filepath}")
        return None
    except Exception as e:
        print(f"Pickle 로드 오류: {e}")
        return None


def get_file_hash(filepath: str) -> Optional[str]:
    """파일 해시값 계산"""
    try:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    except Exception as e:
        print(f"파일 해시 계산 오류: {e}")
        return None


def format_bytes(bytes_value: int) -> str:
    """바이트를 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """초를 읽기 쉬운 시간 형태로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        return f"{seconds/60:.1f}분"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}시간"
    else:
        return f"{seconds/86400:.1f}일"


def get_system_info() -> Dict:
    """시스템 정보 수집"""
    import psutil
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'used': disk.used,
                'percent': (disk.used / disk.total) * 100
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"시스템 정보 수집 오류: {e}")
        return {}


def clean_old_files(directory: str, days_old: int = 30) -> int:
    """오래된 파일 정리"""
    try:
        count = 0
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if file_time < cutoff_date:
                    os.remove(filepath)
                    count += 1
        
        return count
    except Exception as e:
        print(f"파일 정리 오류: {e}")
        return 0


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """DataFrame 유효성 검증"""
    try:
        # 비어있는지 확인
        if df.empty:
            return False
        
        # 필수 컬럼 확인
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            print(f"누락된 컬럼: {missing_columns}")
            return False
        
        # null 값 확인
        if df[required_columns].isnull().any().any():
            print("필수 컬럼에 null 값이 있습니다")
            return False
        
        return True
    
    except Exception as e:
        print(f"DataFrame 검증 오류: {e}")
        return False


def create_time_features(df: pd.DataFrame, time_column: str = 'timestamp') -> pd.DataFrame:
    """시간 기반 특성 생성"""
    try:
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        
        # 시간 특성 추가
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek
        df['month'] = df[time_column].dt.month
        df['quarter'] = df[time_column].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 시간대 구분
        df['time_period'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'])
        
        return df
    
    except Exception as e:
        print(f"시간 특성 생성 오류: {e}")
        return df


def calculate_rolling_features(df: pd.DataFrame, columns: List[str], 
                             windows: List[int] = None) -> pd.DataFrame:
    """롤링 통계 특성 계산"""
    try:
        df = df.copy()
        windows = windows or [5, 15, 30]
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # 롤링 평균
                    df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    
                    # 롤링 표준편차
                    df[f'{col}_std_{window}'] = df[col].rolling(window, min_periods=1).std().fillna(0)
                    
                    # 롤링 최소/최대
                    df[f'{col}_min_{window}'] = df[col].rolling(window, min_periods=1).min()
                    df[f'{col}_max_{window}'] = df[col].rolling(window, min_periods=1).max()
                
                # 트렌드 (차분)
                df[f'{col}_diff'] = df[col].diff().fillna(0)
                df[f'{col}_pct_change'] = df[col].pct_change().fillna(0)
        
        return df
    
    except Exception as e:
        print(f"롤링 특성 계산 오류: {e}")
        return df


if __name__ == "__main__":
    # 유틸리티 함수 테스트
    logger = setup_logging("test")
    logger.info("유틸리티 함수 테스트 시작")
    
    # 테스트 센서 데이터
    test_sensors = {
        'temperature': 75.0,
        'vibration_x': 1.2,
        'current': 18.5,
        'pressure': 2.8
    }
    
    # 건강도 계산 테스트
    health = calculate_health_score(test_sensors)
    print(f"건강도: {health:.1f}%")
    
    # 이상 점수 계산 테스트
    anomaly = calculate_anomaly_score(test_sensors, health)
    print(f"이상 점수: {anomaly:.3f}")
    
    # 상태 결정 테스트
    status = determine_status(health, anomaly)
    print(f"상태: {status}")
    
    # 디바이스 ID 생성 테스트
    device_id = generate_device_id()
    print(f"생성된 디바이스 ID: {device_id}")
    
    # 알림 생성 테스트
    alert = create_alert(device_id, "health_low", "건강도가 임계값 이하로 떨어졌습니다.")
    print(f"생성된 알림: {alert['id']}")
    
    # 시스템 정보 테스트
    system_info = get_system_info()
    if system_info:
        print(f"CPU 사용률: {system_info['cpu_percent']:.1f}%")
        print(f"메모리 사용률: {system_info['memory']['percent']:.1f}%")
    
    logger.info("유틸리티 함수 테스트 완료")