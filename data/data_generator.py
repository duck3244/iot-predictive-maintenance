"""
IoT 센서 데이터 생성기
제조업 장비의 센서 데이터를 시뮬레이션합니다.
"""

import numpy as np
import pandas as pd
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid


class IoTSensorDataGenerator:
    """IoT 센서 데이터를 생성하는 클래스"""
    
    def __init__(self, device_id: str = None, failure_probability: float = 0.02):
        self.device_id = device_id or str(uuid.uuid4())[:8]
        self.failure_probability = failure_probability
        self.current_health = 100.0  # 0-100 건강도
        self.operating_hours = 0
        self.last_maintenance = datetime.now()
        
        # 센서 기준값 설정
        self.sensor_baselines = {
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
        
    def _simulate_degradation(self):
        """장비 성능 저하 시뮬레이션"""
        # 운영 시간에 따른 자연적 성능 저하
        degradation_rate = 0.001 * self.operating_hours
        
        # 랜덤 고장 발생
        if random.random() < self.failure_probability:
            degradation_rate += random.uniform(0.5, 2.0)
            
        self.current_health = max(0, self.current_health - degradation_rate)
        
    def _get_sensor_reading(self, sensor_name: str, base_value: float) -> float:
        """개별 센서 읽기값 생성"""
        # 건강도에 따른 편차 계산
        health_factor = (100 - self.current_health) / 100
        
        # 센서별 이상 패턴 정의
        anomaly_patterns = {
            'temperature': 1.5,       # 온도 상승
            'vibration_x': 3.0,       # 진동 증가
            'vibration_y': 3.0,
            'vibration_z': 2.5,
            'pressure': 0.8,          # 압력 감소
            'rotation_speed': 0.95,   # 속도 감소
            'current': 1.3,           # 전류 증가
            'voltage': 0.98,          # 전압 감소
            'power_factor': 0.9,      # 역률 감소
            'noise_level': 1.8        # 소음 증가
        }
        
        # 기본 노이즈 추가
        noise = np.random.normal(0, base_value * 0.02)
        
        # 건강도에 따른 이상값 추가
        if health_factor > 0.3:  # 건강도 70% 이하일 때
            anomaly_factor = anomaly_patterns.get(sensor_name, 1.0)
            if sensor_name in ['pressure', 'rotation_speed', 'voltage', 'power_factor']:
                # 감소하는 센서들
                anomaly_value = base_value * (anomaly_factor ** health_factor)
            else:
                # 증가하는 센서들
                anomaly_value = base_value * (1 + health_factor * (anomaly_factor - 1))
        else:
            anomaly_value = base_value
            
        return max(0, anomaly_value + noise)
    
    def generate_sensor_data(self) -> Dict:
        """센서 데이터 생성"""
        self._simulate_degradation()
        self.operating_hours += 1/60  # 1분 단위
        
        # 센서 데이터 생성
        sensor_data = {}
        for sensor_name, base_value in self.sensor_baselines.items():
            sensor_data[sensor_name] = round(
                self._get_sensor_reading(sensor_name, base_value), 2
            )
        
        # 계산된 특성 추가
        sensor_data['power_consumption'] = round(
            sensor_data['voltage'] * sensor_data['current'] * sensor_data['power_factor'] / 1000, 2
        )
        
        # 이상 점수 계산 (0-1, 1에 가까울수록 이상)
        anomaly_score = min(1.0, (100 - self.current_health) / 70)
        
        # 최종 데이터 패키지
        data = {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'operating_hours': round(self.operating_hours, 2),
            'health_score': round(self.current_health, 2),
            'anomaly_score': round(anomaly_score, 3),
            'sensors': sensor_data,
            'status': 'normal' if self.current_health > 70 else 'warning' if self.current_health > 30 else 'critical'
        }
        
        return data
    
    def generate_historical_data(self, days: int = 30, interval_minutes: int = 1) -> pd.DataFrame:
        """과거 데이터 생성"""
        records = []
        start_time = datetime.now() - timedelta(days=days)
        
        # 초기 건강도 설정
        self.current_health = random.uniform(70, 100)
        self.operating_hours = 0
        
        total_intervals = days * 24 * 60 // interval_minutes
        
        for i in range(total_intervals):
            current_time = start_time + timedelta(minutes=i * interval_minutes)
            data = self.generate_sensor_data()
            data['timestamp'] = current_time.isoformat()
            
            # 데이터 평탄화
            flat_data = {
                'device_id': data['device_id'],
                'timestamp': data['timestamp'],
                'operating_hours': data['operating_hours'],
                'health_score': data['health_score'],
                'anomaly_score': data['anomaly_score'],
                'status': data['status']
            }
            
            # 센서 데이터 평탄화
            for sensor_name, value in data['sensors'].items():
                flat_data[sensor_name] = value
                
            records.append(flat_data)
            
        return pd.DataFrame(records)


def generate_sample_dataset():
    """샘플 데이터셋 생성"""
    devices = []
    all_data = []
    
    # 5개 장비의 데이터 생성
    for i in range(5):
        device_id = f"DEVICE_{i+1:03d}"
        generator = IoTSensorDataGenerator(
            device_id=device_id, 
            failure_probability=random.uniform(0.01, 0.05)
        )
        
        # 각 장비별 30일 데이터 생성
        device_data = generator.generate_historical_data(days=30)
        all_data.append(device_data)
        devices.append(generator)
    
    # 모든 데이터 결합
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
    
    return combined_data, devices


if __name__ == "__main__":
    # 샘플 데이터 생성 및 저장
    print("IoT 센서 데이터 생성 중...")
    data, devices = generate_sample_dataset()
    
    # CSV로 저장
    data.to_csv('iot_sensor_data.csv', index=False)
    print(f"데이터 생성 완료: {len(data)}개 레코드")
    print(f"파일 저장: iot_sensor_data.csv")
    
    # 실시간 데이터 생성 예시
    print("\n실시간 데이터 생성 예시:")
    generator = IoTSensorDataGenerator("DEVICE_DEMO")
    
    for i in range(5):
        real_time_data = generator.generate_sensor_data()
        print(f"시점 {i+1}: 건강도 {real_time_data['health_score']}%, "
              f"상태: {real_time_data['status']}")
        time.sleep(1)
