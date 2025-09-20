"""
IoT 데이터 저장소 관리
CSV, JSON, 메모리 기반 데이터 저장소를 관리합니다.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import sqlite3
import threading
from abc import ABC, abstractmethod

from config import config, dashboard_config
from utils import setup_logging, save_json, load_json, validate_dataframe


logger = setup_logging(__name__)


class DataStorage(ABC):
    """데이터 저장소 추상 클래스"""
    
    @abstractmethod
    def save_device_data(self, device_id: str, data: Dict) -> bool:
        """디바이스 데이터 저장"""
        pass
    
    @abstractmethod
    def get_device_data(self, device_id: str, count: int = None) -> pd.DataFrame:
        """디바이스 데이터 조회"""
        pass
    
    @abstractmethod
    def get_all_devices(self) -> List[str]:
        """모든 디바이스 ID 조회"""
        pass
    
    @abstractmethod
    def delete_device_data(self, device_id: str) -> bool:
        """디바이스 데이터 삭제"""
        pass


class MemoryStorage(DataStorage):
    """메모리 기반 데이터 저장소"""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or dashboard_config.MAX_DATA_POINTS
        self.data = defaultdict(lambda: deque(maxlen=self.max_size))
        self.lock = threading.RLock()
        logger.info(f"메모리 저장소 초기화 (최대 크기: {self.max_size})")
    
    def save_device_data(self, device_id: str, data: Dict) -> bool:
        """디바이스 데이터 저장"""
        try:
            with self.lock:
                # 데이터 평탄화
                flat_data = self._flatten_data(data)
                self.data[device_id].append(flat_data)
                return True
        except Exception as e:
            logger.error(f"메모리 저장 오류: {e}")
            return False
    
    def get_device_data(self, device_id: str, count: int = None) -> pd.DataFrame:
        """디바이스 데이터 조회"""
        try:
            with self.lock:
                if device_id not in self.data:
                    return pd.DataFrame()
                
                data_list = list(self.data[device_id])
                
                if count:
                    data_list = data_list[-count:]
                
                if not data_list:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data_list)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                
                return df
        except Exception as e:
            logger.error(f"메모리 조회 오류: {e}")
            return pd.DataFrame()
    
    def get_all_devices(self) -> List[str]:
        """모든 디바이스 ID 조회"""
        with self.lock:
            return list(self.data.keys())
    
    def delete_device_data(self, device_id: str) -> bool:
        """디바이스 데이터 삭제"""
        try:
            with self.lock:
                if device_id in self.data:
                    del self.data[device_id]
                    return True
                return False
        except Exception as e:
            logger.error(f"메모리 삭제 오류: {e}")
            return False
    
    def _flatten_data(self, data: Dict) -> Dict:
        """중첩 데이터 평탄화"""
        flat_data = {
            'device_id': data.get('device_id'),
            'timestamp': data.get('timestamp'),
            'operating_hours': data.get('operating_hours'),
            'health_score': data.get('health_score'),
            'anomaly_score': data.get('anomaly_score'),
            'status': data.get('status')
        }
        
        # 센서 데이터 평탄화
        sensors = data.get('sensors', {})
        for sensor_name, value in sensors.items():
            flat_data[sensor_name] = value
        
        return flat_data
    
    def get_memory_usage(self) -> Dict:
        """메모리 사용량 정보"""
        with self.lock:
            total_records = sum(len(device_data) for device_data in self.data.values())
            return {
                'devices_count': len(self.data),
                'total_records': total_records,
                'max_size_per_device': self.max_size,
                'estimated_memory_mb': total_records * 0.001  # 대략적인 추정
            }


class CSVStorage(DataStorage):
    """CSV 파일 기반 데이터 저장소"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.lock = threading.RLock()
        logger.info(f"CSV 저장소 초기화 (디렉토리: {self.data_dir})")
    
    def save_device_data(self, device_id: str, data: Dict) -> bool:
        """디바이스 데이터 저장"""
        try:
            with self.lock:
                filepath = self._get_device_filepath(device_id)
                flat_data = self._flatten_data(data)
                
                # 기존 파일이 있으면 추가, 없으면 새로 생성
                if os.path.exists(filepath):
                    existing_df = pd.read_csv(filepath)
                    new_df = pd.DataFrame([flat_data])
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = pd.DataFrame([flat_data])
                
                combined_df.to_csv(filepath, index=False)
                return True
        
        except Exception as e:
            logger.error(f"CSV 저장 오류: {e}")
            return False
    
    def get_device_data(self, device_id: str, count: int = None) -> pd.DataFrame:
        """디바이스 데이터 조회"""
        try:
            with self.lock:
                filepath = self._get_device_filepath(device_id)
                
                if not os.path.exists(filepath):
                    return pd.DataFrame()
                
                df = pd.read_csv(filepath)
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                
                if count:
                    df = df.tail(count)
                
                return df
        
        except Exception as e:
            logger.error(f"CSV 조회 오류: {e}")
            return pd.DataFrame()
    
    def get_all_devices(self) -> List[str]:
        """모든 디바이스 ID 조회"""
        try:
            device_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            devices = [f.replace('.csv', '') for f in device_files]
            return devices
        except Exception as e:
            logger.error(f"디바이스 목록 조회 오류: {e}")
            return []
    
    def delete_device_data(self, device_id: str) -> bool:
        """디바이스 데이터 삭제"""
        try:
            with self.lock:
                filepath = self._get_device_filepath(device_id)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
                return False
        except Exception as e:
            logger.error(f"CSV 삭제 오류: {e}")
            return False
    
    def _get_device_filepath(self, device_id: str) -> str:
        """디바이스 CSV 파일 경로"""
        return os.path.join(self.data_dir, f"{device_id}.csv")
    
    def _flatten_data(self, data: Dict) -> Dict:
        """중첩 데이터 평탄화"""
        flat_data = {
            'device_id': data.get('device_id'),
            'timestamp': data.get('timestamp'),
            'operating_hours': data.get('operating_hours'),
            'health_score': data.get('health_score'),
            'anomaly_score': data.get('anomaly_score'),
            'status': data.get('status')
        }
        
        # 센서 데이터 평탄화
        sensors = data.get('sensors', {})
        for sensor_name, value in sensors.items():
            flat_data[sensor_name] = value
        
        return flat_data
    
    def export_all_data(self, output_file: str = None) -> str:
        """모든 데이터를 하나의 파일로 내보내기"""
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.data_dir, f"all_data_{timestamp}.csv")
            
            all_data = []
            
            for device_id in self.get_all_devices():
                device_data = self.get_device_data(device_id)
                if not device_data.empty:
                    all_data.append(device_data)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df.to_csv(output_file, index=False)
                logger.info(f"전체 데이터 내보내기 완료: {output_file}")
                return output_file
            else:
                logger.warning("내보낼 데이터가 없습니다")
                return ""
        
        except Exception as e:
            logger.error(f"데이터 내보내기 오류: {e}")
            return ""


class SQLiteStorage(DataStorage):
    """SQLite 데이터베이스 기반 저장소"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(config.DATA_DIR, "iot_data.db")
        self.lock = threading.RLock()
        self._init_database()
        logger.info(f"SQLite 저장소 초기화 (파일: {self.db_path})")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 센서 데이터 테이블 생성
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sensor_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        operating_hours REAL,
                        health_score REAL,
                        anomaly_score REAL,
                        status TEXT,
                        sensor_data TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 인덱스 생성
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_timestamp ON sensor_data(device_id, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_id ON sensor_data(device_id)')
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"데이터베이스 초기화 오류: {e}")
    
    def save_device_data(self, device_id: str, data: Dict) -> bool:
        """디바이스 데이터 저장"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 센서 데이터를 JSON으로 직렬화
                    sensor_data_json = json.dumps(data.get('sensors', {}))
                    
                    cursor.execute('''
                        INSERT INTO sensor_data 
                        (device_id, timestamp, operating_hours, health_score, 
                         anomaly_score, status, sensor_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        device_id,
                        data.get('timestamp'),
                        data.get('operating_hours'),
                        data.get('health_score'),
                        data.get('anomaly_score'),
                        data.get('status'),
                        sensor_data_json
                    ))
                    
                    conn.commit()
                    return True
        
        except Exception as e:
            logger.error(f"SQLite 저장 오류: {e}")
            return False
    
    def get_device_data(self, device_id: str, count: int = None) -> pd.DataFrame:
        """디바이스 데이터 조회"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    query = '''
                        SELECT device_id, timestamp, operating_hours, health_score, 
                               anomaly_score, status, sensor_data
                        FROM sensor_data 
                        WHERE device_id = ?
                        ORDER BY timestamp DESC
                    '''
                    
                    if count:
                        query += f' LIMIT {count}'
                    
                    df = pd.read_sql_query(query, conn, params=(device_id,))
                    
                    if df.empty:
                        return df
                    
                    # 센서 데이터 확장
                    sensor_columns = []
                    for idx, row in df.iterrows():
                        try:
                            sensor_data = json.loads(row['sensor_data'])
                            for sensor_name, value in sensor_data.items():
                                df.at[idx, sensor_name] = value
                                if sensor_name not in sensor_columns:
                                    sensor_columns.append(sensor_name)
                        except:
                            pass
                    
                    # 불필요한 컬럼 제거
                    df = df.drop('sensor_data', axis=1)
                    
                    # timestamp 변환
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    return df
        
        except Exception as e:
            logger.error(f"SQLite 조회 오류: {e}")
            return pd.DataFrame()
    
    def get_all_devices(self) -> List[str]:
        """모든 디바이스 ID 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT device_id FROM sensor_data')
                devices = [row[0] for row in cursor.fetchall()]
                return devices
        except Exception as e:
            logger.error(f"SQLite 디바이스 목록 조회 오류: {e}")
            return []
    
    def delete_device_data(self, device_id: str) -> bool:
        """디바이스 데이터 삭제"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM sensor_data WHERE device_id = ?', (device_id,))
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"SQLite 삭제 오류: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """데이터베이스 통계"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 총 레코드 수
                cursor.execute('SELECT COUNT(*) FROM sensor_data')
                total_records = cursor.fetchone()[0]
                
                # 디바이스 수
                cursor.execute('SELECT COUNT(DISTINCT device_id) FROM sensor_data')
                device_count = cursor.fetchone()[0]
                
                # 최신/최오래된 데이터
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM sensor_data')
                min_time, max_time = cursor.fetchone()
                
                return {
                    'total_records': total_records,
                    'device_count': device_count,
                    'oldest_data': min_time,
                    'newest_data': max_time,
                    'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }
        
        except Exception as e:
            logger.error(f"통계 조회 오류: {e}")
            return {}


class DataManager:
    """데이터 저장소 관리자"""
    
    def __init__(self, storage_type: str = "memory"):
        """
        Args:
            storage_type: 저장소 타입 ("memory", "csv", "sqlite")
        """
        self.storage_type = storage_type
        self.storage = self._create_storage(storage_type)
        logger.info(f"데이터 관리자 초기화 (타입: {storage_type})")
    
    def _create_storage(self, storage_type: str) -> DataStorage:
        """저장소 인스턴스 생성"""
        if storage_type == "memory":
            return MemoryStorage()
        elif storage_type == "csv":
            return CSVStorage()
        elif storage_type == "sqlite":
            return SQLiteStorage()
        else:
            logger.warning(f"알 수 없는 저장소 타입: {storage_type}, 메모리 저장소 사용")
            return MemoryStorage()
    
    def save_data(self, device_id: str, data: Dict) -> bool:
        """데이터 저장"""
        return self.storage.save_device_data(device_id, data)
    
    def get_data(self, device_id: str, count: int = None) -> pd.DataFrame:
        """데이터 조회"""
        return self.storage.get_device_data(device_id, count)
    
    def get_devices(self) -> List[str]:
        """디바이스 목록 조회"""
        return self.storage.get_all_devices()
    
    def delete_data(self, device_id: str) -> bool:
        """데이터 삭제"""
        return self.storage.delete_device_data(device_id)
    
    def get_all_data(self) -> pd.DataFrame:
        """모든 데이터 조회"""
        all_data = []
        
        for device_id in self.get_devices():
            device_data = self.get_data(device_id)
            if not device_data.empty:
                all_data.append(device_data)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_summary(self) -> Dict:
        """저장소 요약 정보"""
        devices = self.get_devices()
        summary = {
            'storage_type': self.storage_type,
            'device_count': len(devices),
            'total_records': 0
        }
        
        for device_id in devices:
            device_data = self.get_data(device_id)
            summary['total_records'] += len(device_data)
        
        # 저장소별 추가 정보
        if hasattr(self.storage, 'get_memory_usage'):
            summary.update(self.storage.get_memory_usage())
        elif hasattr(self.storage, 'get_statistics'):
            summary.update(self.storage.get_statistics())
        
        return summary
    
    def backup_data(self, backup_path: str = None) -> str:
        """데이터 백업"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(config.DATA_DIR, f"backup_{timestamp}.csv")
            
            all_data = self.get_all_data()
            if not all_data.empty:
                all_data.to_csv(backup_path, index=False)
                logger.info(f"데이터 백업 완료: {backup_path}")
                return backup_path
            else:
                logger.warning("백업할 데이터가 없습니다")
                return ""
        
        except Exception as e:
            logger.error(f"데이터 백업 오류: {e}")
            return ""


if __name__ == "__main__":
    # 데이터 저장소 테스트
    print("데이터 저장소 테스트 시작...")
    
    # 테스트 데이터
    test_data = {
        'device_id': 'TEST_DEVICE_001',
        'timestamp': datetime.now().isoformat(),
        'operating_hours': 100.5,
        'health_score': 85.3,
        'anomaly_score': 0.15,
        'status': 'normal',
        'sensors': {
            'temperature': 72.5,
            'vibration_x': 0.8,
            'current': 16.2,
            'pressure': 2.7
        }
    }
    
    # 각 저장소 타입 테스트
    for storage_type in ['memory', 'csv', 'sqlite']:
        print(f"\n{storage_type.upper()} 저장소 테스트:")
        
        manager = DataManager(storage_type)
        
        # 데이터 저장
        success = manager.save_data(test_data['device_id'], test_data)
        print(f"저장 성공: {success}")
        
        # 데이터 조회
        retrieved_data = manager.get_data(test_data['device_id'])
        print(f"조회된 레코드 수: {len(retrieved_data)}")
        
        # 요약 정보
        summary = manager.get_summary()
        print(f"요약: {summary}")
    
    print("\n데이터 저장소 테스트 완료!")
