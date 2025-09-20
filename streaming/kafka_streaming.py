"""
실시간 데이터 스트리밍 처리
메모리 기반 큐를 사용한 Producer와 Consumer 구현 (Kafka 대신 메모리 큐 사용)
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Callable, Optional
import logging
from queue import Queue, Empty
from collections import deque

# 안전한 import 처리
try:
    from data_generator import IoTSensorDataGenerator
except ImportError:
    print("Warning: data_generator.py not found. Please create it first.")
    IoTSensorDataGenerator = None

try:
    from config import config
except ImportError:
    # 기본 설정
    class DefaultConfig:
        LOG_LEVEL = "INFO"
    config = DefaultConfig()

try:
    from utils import setup_logging
except ImportError:
    # 기본 로깅 설정
    def setup_logging(name):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

# 로깅 설정
logger = setup_logging(__name__)


class IoTDataProducer:
    """IoT 센서 데이터를 메모리 큐로 전송하는 Producer"""

    def __init__(self, queue_size: int = 1000):
        """
        Args:
            queue_size: 큐의 최대 크기
        """
        self.queue = Queue(maxsize=queue_size)
        self.devices = {}
        self.running = False
        self.stream_thread = None
        logger.info(f"Producer 초기화 완료 (큐 크기: {queue_size})")

    def add_device(self, device_id: str, failure_probability: float = 0.02):
        """IoT 디바이스 추가"""
        if IoTSensorDataGenerator is None:
            logger.error("IoTSensorDataGenerator를 사용할 수 없습니다.")
            return False

        self.devices[device_id] = IoTSensorDataGenerator(
            device_id=device_id,
            failure_probability=failure_probability
        )
        logger.info(f"디바이스 추가: {device_id}")
        return True

    def send_data(self, device_id: str, data: Dict) -> bool:
        """개별 데이터 전송"""
        try:
            message = {
                'key': device_id,
                'value': data,
                'timestamp': datetime.now().isoformat()
            }

            # 큐가 가득 찬 경우 1초 대기
            self.queue.put(message, timeout=1)
            logger.debug(f"데이터 전송 성공: {device_id}")
            return True

        except Exception as e:
            logger.error(f"데이터 전송 실패: {e}")
            return False

    def start_streaming(self, interval_seconds: int = 1) -> bool:
        """실시간 데이터 스트리밍 시작"""
        if not self.devices:
            logger.error("등록된 디바이스가 없습니다.")
            return False

        self.running = True
        logger.info(f"실시간 스트리밍 시작 (간격: {interval_seconds}초)")

        def stream_worker():
            while self.running:
                try:
                    for device_id, generator in self.devices.items():
                        # 센서 데이터 생성
                        sensor_data = generator.generate_sensor_data()

                        # 큐로 전송
                        self.send_data(device_id, sensor_data)

                except Exception as e:
                    logger.error(f"스트리밍 오류: {e}")

                time.sleep(interval_seconds)

        # 별도 스레드에서 스트리밍 실행
        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()

        return True

    def stop_streaming(self):
        """스트리밍 중지"""
        self.running = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)

        logger.info("스트리밍 중지됨")

    def get_queue_size(self) -> int:
        """현재 큐 크기 반환"""
        return self.queue.qsize()

    def is_running(self) -> bool:
        """실행 상태 확인"""
        return self.running


class IoTDataConsumer:
    """메모리 큐에서 IoT 데이터를 수신하고 처리하는 Consumer"""

    def __init__(self, producer_queue: Queue = None, buffer_size: int = 100):
        """
        Args:
            producer_queue: Producer의 큐 (나중에 연결 가능)
            buffer_size: 디바이스별 데이터 버퍼 크기
        """
        self.queue = producer_queue
        self.running = False
        self.consume_thread = None
        self.data_buffer = {}  # 디바이스별 데이터 버퍼
        self.buffer_size = buffer_size
        self.callbacks = []
        self.message_count = 0
        logger.info(f"Consumer 초기화 완료 (버퍼 크기: {buffer_size})")

    def connect_to_producer(self, producer: IoTDataProducer):
        """Producer와 연결"""
        self.queue = producer.queue
        logger.info("Producer와 연결됨")

    def add_callback(self, callback: Callable[[Dict], None]):
        """데이터 처리 콜백 함수 추가"""
        self.callbacks.append(callback)
        logger.info(f"콜백 함수 추가됨 (총 {len(self.callbacks)}개)")

    def process_message(self, message: Dict):
        """메시지 처리"""
        try:
            device_id = message['key']
            data = message['value']

            # 디바이스별 버퍼에 데이터 추가
            if device_id not in self.data_buffer:
                self.data_buffer[device_id] = deque(maxlen=self.buffer_size)

            self.data_buffer[device_id].append(data)
            self.message_count += 1

            # 등록된 콜백 함수들 실행
            for callback in self.callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"콜백 실행 오류: {e}")

            logger.debug(f"메시지 처리 완료: {device_id} (총 {self.message_count}개)")

        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")

    def start_consuming(self) -> bool:
        """데이터 수신 시작"""
        if not self.queue:
            logger.error("Producer 큐가 연결되지 않았습니다")
            return False

        self.running = True
        logger.info("데이터 수신 시작")

        def consume_worker():
            while self.running:
                try:
                    # 큐에서 메시지 가져오기 (1초 타임아웃)
                    message = self.queue.get(timeout=1)
                    self.process_message(message)
                    self.queue.task_done()

                except Empty:
                    # 큐가 비어있으면 계속 대기
                    continue
                except Exception as e:
                    logger.error(f"메시지 수신 오류: {e}")
                    time.sleep(1)

        # 별도 스레드에서 수신 실행
        self.consume_thread = threading.Thread(target=consume_worker, daemon=True)
        self.consume_thread.start()

        return True

    def stop_consuming(self):
        """데이터 수신 중지"""
        self.running = False
        if self.consume_thread and self.consume_thread.is_alive():
            self.consume_thread.join(timeout=5)

        logger.info("데이터 수신 중지됨")

    def get_device_data(self, device_id: str, count: int = None) -> List[Dict]:
        """특정 디바이스의 최근 데이터 조회"""
        if device_id not in self.data_buffer:
            return []

        data_list = list(self.data_buffer[device_id])
        if count:
            return data_list[-count:]
        return data_list

    def get_message_count(self) -> int:
        """처리된 메시지 수 반환"""
        return self.message_count

    def get_buffer_status(self) -> Dict:
        """버퍼 상태 정보"""
        return {
            'devices': list(self.data_buffer.keys()),
            'buffer_sizes': {device_id: len(buffer) for device_id, buffer in self.data_buffer.items()},
            'total_messages': self.message_count
        }


class RealTimePredictionService:
    """실시간 예측 서비스"""

    def __init__(self, prediction_threshold: float = 0.7):
        """
        Args:
            prediction_threshold: 예측 임계값
        """
        self.prediction_threshold = prediction_threshold
        self.alert_callbacks = []
        self.device_predictions = {}  # 디바이스별 최근 예측 결과
        logger.info(f"예측 서비스 초기화 완료 (임계값: {prediction_threshold})")

    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """알림 콜백 함수 추가"""
        self.alert_callbacks.append(callback)
        logger.info(f"알림 콜백 추가됨 (총 {len(self.alert_callbacks)}개)")

    def process_real_time_data(self, data: Dict):
        """실시간 데이터 처리 및 예측"""
        try:
            device_id = data.get('device_id')
            if not device_id:
                return

            # 기본 임계값 기반 분석
            anomaly_score = data.get('anomaly_score', 0)
            health_score = data.get('health_score', 100)

            # 예측 결과 저장
            self.device_predictions[device_id] = {
                'health_score': health_score,
                'anomaly_score': anomaly_score,
                'timestamp': data.get('timestamp'),
                'status': data.get('status')
            }

            # 알림 조건 확인
            if anomaly_score > self.prediction_threshold:
                alert = {
                    'device_id': device_id,
                    'alert_type': 'anomaly_threshold',
                    'anomaly_score': anomaly_score,
                    'health_score': health_score,
                    'timestamp': data.get('timestamp'),
                    'message': f'디바이스 {device_id}의 이상 점수가 임계값({self.prediction_threshold})을 초과했습니다.',
                    'priority': 'high' if anomaly_score > 0.9 else 'medium'
                }
                self._send_alert(alert)

            elif health_score < 30:
                alert = {
                    'device_id': device_id,
                    'alert_type': 'health_critical',
                    'health_score': health_score,
                    'timestamp': data.get('timestamp'),
                    'message': f'디바이스 {device_id}의 건강도가 위험 수준({health_score:.1f}%)입니다.',
                    'priority': 'critical'
                }
                self._send_alert(alert)

            logger.debug(f"실시간 예측 처리 완료: {device_id}")

        except Exception as e:
            logger.error(f"실시간 데이터 처리 오류: {e}")

    def _send_alert(self, alert: Dict):
        """알림 전송"""
        logger.warning(f"ALERT: {alert['message']}")

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"알림 콜백 오류: {e}")

    def get_device_prediction(self, device_id: str) -> Optional[Dict]:
        """특정 디바이스의 최근 예측 결과 조회"""
        return self.device_predictions.get(device_id)

    def get_all_predictions(self) -> Dict:
        """모든 디바이스의 예측 결과 조회"""
        return self.device_predictions.copy()


class StreamingManager:
    """스트리밍 시스템 전체 관리"""

    def __init__(self, queue_size: int = 1000, buffer_size: int = 100):
        """
        Args:
            queue_size: Producer 큐 크기
            buffer_size: Consumer 버퍼 크기
        """
        self.producer = IoTDataProducer(queue_size)
        self.consumer = IoTDataConsumer(buffer_size=buffer_size)
        self.prediction_service = RealTimePredictionService()
        self.data_callbacks = []
        self.is_running = False
        logger.info("스트리밍 관리자 초기화 완료")

    def setup_system(self, device_configs: List[Dict] = None):
        """시스템 설정"""
        # 기본 디바이스 설정
        default_devices = device_configs or [
            {'device_id': 'DEVICE_001', 'failure_probability': 0.02},
            {'device_id': 'DEVICE_002', 'failure_probability': 0.03},
            {'device_id': 'DEVICE_003', 'failure_probability': 0.01},
        ]

        # Producer에 디바이스 추가
        for device_config in default_devices:
            success = self.producer.add_device(**device_config)
            if not success:
                logger.error(f"디바이스 추가 실패: {device_config['device_id']}")

        # Consumer와 Producer 연결
        self.consumer.connect_to_producer(self.producer)

        # 데이터 처리 콜백 등록
        def data_processor(data):
            # 사용자 정의 콜백 실행
            for callback in self.data_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"사용자 콜백 오류: {e}")

            # 예측 서비스 처리
            self.prediction_service.process_real_time_data(data)

        self.consumer.add_callback(data_processor)

        logger.info("스트리밍 시스템 설정 완료")

    def add_data_callback(self, callback: Callable[[Dict], None]):
        """데이터 처리 콜백 추가"""
        self.data_callbacks.append(callback)
        logger.info(f"데이터 콜백 추가됨 (총 {len(self.data_callbacks)}개)")

    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """알림 콜백 추가"""
        self.prediction_service.add_alert_callback(callback)

    def start_streaming(self, interval_seconds: int = 5) -> bool:
        """전체 스트리밍 시작"""
        try:
            # Consumer 시작
            if not self.consumer.start_consuming():
                logger.error("Consumer 시작 실패")
                return False

            # Producer 시작
            if not self.producer.start_streaming(interval_seconds):
                logger.error("Producer 시작 실패")
                return False

            self.is_running = True
            logger.info("스트리밍 시스템 시작됨")
            return True

        except Exception as e:
            logger.error(f"스트리밍 시작 실패: {e}")
            return False

    def stop_streaming(self):
        """전체 스트리밍 중지"""
        try:
            self.producer.stop_streaming()
            self.consumer.stop_consuming()
            self.is_running = False
            logger.info("스트리밍 시스템 중지됨")

        except Exception as e:
            logger.error(f"스트리밍 중지 실패: {e}")

    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        return {
            'is_running': self.is_running,
            'producer_queue_size': self.producer.get_queue_size(),
            'device_count': len(self.producer.devices),
            'message_count': self.consumer.get_message_count(),
            'predictions': self.prediction_service.get_all_predictions(),
            'consumer_buffer': self.consumer.get_buffer_status()
        }


def demo_streaming():
    """스트리밍 데모"""
    print("=" * 60)
    print(" 메모리 기반 IoT 데이터 스트리밍 데모")
    print("=" * 60)

    if IoTSensorDataGenerator is None:
        print("❌ data_generator.py 파일이 필요합니다.")
        print("먼저 data_generator.py를 생성하고 실행해주세요.")
        return

    # 스트리밍 매니저 초기화
    manager = StreamingManager(queue_size=500, buffer_size=50)

    # 시스템 설정
    device_configs = [
        {'device_id': 'DEMO_DEVICE_001', 'failure_probability': 0.03},
        {'device_id': 'DEMO_DEVICE_002', 'failure_probability': 0.02},
        {'device_id': 'DEMO_DEVICE_003', 'failure_probability': 0.05}
    ]

    print(f"디바이스 설정 중... ({len(device_configs)}개)")
    manager.setup_system(device_configs)

    # 데이터 처리 콜백 등록
    received_count = 0

    def data_processor(data):
        nonlocal received_count
        received_count += 1
        if received_count % 3 == 0:  # 3개마다 출력
            print(f"📡 수신 #{received_count}: {data['device_id']} - "
                  f"건강도: {data['health_score']:.1f}%, "
                  f"이상점수: {data['anomaly_score']:.3f}, "
                  f"상태: {data['status']}")

    manager.add_data_callback(data_processor)

    # 알림 콜백 등록
    alert_count = 0

    def alert_handler(alert):
        nonlocal alert_count
        alert_count += 1
        priority_emoji = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }
        emoji = priority_emoji.get(alert['priority'], '⚪')
        print(f"🚨 알림 #{alert_count}: {emoji} [{alert['priority'].upper()}] {alert['message']}")

    manager.add_alert_callback(alert_handler)

    try:
        print("\n스트리밍 시작... (20초간 실행)")
        print("-" * 50)

        if manager.start_streaming(interval_seconds=2):
            # 20초간 실행
            for i in range(20):
                time.sleep(1)
                if i % 5 == 4:  # 5초마다 상태 출력
                    status = manager.get_system_status()
                    print(f"📊 [{i+1}초] 큐: {status['producer_queue_size']}, "
                          f"메시지: {status['message_count']}, "
                          f"예측: {len(status['predictions'])}")

        print("-" * 50)
        print(f"✅ 스트리밍 완료!")
        print(f"   총 처리 메시지: {received_count}개")
        print(f"   총 알림 발생: {alert_count}개")

        # 최종 상태
        final_status = manager.get_system_status()
        print(f"   최종 큐 크기: {final_status['producer_queue_size']}")
        print(f"   디바이스 수: {final_status['device_count']}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 스트리밍 데모 실패: {e}")
    finally:
        manager.stop_streaming()
        print("데모 종료")


if __name__ == "__main__":
    demo_streaming()