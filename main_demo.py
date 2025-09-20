"""
IoT 예측 유지보수 시스템 통합 데모
모든 구성요소를 순차적으로 실행하여 시스템을 시연합니다.
"""

import os
import sys
import time
import threading
import subprocess
import signal
import json
import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 안전한 import 처리
try:
    from data.data_generator import generate_sample_dataset, IoTSensorDataGenerator
    DATA_GENERATOR_AVAILABLE = True
except ImportError:
    print("Warning: data_generator.py not found")
    DATA_GENERATOR_AVAILABLE = False

try:
    from models.predictive_model import IoTPredictiveMaintenanceModel
    MODEL_AVAILABLE = True
except ImportError:
    print("Warning: predictive_model.py not found")
    MODEL_AVAILABLE = False

try:
    from streaming.kafka_streaming import StreamingManager
    STREAMING_AVAILABLE = True
except ImportError:
    print("Warning: kafka_streaming.py not found")
    STREAMING_AVAILABLE = False

try:
    from core.data_storage import DataManager
    STORAGE_AVAILABLE = True
except ImportError:
    print("Warning: data_storage.py not found")
    STORAGE_AVAILABLE = False

try:
    from alerts.alert_system import AlertManager
    ALERT_AVAILABLE = True
except ImportError:
    print("Warning: alert_system.py not found")
    ALERT_AVAILABLE = False

try:
    from core.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    print("Warning: config.py not found")
    CONFIG_AVAILABLE = False


class IoTSystemDemo:
    """IoT 예측 유지보수 시스템 데모 클래스"""

    def __init__(self):
        self.processes = []
        self.stop_demo = False

    def print_header(self, title):
        """섹션 헤더 출력"""
        print("\n" + "="*70)
        print(f" 🏭 {title}")
        print("="*70)

    def print_step(self, step_num, description):
        """단계별 설명 출력"""
        print(f"\n[단계 {step_num}] {description}")
        print("-" * 60)

    def check_dependencies(self):
        """필요한 모듈 체크"""
        print("📋 시스템 의존성 체크:")

        dependencies = [
            ("데이터 생성기", DATA_GENERATOR_AVAILABLE),
            ("AI 예측 모델", MODEL_AVAILABLE),
            ("스트리밍 처리", STREAMING_AVAILABLE),
            ("데이터 저장소", STORAGE_AVAILABLE),
            ("알림 시스템", ALERT_AVAILABLE),
            ("설정 관리", CONFIG_AVAILABLE)
        ]

        available_count = 0
        for name, available in dependencies:
            status = "✅ 사용 가능" if available else "❌ 사용 불가"
            print(f"   {name}: {status}")
            if available:
                available_count += 1

        print(f"\n사용 가능한 모듈: {available_count}/{len(dependencies)}")

        if available_count < len(dependencies):
            print("\n⚠️  일부 모듈이 없습니다. 해당 데모는 건너뛸 수 있습니다.")

        return available_count > 0

    def demo_data_generation(self):
        """1. 데이터 생성 데모"""
        self.print_header("1. IoT 센서 데이터 생성 데모")

        if not DATA_GENERATOR_AVAILABLE:
            print("❌ data_generator.py 파일이 필요합니다.")
            print("📝 생성 방법: 제공된 data_generator.py 파일을 프로젝트 폴더에 복사하세요.")
            return

        print("🔧 IoT 센서 데이터 생성기를 이용해 제조업 장비의 센서 데이터를 시뮬레이션합니다.")

        try:
            # 단일 디바이스 실시간 데이터 생성
            print("\n📡 실시간 센서 데이터 생성 중...")
            generator = IoTSensorDataGenerator("DEMO_DEVICE", failure_probability=0.05)

            for i in range(5):
                data = generator.generate_sensor_data()
                print(f"   시점 {i+1}: 건강도 {data['health_score']:.1f}%, "
                      f"온도 {data['sensors']['temperature']:.1f}°C, "
                      f"상태: {data['status']}")
                time.sleep(0.5)

            # 과거 데이터 생성
            print("\n📊 과거 데이터 생성 중...")
            sample_data, devices = generate_sample_dataset()

            print(f"✅ 생성 완료: {len(sample_data):,}개 레코드, {len(devices)}개 디바이스")

            # 데이터 저장
            filename = 'demo_iot_data.csv'
            sample_data.to_csv(filename, index=False)
            print(f"💾 데이터 저장: {filename}")

            # 간단한 통계
            print(f"\n📈 데이터 요약:")
            print(f"   평균 건강도: {sample_data['health_score'].mean():.1f}%")
            print(f"   평균 이상점수: {sample_data['anomaly_score'].mean():.3f}")
            print(f"   상태 분포: {dict(sample_data['status'].value_counts())}")

            # 시각화 생성
            self._create_data_visualization(sample_data)

        except Exception as e:
            print(f"❌ 데이터 생성 오류: {e}")

        input("\n⏸️  계속하려면 Enter를 누르세요...")

    def _create_data_visualization(self, sample_data):
        """데이터 시각화 생성"""
        try:
            print("\n🎨 데이터 시각화 생성 중...")

            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('IoT 센서 데이터 분석 대시보드', fontsize=16, fontweight='bold')

            # 1. 디바이스별 건강도 추이
            ax1 = axes[0, 0]
            for device_id in sample_data['device_id'].unique():
                device_data = sample_data[sample_data['device_id'] == device_id]
                ax1.plot(device_data.index, device_data['health_score'],
                        label=device_id, alpha=0.8, linewidth=2)
            ax1.set_title('디바이스별 건강도 추이', fontweight='bold')
            ax1.set_xlabel('시간 인덱스')
            ax1.set_ylabel('건강도 (%)')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 2. 센서 분포 (온도)
            ax2 = axes[0, 1]
            ax2.hist(sample_data['temperature'], bins=30, alpha=0.7,
                    color='orange', edgecolor='black')
            ax2.set_title('온도 센서 분포', fontweight='bold')
            ax2.set_xlabel('온도 (°C)')
            ax2.set_ylabel('빈도')
            ax2.grid(True, alpha=0.3)

            # 3. 건강도 vs 이상점수 산점도
            ax3 = axes[0, 2]
            scatter = ax3.scatter(sample_data['health_score'], sample_data['anomaly_score'],
                                 c=sample_data['health_score'], cmap='RdYlGn', alpha=0.6)
            ax3.set_title('건강도 vs 이상점수 관계', fontweight='bold')
            ax3.set_xlabel('건강도 (%)')
            ax3.set_ylabel('이상점수')
            plt.colorbar(scatter, ax=ax3, label='건강도')
            ax3.grid(True, alpha=0.3)

            # 4. 상태별 파이차트
            ax4 = axes[1, 0]
            status_counts = sample_data['status'].value_counts()
            colors = {'normal': '#4CAF50', 'warning': '#FF9800', 'critical': '#F44336'}
            pie_colors = [colors.get(status, '#808080') for status in status_counts.index]
            ax4.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                   colors=pie_colors, startangle=90)
            ax4.set_title('장비 상태 분포', fontweight='bold')

            # 5. 센서 상관관계 히트맵
            ax5 = axes[1, 1]
            sensor_cols = ['temperature', 'vibration_x', 'vibration_y', 'pressure', 'current']
            available_cols = [col for col in sensor_cols if col in sample_data.columns]

            if len(available_cols) > 1:
                corr_matrix = sample_data[available_cols].corr()
                im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax5.set_xticks(range(len(available_cols)))
                ax5.set_yticks(range(len(available_cols)))
                ax5.set_xticklabels(available_cols, rotation=45)
                ax5.set_yticklabels(available_cols)
                ax5.set_title('센서 상관관계', fontweight='bold')

                # 상관계수 텍스트 추가
                for i in range(len(available_cols)):
                    for j in range(len(available_cols)):
                        text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)

                plt.colorbar(im, ax=ax5)
            else:
                ax5.text(0.5, 0.5, 'insufficient\nsensor data',
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('센서 상관관계', fontweight='bold')

            # 6. 운영시간별 성능
            ax6 = axes[1, 2]
            scatter2 = ax6.scatter(sample_data['operating_hours'], sample_data['health_score'],
                                  c=sample_data['anomaly_score'], cmap='viridis', alpha=0.6)
            ax6.set_title('운영시간 vs 건강도', fontweight='bold')
            ax6.set_xlabel('운영시간 (h)')
            ax6.set_ylabel('건강도 (%)')
            plt.colorbar(scatter2, ax=ax6, label='이상점수')
            ax6.grid(True, alpha=0.3)

            plt.tight_layout()

            # 파일 저장
            filename = 'demo_data_analysis.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"💾 시각화 저장: {filename}")

            # 화면 표시 (가능한 경우)
            try:
                plt.show()
            except:
                print("   (GUI 환경이 아니어서 화면 표시는 생략됩니다)")

            plt.close()

        except Exception as e:
            print(f"⚠️  시각화 생성 중 오류 (계속 진행): {e}")

    def demo_model_training(self):
        """2. 모델 훈련 데모"""
        self.print_header("2. AI 예측 모델 훈련 데모")

        if not MODEL_AVAILABLE:
            print("❌ predictive_model.py 파일이 필요합니다.")
            print("📝 생성 방법: 제공된 predictive_model.py 파일을 프로젝트 폴더에 복사하세요.")
            return

        print("🤖 TensorFlow 2.0을 사용하여 LSTM 기반 예측 모델을 훈련합니다.")

        # 훈련 데이터 로드
        try:
            if os.path.exists('demo_iot_data.csv'):
                data = pd.read_csv('demo_iot_data.csv')
                print(f"✅ 데이터 로드 완료: {len(data):,}개 레코드")
            else:
                print("❌ demo_iot_data.csv 파일을 찾을 수 없습니다.")
                print("   먼저 '데이터 생성 데모'를 실행하세요.")
                return
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return

        # 모델 초기화 및 훈련
        try:
            print("\n🏋️  모델 훈련 시작...")
            model = IoTPredictiveMaintenanceModel(sequence_length=30, prediction_horizon=5)

            print("   - 특성 엔지니어링 수행 중...")
            print("   - LSTM 모델 구성 중...")
            print("   - 훈련 데이터 준비 중...")

            # 훈련 실행 (에포크 수를 줄여서 데모용으로)
            history = model.train(data, epochs=15, batch_size=16)
            print("✅ 모델 훈련 완료!")

            # 모델 저장
            model_name = "demo_model"
            model.save_model(model_name)
            print(f"💾 모델 저장 완료: {model_name}")

            # 샘플 예측 수행
            self._demo_prediction(model, data)

        except Exception as e:
            print(f"❌ 모델 훈련 실패: {e}")
            print("   💡 TensorFlow 설치를 확인하거나 데이터 크기를 줄여보세요.")

        input("\n⏸️  계속하려면 Enter를 누르세요...")

    def _demo_prediction(self, model, data):
        """예측 데모"""
        try:
            print("\n🔮 샘플 예측 수행...")

            sample_devices = data['device_id'].unique()[:3]

            for i, device_id in enumerate(sample_devices, 1):
                device_data = data[data['device_id'] == device_id].tail(100)

                if len(device_data) < 30:  # 최소 시퀀스 길이 확인
                    print(f"   ⚠️  {device_id}: 데이터 부족")
                    continue

                try:
                    prediction = model.predict(device_data, device_id)

                    status_emoji = {
                        'low': '🟢',
                        'medium': '🟡',
                        'high': '🔴'
                    }

                    emoji = status_emoji.get(prediction['risk_level'], '⚪')

                    print(f"   {emoji} 디바이스 {i}: {prediction['device_id']}")
                    print(f"      유지보수 확률: {prediction['maintenance_probability']:.1%}")
                    print(f"      유지보수 필요: {'✅ 예' if prediction['maintenance_needed'] else '❌ 아니오'}")
                    print(f"      위험 수준: {prediction['risk_level']}")

                except Exception as e:
                    print(f"   ❌ {device_id} 예측 실패: {e}")

        except Exception as e:
            print(f"⚠️  예측 데모 중 오류: {e}")

    def demo_data_storage(self):
        """3. 데이터 저장소 데모"""
        self.print_header("3. 데이터 저장소 관리 데모")

        if not STORAGE_AVAILABLE:
            print("❌ data_storage.py 파일이 필요합니다.")
            print("📝 생성 방법: 제공된 data_storage.py 파일을 프로젝트 폴더에 복사하세요.")
            return

        print("💾 다양한 데이터 저장소 옵션을 테스트합니다.")

        # 테스트 데이터 준비
        test_data = {
            'device_id': 'STORAGE_TEST_001',
            'timestamp': datetime.now().isoformat(),
            'operating_hours': 150.5,
            'health_score': 78.3,
            'anomaly_score': 0.25,
            'status': 'normal',
            'sensors': {
                'temperature': 68.5,
                'vibration_x': 0.9,
                'current': 17.2,
                'pressure': 2.8
            }
        }

        # 각 저장소 타입 테스트
        storage_types = ['memory', 'csv', 'sqlite']

        for storage_type in storage_types:
            print(f"\n📦 {storage_type.upper()} 저장소 테스트:")

            try:
                manager = DataManager(storage_type)

                # 데이터 저장
                success = manager.save_data(test_data['device_id'], test_data)
                print(f"   저장: {'✅ 성공' if success else '❌ 실패'}")

                # 데이터 조회
                retrieved_data = manager.get_data(test_data['device_id'])
                print(f"   조회: {len(retrieved_data)}개 레코드")

                # 요약 정보
                summary = manager.get_summary()
                print(f"   요약: {summary}")

                # 추가 데이터 저장 (성능 테스트)
                print("   성능 테스트 중...", end="")
                for i in range(10):
                    test_data['timestamp'] = datetime.now().isoformat()
                    test_data['health_score'] = 70 + i * 2
                    manager.save_data(test_data['device_id'], test_data)
                    print(".", end="", flush=True)
                print(" 완료")

                final_data = manager.get_data(test_data['device_id'])
                print(f"   최종: {len(final_data)}개 레코드")

            except Exception as e:
                print(f"   ❌ 오류: {e}")

        input("\n⏸️  계속하려면 Enter를 누르세요...")

    def demo_alert_system(self):
        """4. 알림 시스템 데모"""
        self.print_header("4. 알림 시스템 데모")

        if not ALERT_AVAILABLE:
            print("❌ alert_system.py 파일이 필요합니다.")
            print("📝 생성 방법: 제공된 alert_system.py 파일을 프로젝트 폴더에 복사하세요.")
            return

        print("🚨 실시간 알림 시스템을 테스트합니다.")

        try:
            # 알림 관리자 초기화
            alert_manager = AlertManager()

            # 알림 콜백 설정
            alerts_received = []

            def alert_callback(alert):
                alerts_received.append(alert)
                priority_emoji = {
                    'low': '🟢',
                    'medium': '🟡',
                    'high': '🟠',
                    'critical': '🔴'
                }
                emoji = priority_emoji.get(alert.priority.value, '⚪')
                print(f"   {emoji} [{alert.priority.value.upper()}] {alert.message}")

            alert_manager.add_callback(alert_callback)

            # 테스트 시나리오
            test_scenarios = [
                {
                    'name': '정상 상태',
                    'data': {
                        'device_id': 'ALERT_TEST_001',
                        'health_score': 85,
                        'anomaly_score': 0.2,
                        'sensors': {'temperature': 70, 'current': 15}
                    }
                },
                {
                    'name': '건강도 저하 경고',
                    'data': {
                        'device_id': 'ALERT_TEST_002',
                        'health_score': 60,
                        'anomaly_score': 0.4,
                        'sensors': {'temperature': 85, 'current': 20}
                    }
                },
                {
                    'name': '이상 점수 높음',
                    'data': {
                        'device_id': 'ALERT_TEST_003',
                        'health_score': 70,
                        'anomaly_score': 0.8,
                        'sensors': {'temperature': 95, 'current': 25}
                    }
                },
                {
                    'name': '위험 수준 (Critical)',
                    'data': {
                        'device_id': 'ALERT_TEST_004',
                        'health_score': 25,
                        'anomaly_score': 0.9,
                        'sensors': {'temperature': 110, 'current': 30}
                    }
                },
                {
                    'name': '센서 오류',
                    'data': {
                        'device_id': 'ALERT_TEST_005',
                        'health_score': 80,
                        'anomaly_score': 0.3,
                        'sensors': {'temperature': 200, 'current': 15}  # 비정상적인 온도
                    }
                }
            ]

            print("\n🧪 알림 테스트 시나리오 실행:")

            for i, scenario in enumerate(test_scenarios, 1):
                print(f"\n시나리오 {i}: {scenario['name']}")
                alert_manager.process_data(scenario['data']['device_id'], scenario['data'])
                time.sleep(0.5)

            # 알림 통계
            print(f"\n📊 알림 시스템 통계:")
            stats = alert_manager.get_alert_statistics()
            for key, value in stats.items():
                print(f"   {key}: {value}")

            print(f"\n📨 총 수신된 알림: {len(alerts_received)}개")

            if alerts_received:
                print("📋 알림 상세:")
                for i, alert in enumerate(alerts_received, 1):
                    print(f"   {i}. [{alert.priority.value}] {alert.device_id}: {alert.alert_type.value}")

        except Exception as e:
            print(f"❌ 알림 시스템 오류: {e}")

        input("\n⏸️  계속하려면 Enter를 누르세요...")

    def demo_streaming(self):
        """5. 실시간 스트리밍 데모"""
        self.print_header("5. 실시간 데이터 스트리밍 데모")

        if not STREAMING_AVAILABLE:
            print("❌ kafka_streaming.py 파일이 필요합니다.")
            print("📝 생성 방법: 제공된 kafka_streaming.py 파일을 프로젝트 폴더에 복사하세요.")
            return

        if not DATA_GENERATOR_AVAILABLE:
            print("❌ data_generator.py 파일도 필요합니다.")
            return

        print("📡 메모리 기반 실시간 데이터 스트리밍을 시연합니다.")

        try:
            # 스트리밍 매니저 초기화
            streaming_manager = StreamingManager()

            # 시스템 설정
            device_configs = [
                {'device_id': 'STREAM_DEVICE_001', 'failure_probability': 0.03},
                {'device_id': 'STREAM_DEVICE_002', 'failure_probability': 0.02},
                {'device_id': 'STREAM_DEVICE_003', 'failure_probability': 0.05}
            ]

            print(f"🔧 시스템 설정 중... ({len(device_configs)}개 디바이스)")
            streaming_manager.setup_system(device_configs)

            # 데이터 처리 콜백
            received_count = 0

            def data_processor(data):
                nonlocal received_count
                received_count += 1
                if received_count % 4 == 0:  # 4개마다 출력
                    print(f"📡 수신 #{received_count}: {data['device_id']} - "
                          f"건강도: {data['health_score']:.1f}%, "
                          f"상태: {data['status']}")

            streaming_manager.add_data_callback(data_processor)

            # 알림 콜백
            alert_count = 0

            def alert_handler(alert):
                nonlocal alert_count
                alert_count += 1
                print(f"🚨 스트리밍 알림 #{alert_count}: [{alert['priority'].upper()}] {alert['message']}")

            streaming_manager.add_alert_callback(alert_handler)

            print("\n🚀 스트리밍 시작... (15초간 실행)")
            print("-" * 60)

            if streaming_manager.start_streaming(interval_seconds=2):
                # 15초간 실행
                for i in range(15):
                    time.sleep(1)
                    if i % 5 == 4:  # 5초마다 상태 출력
                        status = streaming_manager.get_system_status()
                        print(f"📊 [{i+1}초] 큐: {status['producer_queue_size']}, "
                              f"메시지: {status['message_count']}, "
                              f"예측: {len(status['predictions'])}")

                print("-" * 60)
                print(f"✅ 스트리밍 완료!")
                print(f"   📨 총 처리 메시지: {received_count}개")
                print(f"   🚨 총 알림 발생: {alert_count}개")

                # 최종 상태
                final_status = streaming_manager.get_system_status()
                print(f"   📊 최종 큐 크기: {final_status['producer_queue_size']}")
                print(f"   🔧 디바이스 수: {final_status['device_count']}")

            else:
                print("❌ 스트리밍 시작 실패")

        except Exception as e:
            print(f"❌ 스트리밍 데모 실패: {e}")
        finally:
            if 'streaming_manager' in locals():
                streaming_manager.stop_streaming()

        input("\n⏸️  계속하려면 Enter를 누르세요...")

    def demo_api_server(self):
        """6. API 서버 데모"""
        self.print_header("6. REST API 서버 데모")

        print("🌐 Flask 기반 REST API 서버를 시작합니다.")
        print("📋 API 문서: http://localhost:5000/api/health")

        # API 서버 시작 (별도 프로세스)
        try:
            print("\n🚀 API 서버 시작 중...")
            api_process = subprocess.Popen([
                sys.executable, 'api_server.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes.append(api_process)

            # API 서버 시작 대기
            print("⏳ API 서버 초기화 대기...")
            time.sleep(8)

            # API 테스트
            self._test_api_endpoints()

        except FileNotFoundError:
            print("❌ api_server.py 파일을 찾을 수 없습니다.")
            print("📝 생성 방법: 제공된 api_server.py 파일을 프로젝트 폴더에 복사하세요.")
        except Exception as e:
            print(f"❌ API 서버 시작 실패: {e}")

        input("\n⏸️  계속하려면 Enter를 누르세요...")

    def _test_api_endpoints(self):
        """API 엔드포인트 테스트"""
        base_url = "http://localhost:5000"

        print("🧪 API 엔드포인트 테스트:")

        try:
            # 헬스체크
            print("   1. 헬스체크...", end="")
            response = requests.get(f'{base_url}/api/health', timeout=10)
            if response.status_code == 200:
                print(" ✅")
                health_data = response.json()
                print(f"      상태: {health_data['status']}")
                print(f"      버전: {health_data['version']}")
                print(f"      서비스: {health_data['services']}")
            else:
                print(f" ❌ (상태코드: {response.status_code})")
                return

            # 로그인 테스트
            print("   2. 사용자 인증...", end="")
            login_response = requests.post(f'{base_url}/api/auth/login',
                                         json={'username': 'admin', 'password': 'password123'},
                                         timeout=10)

            if login_response.status_code == 200:
                print(" ✅")
                token = login_response.json()['token']
                headers = {'Authorization': f'Bearer {token}'}

                # 디바이스 목록 조회
                print("   3. 디바이스 목록...", end="")
                devices_response = requests.get(f'{base_url}/api/devices',
                                              headers=headers, timeout=10)
                if devices_response.status_code == 200:
                    print(" ✅")
                    devices = devices_response.json()
                    print(f"      등록된 디바이스: {devices['total_count']}개")
                else:
                    print(f" ❌ (상태코드: {devices_response.status_code})")

                # 시스템 통계 조회
                print("   4. 시스템 통계...", end="")
                stats_response = requests.get(f'{base_url}/api/stats/summary',
                                            headers=headers, timeout=10)
                if stats_response.status_code == 200:
                    print(" ✅")
                    stats = stats_response.json()
                    print(f"      평균 건강도: {stats['average_health']:.1f}%")
                    print(f"      상태별 분포: {stats['devices_by_status']}")
                else:
                    print(f" ❌ (상태코드: {stats_response.status_code})")

                # 실시간 데이터 조회
                print("   5. 실시간 데이터...", end="")
                device_data_response = requests.get(f'{base_url}/api/devices/DEVICE_001/data',
                                                  headers=headers, timeout=10)
                if device_data_response.status_code == 200:
                    print(" ✅")
                    device_data = device_data_response.json()
                    print(f"      디바이스: {device_data['device_id']}")
                    print(f"      건강도: {device_data['health_score']:.1f}%")
                    print(f"      상태: {device_data['status']}")
                else:
                    print(f" ❌ (상태코드: {device_data_response.status_code})")

                # 예측 테스트
                print("   6. 고장 예측...", end="")
                prediction_response = requests.post(f'{base_url}/api/predict/DEVICE_001',
                                                  headers=headers, timeout=15)
                if prediction_response.status_code == 200:
                    print(" ✅")
                    prediction = prediction_response.json()
                    print(f"      유지보수 확률: {prediction['maintenance_probability']:.1%}")
                    print(f"      위험 수준: {prediction['risk_level']}")
                    print(f"      예측 방법: {prediction.get('method', 'unknown')}")
                else:
                    print(f" ❌ (상태코드: {prediction_response.status_code})")

            else:
                print(f" ❌ (상태코드: {login_response.status_code})")

        except requests.exceptions.ConnectionError:
            print("\n❌ API 서버에 연결할 수 없습니다.")
            print("   💡 api_server.py가 실행 중인지 확인하세요.")
        except requests.exceptions.Timeout:
            print("\n⏰ API 요청 시간 초과")
        except Exception as e:
            print(f"\n❌ API 테스트 오류: {e}")

    def demo_dashboard(self):
        """7. 대시보드 데모"""
        self.print_header("7. 실시간 모니터링 대시보드 데모")

        print("📊 Streamlit 기반 실시간 모니터링 대시보드를 시작합니다.")
        print("🌐 대시보드 URL: http://localhost:8501")

        try:
            # Streamlit 설치 확인
            import streamlit
            print("✅ Streamlit 설치 확인됨")

            # Streamlit 대시보드 시작
            print("\n🚀 대시보드 시작 중...")
            dashboard_process = subprocess.Popen([
                'streamlit', 'run', 'dashboard.py',
                '--server.port=8501',
                '--server.address=0.0.0.0',
                '--server.headless=true'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes.append(dashboard_process)

            print("⏳ 대시보드 초기화 대기...")
            time.sleep(10)

            print("✅ 대시보드가 시작되었습니다!")
            print("🌐 브라우저에서 http://localhost:8501 을 열어주세요.")
            print("\n📋 대시보드 주요 기능:")
            print("   📊 실시간 모니터링 - 디바이스 상태 실시간 표시")
            print("   📈 데이터 분석 - 과거 데이터 분석 및 시각화")
            print("   🤖 모델 훈련 - AI 모델 훈련 및 평가")
            print("   ⚙️  시스템 설정 - 알림 임계값 및 시스템 구성")

            print("\n💡 사용 팁:")
            print("   - 사이드바에서 페이지를 전환할 수 있습니다")
            print("   - '자동 새로고침'을 체크하면 실시간 업데이트됩니다")
            print("   - 각 차트는 상호작용이 가능합니다")

        except ImportError:
            print("❌ Streamlit이 설치되지 않았습니다.")
            print("📦 설치 명령어: pip install streamlit")
        except FileNotFoundError:
            print("❌ dashboard.py 파일을 찾을 수 없습니다.")
            print("📝 생성 방법: 제공된 dashboard.py 파일을 프로젝트 폴더에 복사하세요.")
        except Exception as e:
            print(f"❌ 대시보드 시작 실패: {e}")
            print("💡 수동 실행: streamlit run dashboard.py")

        input("\n⏸️  계속하려면 Enter를 누르세요...")

    def cleanup(self):
        """리소스 정리"""
        print("\n🧹 시스템 정리 중...")

        # 프로세스 종료
        terminated_count = 0
        for process in self.processes:
            try:
                if process.poll() is None:  # 프로세스가 실행 중인 경우
                    process.terminate()
                    process.wait(timeout=5)
                    terminated_count += 1
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    terminated_count += 1
                    print("⚠️  프로세스 강제 종료됨")
                except:
                    pass
            except Exception as e:
                print(f"⚠️  프로세스 종료 중 오류: {e}")

        if terminated_count > 0:
            print(f"✅ {terminated_count}개 프로세스 종료됨")

        print("✅ 정리 완료!")

    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        print("\n\n🛑 데모 중단 신호 수신...")
        self.stop_demo = True
        self.cleanup()
        sys.exit(0)

    def run_full_demo(self):
        """전체 데모 실행"""
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            self.print_header("IoT 예측 유지보수 시스템 통합 데모")
            print("🎯 이 데모는 전체 시스템의 기능을 순차적으로 시연합니다.")
            print("📝 각 단계에서 Enter를 눌러 다음 단계로 진행할 수 있습니다.")
            print("🚫 Ctrl+C를 눌러 언제든지 중단할 수 있습니다.")

            # 의존성 체크
            if not self.check_dependencies():
                print("\n❌ 필요한 모듈이 없어서 데모를 실행할 수 없습니다.")
                return

            input("\n🚀 시작하려면 Enter를 누르세요...")

            # 데모 단계별 실행
            demo_steps = [
                ("데이터 생성", self.demo_data_generation),
                ("AI 모델 훈련", self.demo_model_training),
                ("데이터 저장소", self.demo_data_storage),
                ("알림 시스템", self.demo_alert_system),
                ("실시간 스트리밍", self.demo_streaming),
                ("API 서버", self.demo_api_server),
                ("웹 대시보드", self.demo_dashboard)
            ]

            for i, (name, demo_func) in enumerate(demo_steps, 1):
                if self.stop_demo:
                    break

                print(f"\n{'='*20} 진행률: {i}/{len(demo_steps)} {'='*20}")
                try:
                    demo_func()
                except KeyboardInterrupt:
                    print(f"\n⏸️  {name} 데모가 중단되었습니다.")
                    break
                except Exception as e:
                    print(f"\n❌ {name} 데모 중 오류 발생: {e}")
                    print("   계속 진행합니다...")
                    input("\n⏸️  계속하려면 Enter를 누르세요...")

            if not self.stop_demo:
                self.print_header("🎉 데모 완료!")
                print("✅ IoT 예측 유지보수 시스템 데모가 성공적으로 완료되었습니다!")

                print("\n📁 생성된 파일들:")
                files_to_check = [
                    ("demo_iot_data.csv", "샘플 IoT 센서 데이터"),
                    ("demo_data_analysis.png", "데이터 분석 시각화"),
                    ("demo_model.h5", "훈련된 AI 예측 모델"),
                    ("demo_model_scaler.pkl", "데이터 전처리 스케일러"),
                    ("demo_model_metadata.json", "모델 메타데이터")
                ]

                for filename, description in files_to_check:
                    if os.path.exists(filename):
                        print(f"   ✅ {filename} - {description}")
                    else:
                        print(f"   ❌ {filename} - {description} (생성되지 않음)")

                print("\n🏗️  시스템 구성요소:")
                components = [
                    ("🔧 데이터 생성기", "10종 센서를 가진 IoT 장비 시뮬레이터"),
                    ("🤖 AI 예측 모델", "TensorFlow 2.0 LSTM 기반 고장 예측"),
                    ("💾 데이터 저장소", "메모리/CSV/SQLite 다중 저장소"),
                    ("🚨 알림 시스템", "다단계 임계값 기반 지능형 알림"),
                    ("📡 스트리밍 처리", "메모리 큐 기반 실시간 데이터 처리"),
                    ("🔌 REST API", "JWT 인증 기반 완전한 웹 API"),
                    ("📊 웹 대시보드", "Streamlit 기반 실시간 모니터링")
                ]

                for icon_name, description in components:
                    print(f"   {icon_name} - {description}")

                print("\n🎓 다음 단계:")
                print("   1. 실제 센서 데이터로 모델 재훈련")
                print("   2. 프로덕션 환경에 맞는 데이터베이스 연동")
                print("   3. 클라우드 배포 및 스케일링")
                print("   4. 모바일 알림 시스템 추가")

                print("\n💡 추가 정보:")
                print("   📖 README.md - 상세한 사용법 및 API 문서")
                print("   🔧 config.py - 시스템 설정 및 튜닝")
                print("   🛠️  requirements.txt - Python 의존성 목록")

        except KeyboardInterrupt:
            print("\n\n⏸️  사용자에 의해 데모가 중단되었습니다.")
        except Exception as e:
            print(f"\n❌ 데모 실행 중 오류 발생: {e}")
        finally:
            self.cleanup()


def print_usage():
    """사용법 출력"""
    print("🏭 IoT 예측 유지보수 시스템 데모")
    print("="*50)
    print("사용법:")
    print("  python main_demo.py           # 전체 데모 실행")
    print("  python main_demo.py [명령어]  # 개별 데모 실행")
    print("\n사용 가능한 명령어:")
    print("  data      - 데이터 생성 데모")
    print("  model     - AI 모델 훈련 데모")
    print("  storage   - 데이터 저장소 데모")
    print("  alert     - 알림 시스템 데모")
    print("  streaming - 실시간 스트리밍 데모")
    print("  api       - REST API 서버 데모")
    print("  dashboard - 웹 대시보드 데모")
    print("  help      - 이 도움말 표시")
    print("\n예시:")
    print("  python main_demo.py data")
    print("  python main_demo.py model")


def main():
    """메인 함수"""
    demo = IoTSystemDemo()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        command_map = {
            "data": demo.demo_data_generation,
            "model": demo.demo_model_training,
            "storage": demo.demo_data_storage,
            "alert": demo.demo_alert_system,
            "streaming": demo.demo_streaming,
            "api": demo.demo_api_server,
            "dashboard": demo.demo_dashboard,
            "help": print_usage
        }

        if command in command_map:
            if command == "help":
                print_usage()
            else:
                print(f"🚀 {command.upper()} 데모 실행")
                try:
                    command_map[command]()
                except KeyboardInterrupt:
                    print(f"\n⏸️  {command} 데모가 중단되었습니다.")
                except Exception as e:
                    print(f"\n❌ {command} 데모 실행 중 오류: {e}")
                finally:
                    demo.cleanup()
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print_usage()
    else:
        # 전체 데모 실행
        demo.run_full_demo()


if __name__ == "__main__":
    main()