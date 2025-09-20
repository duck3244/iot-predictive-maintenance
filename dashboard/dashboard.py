"""
Streamlit 기반 실시간 IoT 모니터링 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import redis
from data.data_generator import IoTSensorDataGenerator, generate_sample_dataset
from models.predictive_model import IoTPredictiveMaintenanceModel
import warnings

warnings.filterwarnings('ignore')

# Streamlit 페이지 설정
st.set_page_config(
    page_title="IoT 예측 유지보수 대시보드",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


class DashboardDataManager:
    """대시보드 데이터 관리 클래스"""
    
    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        self.redis_client = None
        self.local_data = defaultdict(lambda: deque(maxlen=1000))
        self.devices = {}
        self.model = None
        
        if use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                self.redis_client.ping()
                st.success("Redis 연결 성공")
            except:
                st.warning("Redis 연결 실패, 로컬 메모리 사용")
                self.use_redis = False
    
    def initialize_devices(self, device_count: int = 5):
        """테스트용 디바이스 초기화"""
        for i in range(device_count):
            device_id = f"DEVICE_{i+1:03d}"
            self.devices[device_id] = IoTSensorDataGenerator(
                device_id=device_id,
                failure_probability=np.random.uniform(0.01, 0.05)
            )
    
    def load_model(self, model_path: str = "iot_predictive_model"):
        """예측 모델 로드"""
        try:
            self.model = IoTPredictiveMaintenanceModel()
            self.model.load_model(model_path)
            return True
        except:
            return False
    
    def generate_real_time_data(self, device_id: str) -> dict:
        """실시간 데이터 생성"""
        if device_id not in self.devices:
            return None
        
        data = self.devices[device_id].generate_sensor_data()
        
        # 데이터 저장
        if self.use_redis and self.redis_client:
            self.redis_client.lpush(f"device:{device_id}", json.dumps(data))
            self.redis_client.ltrim(f"device:{device_id}", 0, 999)  # 최대 1000개 유지
        else:
            self.local_data[device_id].append(data)
        
        return data
    
    def get_device_data(self, device_id: str, count: int = 100) -> pd.DataFrame:
        """디바이스 데이터 조회"""
        if self.use_redis and self.redis_client:
            data_list = self.redis_client.lrange(f"device:{device_id}", 0, count-1)
            data = [json.loads(d) for d in reversed(data_list)]
        else:
            data = list(self.local_data[device_id])[-count:]
        
        if not data:
            return pd.DataFrame()
        
        # 데이터 평탄화
        flat_data = []
        for record in data:
            flat_record = {
                'device_id': record['device_id'],
                'timestamp': record['timestamp'],
                'operating_hours': record['operating_hours'],
                'health_score': record['health_score'],
                'anomaly_score': record['anomaly_score'],
                'status': record['status']
            }
            flat_record.update(record['sensors'])
            flat_data.append(flat_record)
        
        df = pd.DataFrame(flat_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_all_devices_summary(self) -> dict:
        """모든 디바이스 요약 정보"""
        summary = {}
        for device_id in self.devices.keys():
            latest_data = self.get_device_data(device_id, count=1)
            if not latest_data.empty:
                summary[device_id] = {
                    'health_score': latest_data['health_score'].iloc[0],
                    'anomaly_score': latest_data['anomaly_score'].iloc[0],
                    'status': latest_data['status'].iloc[0],
                    'last_update': latest_data['timestamp'].iloc[0]
                }
        return summary


@st.cache_data
def load_sample_data():
    """샘플 데이터 로드"""
    data, _ = generate_sample_dataset()
    return data


def create_device_overview_chart(summary_data):
    """디바이스 개요 차트 생성"""
    if not summary_data:
        return go.Figure()
    
    devices = list(summary_data.keys())
    health_scores = [summary_data[d]['health_score'] for d in devices]
    anomaly_scores = [summary_data[d]['anomaly_score'] for d in devices]
    statuses = [summary_data[d]['status'] for d in devices]
    
    # 상태별 색상
    colors = {
        'normal': '#4CAF50',
        'warning': '#FF9800', 
        'critical': '#F44336'
    }
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['건강도', '이상 점수'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 건강도 바 차트
    fig.add_trace(
        go.Bar(
            x=devices,
            y=health_scores,
            name='건강도',
            marker_color=[colors.get(s, '#808080') for s in statuses],
            text=[f'{h:.1f}%' for h in health_scores],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 이상 점수 바 차트
    fig.add_trace(
        go.Bar(
            x=devices,
            y=anomaly_scores,
            name='이상 점수',
            marker_color='rgba(255, 99, 132, 0.7)',
            text=[f'{a:.3f}' for a in anomaly_scores],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="디바이스 상태 개요"
    )
    
    return fig


def create_time_series_chart(df, metrics):
    """시계열 차트 생성"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=metrics,
        vertical_spacing=0.05
    )
    
    colors = px.colors.qualitative.Plotly
    
    for i, metric in enumerate(metrics, 1):
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=colors[i-1 % len(colors)]),
                    marker=dict(size=4)
                ),
                row=i, col=1
            )
    
    fig.update_layout(
        height=200 * len(metrics),
        showlegend=False
    )
    
    return fig


def create_anomaly_heatmap(df):
    """이상 탐지 히트맵 생성"""
    if df.empty or len(df) < 10:
        return go.Figure()
    
    # 센서 데이터만 선택
    sensor_cols = ['temperature', 'vibration_x', 'vibration_y', 'vibration_z', 
                   'pressure', 'rotation_speed', 'current', 'voltage', 'noise_level']
    
    available_cols = [col for col in sensor_cols if col in df.columns]
    
    if not available_cols:
        return go.Figure()
    
    # 정규화된 데이터
    sensor_data = df[available_cols].tail(50)  # 최근 50개 데이터포인트
    normalized_data = (sensor_data - sensor_data.mean()) / sensor_data.std()
    
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data.T.values,
        x=list(range(len(normalized_data))),
        y=available_cols,
        colorscale='RdYlBu_r',
        zmid=0
    ))
    
    fig.update_layout(
        title="센서 이상 패턴 (정규화된 값)",
        xaxis_title="시간 순서",
        yaxis_title="센서",
        height=400
    )
    
    return fig


def main():
    """메인 대시보드"""
    
    # 제목
    st.title("🏭 IoT 예측 유지보수 대시보드")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("설정")
    
    # 데이터 매니저 초기화
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DashboardDataManager(use_redis=False)
        st.session_state.data_manager.initialize_devices(5)
    
    data_manager = st.session_state.data_manager
    
    # 모델 로드 옵션
    if st.sidebar.button("예측 모델 로드"):
        if data_manager.load_model():
            st.sidebar.success("모델 로드 성공!")
        else:
            st.sidebar.error("모델 로드 실패")
    
    # 실시간 데이터 생성 옵션
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (5초)", value=True)
    if auto_refresh:
        # 실시간 데이터 생성
        for device_id in data_manager.devices.keys():
            data_manager.generate_real_time_data(device_id)
        
        # 자동 새로고침
        time.sleep(0.1)
        try:
            st.rerun()  # Streamlit >= 1.27.0
        except AttributeError:
            st.experimental_rerun()  # Streamlit < 1.27.0
    
    # 새로고침 버튼
    if st.sidebar.button("수동 새로고침"):
        for device_id in data_manager.devices.keys():
            data_manager.generate_real_time_data(device_id)
        try:
            st.rerun()  # Streamlit >= 1.27.0
        except AttributeError:
            st.experimental_rerun()  # Streamlit < 1.27.0
    
    # 메인 대시보드
    col1, col2, col3, col4 = st.columns(4)
    
    # 전체 요약 정보
    summary = data_manager.get_all_devices_summary()
    
    if summary:
        # 전체 통계
        total_devices = len(summary)
        healthy_devices = sum(1 for d in summary.values() if d['status'] == 'normal')
        warning_devices = sum(1 for d in summary.values() if d['status'] == 'warning')
        critical_devices = sum(1 for d in summary.values() if d['status'] == 'critical')
        
        avg_health = np.mean([d['health_score'] for d in summary.values()])
        
        with col1:
            st.metric("총 디바이스", total_devices)
        with col2:
            st.metric("정상", healthy_devices, delta=f"{healthy_devices/total_devices*100:.1f}%")
        with col3:
            st.metric("경고", warning_devices, delta=f"{warning_devices/total_devices*100:.1f}%")
        with col4:
            st.metric("위험", critical_devices, delta=f"{critical_devices/total_devices*100:.1f}%")
        
        # 전체 평균 건강도
        st.metric("전체 평균 건강도", f"{avg_health:.1f}%")
    
    st.markdown("---")
    
    # 디바이스 개요 차트
    if summary:
        st.subheader("📊 디바이스 상태 개요")
        overview_chart = create_device_overview_chart(summary)
        st.plotly_chart(overview_chart, use_container_width=True)
    
    st.markdown("---")
    
    # 개별 디바이스 분석
    st.subheader("🔍 개별 디바이스 분석")
    
    selected_device = st.selectbox(
        "분석할 디바이스 선택:",
        list(data_manager.devices.keys())
    )
    
    if selected_device:
        device_data = data_manager.get_device_data(selected_device, count=200)
        
        if not device_data.empty:
            # 디바이스 상세 정보
            latest = device_data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("현재 건강도", f"{latest['health_score']:.1f}%")
            with col2:
                st.metric("이상 점수", f"{latest['anomaly_score']:.3f}")
            with col3:
                st.metric("상태", latest['status'])
            with col4:
                st.metric("운영 시간", f"{latest['operating_hours']:.1f}h")
            
            # 시계열 차트
            st.subheader("📈 시계열 추이")
            
            chart_metrics = st.multiselect(
                "표시할 메트릭 선택:",
                ['health_score', 'anomaly_score', 'temperature', 'vibration_x', 
                 'vibration_y', 'pressure', 'rotation_speed', 'current'],
                default=['health_score', 'anomaly_score', 'temperature']
            )
            
            if chart_metrics:
                ts_chart = create_time_series_chart(device_data, chart_metrics)
                st.plotly_chart(ts_chart, use_container_width=True)
            
            # 이상 패턴 히트맵
            st.subheader("🔥 센서 이상 패턴")
            heatmap = create_anomaly_heatmap(device_data)
            st.plotly_chart(heatmap, use_container_width=True)
            
            # 예측 결과 (모델이 로드된 경우)
            if data_manager.model and data_manager.model.model_trained:
                st.subheader("🔮 예측 결과")
                try:
                    prediction = data_manager.model.predict(device_data, selected_device)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("고장 확률", f"{prediction['maintenance_probability']:.1%}")
                    with col2:
                        maintenance_status = "필요" if prediction['maintenance_needed'] else "불필요"
                        st.metric("유지보수", maintenance_status)
                    with col3:
                        st.metric("위험 수준", prediction['risk_level'])
                        
                except Exception as e:
                    st.error(f"예측 오류: {e}")
            
            # 원시 데이터 테이블
            if st.checkbox("원시 데이터 보기"):
                st.subheader("📋 원시 데이터")
                st.dataframe(device_data.tail(20))
    
    # 알림 시스템
    st.markdown("---")
    st.subheader("🚨 실시간 알림")
    
    alerts = []
    for device_id, info in summary.items():
        if info['status'] == 'critical':
            alerts.append(f"🔴 {device_id}: 위험 상태 - 건강도 {info['health_score']:.1f}%")
        elif info['status'] == 'warning':
            alerts.append(f"🟡 {device_id}: 경고 상태 - 건강도 {info['health_score']:.1f}%")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ 모든 디바이스가 정상 상태입니다.")


def analytics_page():
    """분석 페이지"""
    st.title("📊 IoT 데이터 분석")
    
    # 샘플 데이터 로드
    st.subheader("📈 과거 데이터 분석")
    
    if st.button("샘플 데이터 로드"):
        with st.spinner("샘플 데이터 생성 중..."):
            sample_data = load_sample_data()
            st.session_state.sample_data = sample_data
            st.success(f"데이터 로드 완료: {len(sample_data)}개 레코드")
    
    if 'sample_data' in st.session_state:
        data = st.session_state.sample_data
        
        # 데이터 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 레코드", len(data))
        with col2:
            st.metric("디바이스 수", data['device_id'].nunique())
        with col3:
            st.metric("평균 건강도", f"{data['health_score'].mean():.1f}%")
        with col4:
            st.metric("고장 위험 비율", f"{(data['anomaly_score'] > 0.7).mean():.1%}")
        
        # 디바이스별 분석
        st.subheader("디바이스별 성능 분석")
        
        device_stats = data.groupby('device_id').agg({
            'health_score': ['mean', 'min', 'std'],
            'anomaly_score': ['mean', 'max'],
            'operating_hours': 'max'
        }).round(2)
        
        device_stats.columns = ['평균_건강도', '최소_건강도', '건강도_표준편차', 
                               '평균_이상점수', '최대_이상점수', '총_운영시간']
        
        st.dataframe(device_stats)
        
        # 상관관계 분석
        st.subheader("센서 상관관계 분석")
        
        sensor_cols = ['temperature', 'vibration_x', 'vibration_y', 'vibration_z', 
                      'pressure', 'rotation_speed', 'current', 'voltage', 'noise_level']
        available_sensors = [col for col in sensor_cols if col in data.columns]
        
        if available_sensors:
            corr_matrix = data[available_sensors + ['health_score', 'anomaly_score']].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="센서 간 상관관계",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 고장 패턴 분석
        st.subheader("고장 패턴 분석")
        
        # 건강도별 분포
        fig_health = px.histogram(
            data, 
            x='health_score', 
            nbins=20,
            title="건강도 분포",
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_health, use_container_width=True)
        
        # 시간대별 이상 패턴
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            
            hourly_anomaly = data.groupby('hour')['anomaly_score'].mean().reset_index()
            
            fig_hourly = px.line(
                hourly_anomaly,
                x='hour',
                y='anomaly_score',
                title="시간대별 평균 이상 점수",
                markers=True
            )
            st.plotly_chart(fig_hourly, use_container_width=True)


def model_training_page():
    """모델 훈련 페이지"""
    st.title("🤖 AI 모델 훈련")
    
    st.markdown("""
    이 페이지에서는 새로운 예측 모델을 훈련하거나 기존 모델을 업데이트할 수 있습니다.
    """)
    
    # 훈련 매개변수 설정
    st.subheader("🔧 훈련 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        sequence_length = st.slider("시퀀스 길이", 10, 120, 60)
        prediction_horizon = st.slider("예측 기간", 1, 30, 10)
    with col2:
        epochs = st.slider("훈련 에포크", 10, 100, 50)
        batch_size = st.selectbox("배치 크기", [16, 32, 64, 128], index=1)
    
    # 데이터 준비
    if st.button("훈련 데이터 생성"):
        with st.spinner("훈련 데이터 생성 중..."):
            training_data = load_sample_data()
            st.session_state.training_data = training_data
            st.success("훈련 데이터 준비 완료!")
    
    # 모델 훈련
    if st.button("모델 훈련 시작") and 'training_data' in st.session_state:
        try:
            with st.spinner("모델 훈련 중... 시간이 오래 걸릴 수 있습니다."):
                # 모델 초기화
                model = IoTPredictiveMaintenanceModel(
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon
                )
                
                # 훈련 실행
                training_data = st.session_state.training_data
                
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 간단한 훈련 (실제로는 더 복잡한 콜백이 필요)
                status_text.text("모델 훈련 중...")
                history = model.train(training_data, epochs=epochs, batch_size=batch_size)
                
                progress_bar.progress(100)
                status_text.text("훈련 완료!")
                
                # 모델 저장
                model.save_model("dashboard_model")
                
                st.success("모델 훈련 및 저장 완료!")
                
                # 훈련 결과 표시
                if hasattr(history, 'history'):
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=['Loss', 'Accuracy']
                    )
                    
                    fig.add_trace(
                        go.Scatter(y=history.history['loss'], name='Training Loss'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(y=history.history['val_loss'], name='Validation Loss'),
                        row=1, col=1
                    )
                    
                    if 'accuracy' in history.history:
                        fig.add_trace(
                            go.Scatter(y=history.history['accuracy'], name='Training Accuracy'),
                            row=1, col=2
                        )
                        fig.add_trace(
                            go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'),
                            row=1, col=2
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"훈련 중 오류 발생: {e}")


def settings_page():
    """설정 페이지"""
    st.title("⚙️ 시스템 설정")
    
    st.subheader("📡 데이터 소스 설정")
    
    # Kafka 설정
    with st.expander("Kafka 설정"):
        kafka_servers = st.text_input("Kafka 서버", "localhost:9092")
        kafka_topic = st.text_input("Kafka 토픽", "iot_sensor_data")
        
        if st.button("Kafka 연결 테스트"):
            st.info("Kafka 연결 테스트 기능은 실제 Kafka 서버가 필요합니다.")
    
    # Redis 설정
    with st.expander("Redis 설정"):
        redis_host = st.text_input("Redis 호스트", "localhost")
        redis_port = st.number_input("Redis 포트", value=6379)
        
        if st.button("Redis 연결 테스트"):
            try:
                import redis
                r = redis.Redis(host=redis_host, port=redis_port)
                r.ping()
                st.success("Redis 연결 성공!")
            except:
                st.error("Redis 연결 실패")
    
    st.subheader("🚨 알림 설정")
    
    # 임계값 설정
    health_threshold = st.slider("건강도 경고 임계값", 0, 100, 70)
    anomaly_threshold = st.slider("이상 점수 경고 임계값", 0.0, 1.0, 0.7)
    
    # 이메일 알림 설정
    with st.expander("이메일 알림 설정"):
        email_enabled = st.checkbox("이메일 알림 활성화")
        if email_enabled:
            smtp_server = st.text_input("SMTP 서버")
            smtp_port = st.number_input("SMTP 포트", value=587)
            sender_email = st.text_input("발신자 이메일")
            receiver_emails = st.text_area("수신자 이메일 (줄바꿈으로 구분)")
    
    st.subheader("💾 데이터 관리")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("데이터 내보내기"):
            st.info("데이터 내보내기 기능")
    
    with col2:
        if st.button("시스템 초기화"):
            if st.checkbox("정말로 초기화하시겠습니까?"):
                st.warning("시스템 초기화는 구현되지 않았습니다.")


if __name__ == "__main__":
    # 네비게이션 설정
    st.sidebar.title("🏭 IoT 대시보드")
    
    pages = {
        "실시간 모니터링": main,
        "데이터 분석": analytics_page,
        "모델 훈련": model_training_page,
        "설정": settings_page
    }
    
    page = st.sidebar.selectbox("페이지 선택", list(pages.keys()))
    
    # 선택된 페이지 실행
    pages[page]()