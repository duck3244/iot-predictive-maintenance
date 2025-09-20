"""
IoT 알림 시스템
다양한 알림 조건과 알림 방식을 관리합니다.
"""

import smtplib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

from core.config import alert_config
from core.utils import setup_logging, create_alert

logger = setup_logging(__name__)


class AlertType(Enum):
    """알림 타입"""
    HEALTH_LOW = "health_low"
    ANOMALY_HIGH = "anomaly_high"
    PREDICTION_HIGH = "prediction_high"
    SENSOR_FAULT = "sensor_fault"
    MAINTENANCE_DUE = "maintenance_due"
    SYSTEM_ERROR = "system_error"
    CONNECTION_LOST = "connection_lost"


class AlertPriority(Enum):
    """알림 우선순위"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """알림 데이터 클래스"""
    id: str
    device_id: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    timestamp: datetime
    status: str = "active"
    metadata: Dict = None
    acknowledged_by: str = None
    acknowledged_at: datetime = None
    resolved_at: datetime = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        """딕셔너리에서 생성"""
        data = data.copy()
        data['alert_type'] = AlertType(data['alert_type'])
        data['priority'] = AlertPriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('acknowledged_at'):
            data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        return cls(**data)


class AlertRule:
    """알림 규칙"""
    
    def __init__(self, name: str, condition: Callable[[Dict], bool], 
                 alert_type: AlertType, priority: AlertPriority, 
                 message_template: str, cooldown_minutes: int = 15):
        self.name = name
        self.condition = condition
        self.alert_type = alert_type
        self.priority = priority
        self.message_template = message_template
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = {}  # device_id -> datetime
    
    def should_trigger(self, device_id: str, data: Dict) -> bool:
        """알림을 트리거해야 하는지 확인"""
        # 조건 확인
        if not self.condition(data):
            return False
        
        # 쿨다운 확인
        if device_id in self.last_triggered:
            time_since_last = datetime.now() - self.last_triggered[device_id]
            if time_since_last < timedelta(minutes=self.cooldown_minutes):
                return False
        
        return True
    
    def trigger(self, device_id: str, data: Dict) -> Alert:
        """알림 생성"""
        self.last_triggered[device_id] = datetime.now()
        
        message = self.message_template.format(
            device_id=device_id,
            **data
        )
        
        alert = Alert(
            id=f"{device_id}_{self.alert_type.value}_{int(time.time())}",
            device_id=device_id,
            alert_type=self.alert_type,
            priority=self.priority,
            message=message,
            timestamp=datetime.now(),
            metadata=data
        )
        
        return alert


class EmailNotifier:
    """이메일 알림 발송"""
    
    def __init__(self):
        self.enabled = alert_config.EMAIL_ENABLED
        self.smtp_server = alert_config.SMTP_SERVER
        self.smtp_port = alert_config.SMTP_PORT
        self.username = alert_config.EMAIL_USERNAME
        self.password = alert_config.EMAIL_PASSWORD
        self.from_email = alert_config.EMAIL_FROM
        self.to_emails = alert_config.EMAIL_TO
    
    def send_alert(self, alert: Alert) -> bool:
        """알림 이메일 전송"""
        if not self.enabled or not self.to_emails:
            return True
        
        try:
            # 이메일 메시지 구성
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.priority.value.upper()}] IoT 알림: {alert.device_id}"
            
            # 이메일 본문
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # SMTP 서버 연결 및 전송
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            logger.info(f"이메일 알림 전송 성공: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"이메일 전송 실패: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """이메일 본문 생성"""
        priority_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        
        color = priority_colors.get(alert.priority.value, '#6c757d')
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <h2 style="color: {color};">IoT 시스템 알림</h2>
                
                <table style="border-collapse: collapse; width: 100%;">
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">디바이스 ID</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.device_id}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">알림 타입</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.alert_type.value}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">우선순위</td>
                        <td style="border: 1px solid #ddd; padding: 8px; color: {color}; font-weight: bold;">
                            {alert.priority.value.upper()}
                        </td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">시간</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">메시지</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.message}</td>
                    </tr>
                </table>
                
                <p style="margin-top: 20px; color: #666;">
                    이 알림은 IoT 예측 유지보수 시스템에서 자동으로 생성되었습니다.
                </p>
            </div>
        </body>
        </html>
        """


class AlertManager:
    """알림 관리자"""
    
    def __init__(self):
        self.rules = []
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_history = deque(maxlen=1000)
        self.notifiers = []
        self.callbacks = []
        self.lock = threading.RLock()
        
        # 기본 알림 규칙 설정
        self._setup_default_rules()
        
        # 이메일 알림 설정
        if alert_config.EMAIL_ENABLED:
            self.notifiers.append(EmailNotifier())
        
        logger.info("알림 관리자 초기화 완료")
    
    def _setup_default_rules(self):
        """기본 알림 규칙 설정"""
        # 건강도 저하 알림
        self.add_rule(AlertRule(
            name="health_low",
            condition=lambda data: data.get('health_score', 100) < alert_config.HEALTH_ALERT_THRESHOLD,
            alert_type=AlertType.HEALTH_LOW,
            priority=AlertPriority.MEDIUM,
            message_template="디바이스 {device_id}의 건강도가 {health_score:.1f}%로 임계값 이하입니다.",
            cooldown_minutes=alert_config.ALERT_COOLDOWN_MINUTES
        ))
        
        # 건강도 위험 알림
        self.add_rule(AlertRule(
            name="health_critical",
            condition=lambda data: data.get('health_score', 100) < 30,
            alert_type=AlertType.HEALTH_LOW,
            priority=AlertPriority.CRITICAL,
            message_template="디바이스 {device_id}의 건강도가 위험 수준({health_score:.1f}%)입니다. 즉시 점검이 필요합니다.",
            cooldown_minutes=5
        ))
        
        # 이상 점수 높음 알림
        self.add_rule(AlertRule(
            name="anomaly_high",
            condition=lambda data: data.get('anomaly_score', 0) > alert_config.ANOMALY_ALERT_THRESHOLD,
            alert_type=AlertType.ANOMALY_HIGH,
            priority=AlertPriority.HIGH,
            message_template="디바이스 {device_id}에서 이상 패턴이 감지되었습니다. (이상 점수: {anomaly_score:.3f})",
            cooldown_minutes=alert_config.ALERT_COOLDOWN_MINUTES
        ))
        
        # 센서 오류 알림
        self.add_rule(AlertRule(
            name="sensor_fault",
            condition=self._check_sensor_fault,
            alert_type=AlertType.SENSOR_FAULT,
            priority=AlertPriority.HIGH,
            message_template="디바이스 {device_id}에서 센서 오류가 감지되었습니다.",
            cooldown_minutes=alert_config.ALERT_COOLDOWN_MINUTES
        ))
    
    def _check_sensor_fault(self, data: Dict) -> bool:
        """센서 오류 확인"""
        sensors = data.get('sensors', {})
        
        # 센서값이 비정상적으로 높거나 낮은 경우
        for sensor_name, value in sensors.items():
            if sensor_name == 'temperature' and (value < 0 or value > 150):
                return True
            elif sensor_name.startswith('vibration') and (value < 0 or value > 10):
                return True
            elif sensor_name == 'pressure' and (value < 0 or value > 10):
                return True
            elif sensor_name == 'current' and (value < 0 or value > 50):
                return True
        
        return False
    
    def add_rule(self, rule: AlertRule):
        """알림 규칙 추가"""
        with self.lock:
            self.rules.append(rule)
            logger.info(f"알림 규칙 추가: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """알림 규칙 제거"""
        with self.lock:
            for i, rule in enumerate(self.rules):
                if rule.name == rule_name:
                    del self.rules[i]
                    logger.info(f"알림 규칙 제거: {rule_name}")
                    return True
            return False
    
    def add_notifier(self, notifier):
        """알림 발송자 추가"""
        self.notifiers.append(notifier)
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """알림 콜백 추가"""
        self.callbacks.append(callback)
    
    def process_data(self, device_id: str, data: Dict):
        """데이터 처리 및 알림 확인"""
        with self.lock:
            triggered_alerts = []
            
            for rule in self.rules:
                if rule.should_trigger(device_id, data):
                    alert = rule.trigger(device_id, data)
                    triggered_alerts.append(alert)
            
            # 알림 처리
            for alert in triggered_alerts:
                self._handle_alert(alert)
    
    def _handle_alert(self, alert: Alert):
        """알림 처리"""
        # 활성 알림에 추가
        self.active_alerts[alert.id] = alert
        
        # 히스토리에 추가
        self.alert_history.append(alert)
        
        logger.warning(f"새 알림 생성: {alert.message}")
        
        # 알림 발송
        for notifier in self.notifiers:
            try:
                notifier.send_alert(alert)
            except Exception as e:
                logger.error(f"알림 발송 실패: {e}")
        
        # 콜백 실행
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"알림 콜백 실패: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """알림 확인 처리"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                alert.status = "acknowledged"
                
                logger.info(f"알림 확인됨: {alert_id} by {user}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """알림 해결 처리"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.now()
                alert.status = "resolved"
                
                # 활성 알림에서 제거
                del self.active_alerts[alert_id]
                
                logger.info(f"알림 해결됨: {alert_id} by {user}")
                return True
            return False
    
    def get_active_alerts(self, device_id: str = None) -> List[Alert]:
        """활성 알림 조회"""
        with self.lock:
            alerts = list(self.active_alerts.values())
            
            if device_id:
                alerts = [alert for alert in alerts if alert.device_id == device_id]
            
            # 우선순위와 시간순 정렬
            priority_order = {
                AlertPriority.CRITICAL: 0,
                AlertPriority.HIGH: 1,
                AlertPriority.MEDIUM: 2,
                AlertPriority.LOW: 3
            }
            
            alerts.sort(key=lambda a: (priority_order[a.priority], a.timestamp), reverse=True)
            return alerts
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """알림 히스토리 조회"""
        with self.lock:
            history = list(self.alert_history)
            return history[-limit:] if limit else history
    
    def get_alert_statistics(self) -> Dict:
        """알림 통계"""
        with self.lock:
            total_alerts = len(self.alert_history)
            active_count = len(self.active_alerts)
            
            # 타입별 통계
            type_counts = defaultdict(int)
            priority_counts = defaultdict(int)
            
            for alert in self.alert_history:
                type_counts[alert.alert_type.value] += 1
                priority_counts[alert.priority.value] += 1
            
            # 최근 24시간 알림
            yesterday = datetime.now() - timedelta(days=1)
            recent_alerts = [
                alert for alert in self.alert_history 
                if alert.timestamp > yesterday
            ]
            
            return {
                'total_alerts': total_alerts,
                'active_alerts': active_count,
                'recent_24h': len(recent_alerts),
                'by_type': dict(type_counts),
                'by_priority': dict(priority_counts),
                'rules_count': len(self.rules)
            }
    
    def create_manual_alert(self, device_id: str, message: str, 
                          priority: AlertPriority = AlertPriority.MEDIUM) -> Alert:
        """수동 알림 생성"""
        alert = Alert(
            id=f"manual_{device_id}_{int(time.time())}",
            device_id=device_id,
            alert_type=AlertType.SYSTEM_ERROR,
            priority=priority,
            message=message,
            timestamp=datetime.now(),
            metadata={'manual': True}
        )
        
        self._handle_alert(alert)
        return alert


class AlertDashboard:
    """알림 대시보드 헬퍼"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
    
    def get_dashboard_data(self) -> Dict:
        """대시보드용 알림 데이터"""
        active_alerts = self.alert_manager.get_active_alerts()
        statistics = self.alert_manager.get_alert_statistics()
        
        # 우선순위별 카운트
        priority_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for alert in active_alerts:
            priority_counts[alert.priority.value] += 1
        
        # 최근 알림 (최대 10개)
        recent_alerts = [
            alert.to_dict() for alert in active_alerts[:10]
        ]
        
        return {
            'active_count': len(active_alerts),
            'priority_breakdown': priority_counts,
            'recent_alerts': recent_alerts,
            'statistics': statistics
        }


if __name__ == "__main__":
    # 알림 시스템 테스트
    print("알림 시스템 테스트 시작...")
    
    manager = AlertManager()
    
    # 테스트 콜백
    def test_callback(alert: Alert):
        print(f"콜백 알림: [{alert.priority.value.upper()}] {alert.message}")
    
    manager.add_callback(test_callback)
    
    # 테스트 데이터
    test_cases = [
        {
            'device_id': 'TEST_001',
            'health_score': 60,  # 건강도 저하
            'anomaly_score': 0.3,
            'sensors': {'temperature': 75, 'current': 15}
        },
        {
            'device_id': 'TEST_002',
            'health_score': 20,  # 위험 수준
            'anomaly_score': 0.8,  # 이상 점수 높음
            'sensors': {'temperature': 95, 'current': 25}
        },
        {
            'device_id': 'TEST_003',
            'health_score': 90,
            'anomaly_score': 0.1,
            'sensors': {'temperature': 200, 'current': 15}  # 센서 오류
        }
    ]
    
    # 테스트 실행
    for i, test_data in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        manager.process_data(test_data['device_id'], test_data)
    
    # 통계 출력
    print(f"\n알림 통계:")
    stats = manager.get_alert_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n알림 시스템 테스트 완료!")
