"""
Flask 기반 REST API 서버
IoT 예측 유지보수 시스템의 API 엔드포인트 제공
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
import traceback
from functools import wraps
import jwt
import os
from werkzeug.security import generate_password_hash, check_password_hash

from data.data_generator import IoTSensorDataGenerator
from models.predictive_model import IoTPredictiveMaintenanceModel
from streaming.kafka_streaming import IoTDataProducer, IoTDataConsumer

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'iot-predictive-maintenance-secret-key')
CORS(app)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 글로벌 변수
model = None
data_producer = None
devices = {}
users = {
    "admin": generate_password_hash("password123"),
    "operator": generate_password_hash("operator123")
}


def token_required(f):
    """JWT 토큰 인증 데코레이터"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated


def handle_errors(f):
    """에러 처리 데코레이터"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Internal server error',
                'message': str(e)
            }), 500
    
    return decorated


# 인증 엔드포인트
@app.route('/api/auth/login', methods=['POST'])
@handle_errors
def login():
    """사용자 로그인"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password required'}), 400
    
    username = data['username']
    password = data['password']
    
    if username in users and check_password_hash(users[username], password):
        token = jwt.encode({
            'user': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'user': username,
            'expires_in': 86400  # 24시간
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401


# 디바이스 관리 엔드포인트
@app.route('/api/devices', methods=['GET'])
@token_required
@handle_errors
def get_devices(current_user):
    """등록된 모든 디바이스 목록 조회"""
    device_list = []
    
    for device_id, generator in devices.items():
        device_info = {
            'device_id': device_id,
            'operating_hours': generator.operating_hours,
            'current_health': generator.current_health,
            'last_maintenance': generator.last_maintenance.isoformat()
        }
        device_list.append(device_info)
    
    return jsonify({
        'devices': device_list,
        'total_count': len(device_list)
    })


@app.route('/api/devices', methods=['POST'])
@token_required
@handle_errors
def create_device(current_user):
    """새 디바이스 등록"""
    data = request.get_json()
    
    if not data or not data.get('device_id'):
        return jsonify({'error': 'device_id is required'}), 400
    
    device_id = data['device_id']
    failure_probability = data.get('failure_probability', 0.02)
    
    if device_id in devices:
        return jsonify({'error': 'Device already exists'}), 409
    
    # 새 디바이스 생성
    devices[device_id] = IoTSensorDataGenerator(
        device_id=device_id,
        failure_probability=failure_probability
    )
    
    logger.info(f"새 디바이스 등록: {device_id}")
    
    return jsonify({
        'message': 'Device created successfully',
        'device_id': device_id
    }), 201


@app.route('/api/devices/<device_id>', methods=['DELETE'])
@token_required
@handle_errors
def delete_device(current_user, device_id):
    """디바이스 삭제"""
    if device_id not in devices:
        return jsonify({'error': 'Device not found'}), 404
    
    del devices[device_id]
    logger.info(f"디바이스 삭제: {device_id}")
    
    return jsonify({'message': 'Device deleted successfully'})


# 센서 데이터 엔드포인트
@app.route('/api/devices/<device_id>/data', methods=['GET'])
@token_required
@handle_errors
def get_device_data(current_user, device_id):
    """특정 디바이스의 실시간 센서 데이터 조회"""
    if device_id not in devices:
        return jsonify({'error': 'Device not found'}), 404
    
    generator = devices[device_id]
    sensor_data = generator.generate_sensor_data()
    
    return jsonify(sensor_data)


@app.route('/api/devices/<device_id>/historical', methods=['GET'])
@token_required
@handle_errors
def get_historical_data(current_user, device_id):
    """디바이스의 과거 데이터 생성 및 조회"""
    if device_id not in devices:
        return jsonify({'error': 'Device not found'}), 404
    
    # 쿼리 파라미터
    days = request.args.get('days', 7, type=int)
    interval_minutes = request.args.get('interval', 5, type=int)
    
    generator = devices[device_id]
    historical_df = generator.generate_historical_data(days=days, interval_minutes=interval_minutes)
    
    # JSON 직렬화 가능한 형태로 변환
    historical_data = historical_df.to_dict('records')
    
    return jsonify({
        'device_id': device_id,
        'period_days': days,
        'interval_minutes': interval_minutes,
        'total_records': len(historical_data),
        'data': historical_data
    })


# 예측 엔드포인트
@app.route('/api/predict/<device_id>', methods=['POST'])
@token_required
@handle_errors
def predict_maintenance(current_user, device_id):
    """특정 디바이스의 유지보수 필요성 예측"""
    global model
    
    if not model or not model.model_trained:
        return jsonify({'error': 'Prediction model not available'}), 503
    
    if device_id not in devices:
        return jsonify({'error': 'Device not found'}), 404
    
    # 과거 데이터 생성 (예측을 위한 충분한 데이터)
    generator = devices[device_id]
    historical_df = generator.generate_historical_data(days=2, interval_minutes=1)
    
    # 예측 수행
    prediction = model.predict(historical_df, device_id)
    
    return jsonify(prediction)


@app.route('/api/predict/batch', methods=['POST'])
@token_required
@handle_errors
def batch_predict(current_user):
    """모든 디바이스에 대한 일괄 예측"""
    global model
    
    if not model or not model.model_trained:
        return jsonify({'error': 'Prediction model not available'}), 503
    
    results = []
    
    for device_id in devices.keys():
        try:
            generator = devices[device_id]
            historical_df = generator.generate_historical_data(days=2, interval_minutes=1)
            prediction = model.predict(historical_df, device_id)
            results.append(prediction)
        except Exception as e:
            logger.error(f"디바이스 {device_id} 예측 오류: {e}")
            results.append({
                'device_id': device_id,
                'error': str(e)
            })
    
    return jsonify({
        'predictions': results,
        'total_devices': len(results)
    })


# 모델 관리 엔드포인트
@app.route('/api/model/info', methods=['GET'])
@token_required
@handle_errors
def get_model_info(current_user):
    """모델 정보 조회"""
    global model
    
    if not model:
        return jsonify({'error': 'Model not loaded'}), 404
    
    info = {
        'model_trained': model.model_trained,
        'sequence_length': model.sequence_length,
        'prediction_horizon': model.prediction_horizon,
        'feature_columns': model.feature_columns
    }
    
    return jsonify(info)


@app.route('/api/model/load', methods=['POST'])
@token_required
@handle_errors
def load_model(current_user):
    """모델 로드"""
    global model
    
    data = request.get_json()
    model_path = data.get('model_path', 'iot_predictive_model')
    
    try:
        model = IoTPredictiveMaintenanceModel()
        model.load_model(model_path)
        
        logger.info(f"모델 로드 성공: {model_path}")
        
        return jsonify({
            'message': 'Model loaded successfully',
            'model_path': model_path
        })
    
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500


# 스트리밍 엔드포인트
@app.route('/api/streaming/start', methods=['POST'])
@token_required
@handle_errors
def start_streaming(current_user):
    """실시간 데이터 스트리밍 시작"""
    global data_producer
    
    data = request.get_json()
    kafka_servers = data.get('kafka_servers', ['localhost:9092'])
    topic = data.get('topic', 'iot_sensor_data')
    interval = data.get('interval_seconds', 5)
    
    try:
        # Producer 초기화
        data_producer = IoTDataProducer(kafka_servers=kafka_servers, topic=topic)
        
        # 등록된 디바이스들을 Producer에 추가
        for device_id, generator in devices.items():
            data_producer.add_device(device_id, generator.failure_probability)
        
        # 스트리밍 시작
        success = data_producer.start_streaming(interval_seconds=interval)
        
        if success:
            logger.info("실시간 스트리밍 시작")
            return jsonify({'message': 'Streaming started successfully'})
        else:
            return jsonify({'error': 'Failed to start streaming'}), 500
    
    except Exception as e:
        logger.error(f"스트리밍 시작 실패: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/streaming/stop', methods=['POST'])
@token_required
@handle_errors
def stop_streaming(current_user):
    """실시간 데이터 스트리밍 중지"""
    global data_producer
    
    if data_producer:
        data_producer.stop_streaming()
        data_producer = None
        logger.info("실시간 스트리밍 중지")
        return jsonify({'message': 'Streaming stopped successfully'})
    
    return jsonify({'error': 'No active streaming session'}), 400


# 통계 엔드포인트
@app.route('/api/stats/summary', methods=['GET'])
@token_required
@handle_errors
def get_system_summary(current_user):
    """시스템 전체 요약 통계"""
    summary = {
        'total_devices': len(devices),
        'devices_by_status': {'normal': 0, 'warning': 0, 'critical': 0},
        'average_health': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    if devices:
        health_scores = []
        
        for device_id, generator in devices.items():
            current_data = generator.generate_sensor_data()
            health_score = current_data['health_score']
            status = current_data['status']
            
            health_scores.append(health_score)
            summary['devices_by_status'][status] += 1
        
        summary['average_health'] = np.mean(health_scores)
    
    return jsonify(summary)


@app.route('/api/stats/device/<device_id>', methods=['GET'])
@token_required
@handle_errors
def get_device_stats(current_user, device_id):
    """특정 디바이스의 상세 통계"""
    if device_id not in devices:
        return jsonify({'error': 'Device not found'}), 404
    
    # 통계 계산을 위한 짧은 기간 데이터 생성
    generator = devices[device_id]
    stats_df = generator.generate_historical_data(days=1, interval_minutes=10)
    
    if stats_df.empty:
        return jsonify({'error': 'No data available'}), 404
    
    stats = {
        'device_id': device_id,
        'current_health': float(stats_df['health_score'].iloc[-1]),
        'health_trend': {
            'mean': float(stats_df['health_score'].mean()),
            'min': float(stats_df['health_score'].min()),
            'max': float(stats_df['health_score'].max()),
            'std': float(stats_df['health_score'].std())
        },
        'anomaly_trend': {
            'mean': float(stats_df['anomaly_score'].mean()),
            'max': float(stats_df['anomaly_score'].max()),
            'current': float(stats_df['anomaly_score'].iloc[-1])
        },
        'operating_hours': float(stats_df['operating_hours'].iloc[-1]),
        'data_points': len(stats_df),
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(stats)


# 헬스체크 엔드포인트
@app.route('/api/health', methods=['GET'])
def health_check():
    """API 서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'model_loaded': model is not None and model.model_trained if model else False,
            'streaming_active': data_producer is not None and data_producer.running if data_producer else False,
            'devices_count': len(devices)
        }
    })


# 초기화 함수
def initialize_system():
    """시스템 초기화"""
    global model, devices
    
    logger.info("시스템 초기화 시작...")
    
    # 기본 디바이스 생성
    default_devices = [
        ('DEVICE_001', 0.02),
        ('DEVICE_002', 0.03),
        ('DEVICE_003', 0.01),
        ('DEVICE_004', 0.04),
        ('DEVICE_005', 0.02)
    ]
    
    for device_id, failure_prob in default_devices:
        devices[device_id] = IoTSensorDataGenerator(
            device_id=device_id,
            failure_probability=failure_prob
        )
        logger.info(f"기본 디바이스 생성: {device_id}")
    
    # 모델 로드 시도
    try:
        model = IoTPredictiveMaintenanceModel()
        model.load_model('iot_predictive_model')
        logger.info("예측 모델 로드 성공")
    except Exception as e:
        logger.warning(f"모델 로드 실패: {e}")
        logger.info("모델 없이 시스템 시작")
    
    logger.info("시스템 초기화 완료")


if __name__ == '__main__':
    # 시스템 초기화
    initialize_system()
    
    # Flask 서버 시작
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"API 서버 시작: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)