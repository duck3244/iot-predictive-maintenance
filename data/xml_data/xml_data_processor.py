"""
수정된 XML 데이터 처리기 - 실제 SCARA 로봇 데이터 구조에 맞춤
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path
import logging
from collections import defaultdict

# 안전한 import
try:
    from config import config
except ImportError:
    class Config:
        DATA_DIR = "data"
    config = Config()

try:
    from utils import setup_logging, validate_dataframe, calculate_health_score
except ImportError:
    def setup_logging(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

    def validate_dataframe(df, required_columns):
        return not df.empty and all(col in df.columns for col in required_columns)

    def calculate_health_score(sensors):
        return 100.0

logger = setup_logging(__name__)


class FixedXMLDataProcessor:
    """실제 SCARA 로봇 데이터에 맞춘 XML 처리기"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(config.DATA_DIR, "xml_data")
        os.makedirs(self.data_dir, exist_ok=True)

        # 실제 데이터에서 발견된 정확한 태그 매핑
        self.tag_mapping = {
            # Joint 1 (J1)
            '::[scararobot]Ax_J1.ActualPosition': 'joint1_actual_pos',
            '::[scararobot]Ax_J1.PositionCommand': 'joint1_cmd_pos',
            '::[scararobot]Ax_J1.PositionError': 'joint1_pos_error',
            '::[scararobot]Ax_J1.TorqueCommand': 'joint1_torque_cmd',
            '::[scararobot]Ax_J1.TorqueFeedback': 'joint1_torque_feedback',

            # Joint 2 (J2)
            '::[scararobot]Ax_J2.ActualPosition': 'joint2_actual_pos',
            '::[scararobot]Ax_J2.PositionCommand': 'joint2_cmd_pos',
            '::[scararobot]Ax_J2.PositionError': 'joint2_pos_error',
            '::[scararobot]Ax_J2.TorqueCommand': 'joint2_torque_cmd',
            '::[scararobot]Ax_J2.TorqueFeedback': 'joint2_torque_feedback',

            # Joint 3 (J3)
            '::[scararobot]Ax_J3.ActualPosition': 'joint3_actual_pos',
            '::[scararobot]Ax_J3.PositionCommand': 'joint3_cmd_pos',
            '::[scararobot]Ax_J3.PositionError': 'joint3_pos_error',
            '::[scararobot]Ax_J3.TorqueCommand': 'joint3_torque_cmd',
            '::[scararobot]Ax_J3.TorqueFeedback': 'joint3_torque_feedback',

            # Joint 6 (J6)
            '::[scararobot]Ax_J6.ActualPosition': 'joint6_actual_pos',
            '::[scararobot]Ax_J6.PositionCommand': 'joint6_cmd_pos',
            '::[scararobot]Ax_J6.PositionError': 'joint6_pos_error',
            '::[scararobot]Ax_J6.TorqueCommand': 'joint6_torque_cmd',
            '::[scararobot]Ax_J6.TorqueFeedback': 'joint6_torque_feedback',

            # Cartesian 좌표계
            '::[scararobot]CS_Cartesian.ActualPosition[0]': 'cartesian_x',
            '::[scararobot]CS_Cartesian.ActualPosition[1]': 'cartesian_y',
            '::[scararobot]CS_Cartesian.ActualPosition[2]': 'cartesian_z',
            '::[scararobot]CS_Cartesian.ActualPosition[3]': 'cartesian_rx',
            '::[scararobot]CS_Cartesian.ActualPosition[4]': 'cartesian_ry',
            '::[scararobot]CS_Cartesian.ActualPosition[5]': 'cartesian_rz',

            # SCARA 좌표계
            '::[scararobot]CS_SCARA.ActualPosition[0]': 'scara_x',
            '::[scararobot]CS_SCARA.ActualPosition[1]': 'scara_y',
            '::[scararobot]CS_SCARA.ActualPosition[2]': 'scara_z',
        }

        logger.info(f"수정된 XML 데이터 처리기 초기화: {self.data_dir}")
        logger.info(f"매핑된 태그 수: {len(self.tag_mapping)}개")

    def parse_xml_file(self, file_path: str) -> List[Dict]:
        """단일 XML 파일 파싱 (수정된 버전)"""
        try:
            # XML 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # XML 파싱 (루트 태그가 이미 있음)
            root = ET.fromstring(content)

            # 데이터 추출
            records = []
            for historical_data in root.findall('.//HistoricalTextData'):
                try:
                    tag_name = historical_data.find('TagName')
                    status = historical_data.find('Status')
                    tag_value = historical_data.find('TagValue')
                    timestamp = historical_data.find('TimeStamp')

                    if all(elem is not None for elem in [tag_name, status, tag_value, timestamp]):
                        record = {
                            'tag_name': tag_name.text.strip() if tag_name.text else '',
                            'status': int(status.text) if status.text else 0,
                            'tag_value': float(tag_value.text) if tag_value.text else 0.0,
                            'timestamp': timestamp.text.strip() if timestamp.text else '',
                            'file_source': os.path.basename(file_path)
                        }
                        records.append(record)

                except Exception as e:
                    logger.debug(f"레코드 파싱 오류: {e}")
                    continue

            logger.debug(f"파일 {os.path.basename(file_path)}: {len(records)}개 레코드 추출")
            return records

        except ET.ParseError as e:
            logger.error(f"XML 파싱 오류 ({file_path}): {e}")
            return []
        except Exception as e:
            logger.error(f"파일 처리 오류 ({file_path}): {e}")
            return []

    def process_multiple_files(self, file_pattern: str = "*.dat", max_files: int = None) -> pd.DataFrame:
        """여러 XML/DAT 파일을 일괄 처리"""
        file_paths = glob.glob(os.path.join(self.data_dir, file_pattern))

        if not file_paths:
            logger.warning(f"파일을 찾을 수 없습니다: {os.path.join(self.data_dir, file_pattern)}")
            return pd.DataFrame()

        # 파일 수 제한 (테스트용)
        if max_files:
            file_paths = file_paths[:max_files]

        logger.info(f"{len(file_paths)}개 파일 처리 시작")

        all_records = []
        processed_files = 0

        for file_path in sorted(file_paths):
            try:
                records = self.parse_xml_file(file_path)
                if records:
                    all_records.extend(records)
                    processed_files += 1

                # 진행 상황 출력
                if processed_files % 50 == 0:
                    logger.info(f"진행 상황: {processed_files}/{len(file_paths)} 파일 처리 완료")

            except Exception as e:
                logger.error(f"파일 처리 실패 ({file_path}): {e}")
                continue

        if not all_records:
            logger.error("처리된 레코드가 없습니다")
            return pd.DataFrame()

        # DataFrame 생성
        df = pd.DataFrame(all_records)
        logger.info(f"총 {len(df)}개 레코드 추출 완료")

        return df

    def clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 정리 및 표준화 (수정된 버전)"""
        if df.empty:
            return df

        logger.info("데이터 정리 및 표준화 시작")
        original_count = len(df)

        # 타임스탬프 정리
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # 표준 센서명으로 변환
        df['sensor_name'] = df['tag_name'].map(self.tag_mapping)

        # 매핑된 센서만 필터링
        mapped_data = df[df['sensor_name'].notna()]

        logger.info(f"매핑 결과:")
        logger.info(f"  원본 레코드: {original_count:,}개")
        logger.info(f"  고유 태그: {df['tag_name'].nunique()}개")
        logger.info(f"  매핑된 태그: {mapped_data['tag_name'].nunique()}개")
        logger.info(f"  매핑된 레코드: {len(mapped_data):,}개")

        if mapped_data.empty:
            logger.error("매핑된 데이터가 없습니다!")
            logger.info("발견된 태그 (상위 10개):")
            for tag in df['tag_name'].value_counts().head(10).index:
                logger.info(f"  - {tag}")
            return pd.DataFrame()

        # 상태값 필터링 (정상 데이터만)
        clean_data = mapped_data[mapped_data['status'] == 0]

        # 정렬
        clean_data = clean_data.sort_values(['timestamp', 'sensor_name']).reset_index(drop=True)

        logger.info(f"정리 후 {len(clean_data)}개 레코드")
        return clean_data

    def pivot_to_timeseries(self, df: pd.DataFrame, time_interval: str = '5S') -> pd.DataFrame:
        """센서 데이터를 시계열 형태로 피벗"""
        if df.empty:
            return df

        logger.info("시계열 데이터로 변환")

        # 피벗 테이블 생성
        pivot_df = df.pivot_table(
            index='timestamp',
            columns='sensor_name',
            values='tag_value',
            aggfunc='mean'  # 같은 시간대 중복 데이터는 평균
        )

        # 시간 간격으로 리샘플링
        if time_interval:
            pivot_df = pivot_df.resample(time_interval).mean()

        # 결측값 처리 (선형 보간)
        pivot_df = pivot_df.interpolate(method='linear', limit_direction='both')

        # 여전히 결측값이 있으면 앞/뒤 값으로 채우기
        pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')

        # 인덱스를 컬럼으로 변환
        pivot_df = pivot_df.reset_index()

        # 컬럼명 정리
        pivot_df.columns.name = None

        logger.info(f"시계열 변환 완료: {len(pivot_df)}개 시점, {len(pivot_df.columns)-1}개 센서")
        return pivot_df

    def add_engineering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 엔지니어링 추가 (SCARA 로봇 특화)"""
        if df.empty:
            return df

        logger.info("SCARA 로봇 특성 엔지니어링 수행")

        # 시간 기반 특성
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['minute'] = df['timestamp'].dt.minute

        # 관절별 위치 오차 특성
        joint_error_cols = [col for col in df.columns if 'pos_error' in col]
        if joint_error_cols:
            # 전체 위치 오차 (RMS)
            df['total_position_error'] = np.sqrt(
                sum(df[col].fillna(0)**2 for col in joint_error_cols) / len(joint_error_cols)
            )

            # 평균 위치 오차
            df['mean_position_error'] = df[joint_error_cols].mean(axis=1)

            # 최대 위치 오차
            df['max_position_error'] = df[joint_error_cols].max(axis=1)

            # 위치 오차 표준편차 (불균형 지표)
            df['position_error_std'] = df[joint_error_cols].std(axis=1)

        # 토크 관련 특성
        torque_cmd_cols = [col for col in df.columns if 'torque_cmd' in col]
        torque_fb_cols = [col for col in df.columns if 'torque_feedback' in col]

        if torque_cmd_cols:
            # 전체 토크 명령
            df['total_torque_cmd'] = df[torque_cmd_cols].abs().sum(axis=1)

            # 토크 불균형
            df['torque_cmd_imbalance'] = df[torque_cmd_cols].std(axis=1)

            # 토크 변화율
            for col in torque_cmd_cols:
                df[f'{col}_change'] = df[col].diff().abs()

        if torque_fb_cols:
            # 토크 피드백 합계
            df['total_torque_feedback'] = df[torque_fb_cols].abs().sum(axis=1)

        # 관절 속도 추정 (위치 변화율)
        actual_pos_cols = [col for col in df.columns if 'actual_pos' in col and 'joint' in col]
        for col in actual_pos_cols:
            joint_name = col.replace('_actual_pos', '')
            df[f'{joint_name}_velocity_est'] = df[col].diff().abs()

        # Cartesian 좌표 특성
        cartesian_cols = [col for col in df.columns if 'cartesian' in col]
        if len(cartesian_cols) >= 3:
            # Cartesian 위치 벡터 크기
            xyz_cols = [col for col in cartesian_cols if any(axis in col for axis in ['_x', '_y', '_z'])]
            if len(xyz_cols) >= 3:
                df['cartesian_distance'] = np.sqrt(
                    sum(df[col].fillna(0)**2 for col in xyz_cols[:3])
                )

        # 롤링 통계 (이동 평균, 표준편차)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        windows = [5, 10, 20]

        for window in windows:
            for col in numeric_cols:
                if col not in ['hour', 'day_of_week', 'minute']:
                    df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window, min_periods=1).std().fillna(0)

        # SCARA 로봇 건강도 계산
        df['health_score'] = self._calculate_scara_health(df)

        # 이상 점수 계산
        df['anomaly_score'] = self._calculate_scara_anomaly(df)

        # 상태 분류
        df['status'] = df.apply(self._classify_scara_status, axis=1)

        # 디바이스 ID 추가
        df['device_id'] = 'SCARA_ROBOT_001'

        logger.info(f"특성 엔지니어링 완료: {len(df.columns)}개 특성")
        return df

    def _calculate_scara_health(self, df: pd.DataFrame) -> pd.Series:
        """SCARA 로봇 건강도 계산"""
        health_scores = []

        for _, row in df.iterrows():
            score = 100.0

            # 위치 오차 페널티 (가장 중요)
            if 'total_position_error' in row and not pd.isna(row['total_position_error']):
                error_penalty = min(60, row['total_position_error'] * 20)
                score -= error_penalty

            # 토크 불균형 페널티
            if 'torque_cmd_imbalance' in row and not pd.isna(row['torque_cmd_imbalance']):
                imbalance_penalty = min(25, row['torque_cmd_imbalance'] * 2)
                score -= imbalance_penalty

            # 과도한 토크 페널티
            if 'total_torque_cmd' in row and not pd.isna(row['total_torque_cmd']):
                if row['total_torque_cmd'] > 100:  # 임계값
                    torque_penalty = min(15, (row['total_torque_cmd'] - 100) * 0.1)
                    score -= torque_penalty

            health_scores.append(max(0, score))

        return pd.Series(health_scores)

    def _calculate_scara_anomaly(self, df: pd.DataFrame) -> pd.Series:
        """SCARA 로봇 이상 점수 계산"""
        anomaly_scores = []

        for _, row in df.iterrows():
            score = 0.0

            # 위치 오차 기반
            if 'total_position_error' in row and not pd.isna(row['total_position_error']):
                score += min(0.6, row['total_position_error'] / 5)

            # 건강도 기반
            if 'health_score' in row:
                score += (100 - row['health_score']) / 150

            # 토크 불균형 기반
            if 'torque_cmd_imbalance' in row and not pd.isna(row['torque_cmd_imbalance']):
                score += min(0.3, row['torque_cmd_imbalance'] / 30)

            # 위치 오차 불균형 기반
            if 'position_error_std' in row and not pd.isna(row['position_error_std']):
                score += min(0.1, row['position_error_std'] / 10)

            anomaly_scores.append(min(1.0, score))

        return pd.Series(anomaly_scores)

    def _classify_scara_status(self, row) -> str:
        """SCARA 로봇 상태 분류"""
        health_score = row.get('health_score', 100)
        anomaly_score = row.get('anomaly_score', 0)

        if health_score >= 85 and anomaly_score < 0.2:
            return 'normal'
        elif health_score >= 60 and anomaly_score < 0.5:
            return 'warning'
        else:
            return 'critical'

    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """처리된 데이터 저장"""
        if filename is None:
            filename = f"processed_scara_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)

        logger.info(f"처리된 데이터 저장: {filepath}")
        return filepath

    def process_full_pipeline(self, file_pattern: str = "*.dat",
                            time_interval: str = '5S',
                            max_files: int = None,
                            save_result: bool = True) -> pd.DataFrame:
        """전체 처리 파이프라인 실행"""
        logger.info("SCARA 로봇 데이터 처리 파이프라인 시작")

        # 1. XML 파일들 파싱
        df_raw = self.process_multiple_files(file_pattern, max_files)
        if df_raw.empty:
            logger.error("원시 데이터가 없습니다")
            return pd.DataFrame()

        # 2. 데이터 정리 및 표준화
        df_clean = self.clean_and_standardize(df_raw)
        if df_clean.empty:
            logger.error("정리된 데이터가 없습니다")
            return pd.DataFrame()

        # 3. 시계열 변환
        df_timeseries = self.pivot_to_timeseries(df_clean, time_interval)
        if df_timeseries.empty:
            logger.error("시계열 변환 실패")
            return pd.DataFrame()

        # 4. 특성 엔지니어링
        df_final = self.add_engineering_features(df_timeseries)

        # 5. 결과 저장
        if save_result:
            saved_path = self.save_processed_data(df_final)
            logger.info(f"최종 결과 저장: {saved_path}")

        # 6. 요약 정보
        logger.info("="*60)
        logger.info("🎉 SCARA 로봇 데이터 처리 완료!")
        logger.info(f"  원시 레코드: {len(df_raw):,}개")
        logger.info(f"  정리 후: {len(df_clean):,}개")
        logger.info(f"  최종 시점: {len(df_final):,}개")
        logger.info(f"  최종 특성: {len(df_final.columns):,}개")

        if not df_final.empty:
            logger.info(f"  시간 범위: {df_final['timestamp'].min()} ~ {df_final['timestamp'].max()}")
            logger.info(f"  평균 건강도: {df_final['health_score'].mean():.1f}%")
            logger.info(f"  평균 이상점수: {df_final['anomaly_score'].mean():.3f}")

            status_counts = df_final['status'].value_counts()
            logger.info(f"  상태 분포: {dict(status_counts)}")

        logger.info("="*60)

        return df_final


def quick_test_fixed_processor():
    """수정된 프로세서 빠른 테스트"""
    print("🧪 수정된 SCARA 로봇 데이터 처리기 테스트")
    print("="*50)

    # 현재 디렉토리에서 처리기 초기화
    processor = FixedXMLDataProcessor('.')

    try:
        # 소수 파일로 테스트 (처음 5개 파일만)
        result_df = processor.process_full_pipeline(
            file_pattern="*.dat",
            time_interval='10S',  # 10초 간격
            max_files=5,  # 처음 5개 파일만
            save_result=True
        )

        if not result_df.empty:
            print(f"\n✅ 테스트 성공!")
            print(f"📊 결과: {len(result_df)}개 시점, {len(result_df.columns)}개 특성")

            print(f"\n📋 주요 특성:")
            feature_categories = {
                '관절 위치': [col for col in result_df.columns if 'actual_pos' in col],
                '위치 오차': [col for col in result_df.columns if 'pos_error' in col],
                '토크': [col for col in result_df.columns if 'torque' in col],
                '엔지니어링': [col for col in result_df.columns if any(x in col for x in ['total_', 'mean_', 'max_'])],
                '타겟': [col for col in result_df.columns if col in ['health_score', 'anomaly_score', 'status']]
            }

            for category, cols in feature_categories.items():
                if cols:
                    print(f"  {category}: {len(cols)}개 - {cols[:3]}{'...' if len(cols) > 3 else ''}")

            print(f"\n📈 통계:")
            print(f"  평균 건강도: {result_df['health_score'].mean():.1f}%")
            print(f"  평균 이상점수: {result_df['anomaly_score'].mean():.3f}")
            print(f"  상태 분포: {dict(result_df['status'].value_counts())}")

        else:
            print(f"\n❌ 테스트 실패: 결과 데이터가 비어있습니다")

    except Exception as e:
        print(f"\n❌ 테스트 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test_fixed_processor()