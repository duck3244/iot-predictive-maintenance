"""
XML 데이터 처리 실행 스크립트
실제 .dat 파일들을 처리하여 AI 학습용 데이터로 변환
"""

import os
import sys
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 현재 디렉토리를 Python path에 추가 (모듈 import를 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from xml_data_processor import XMLDataProcessor
    print("✅ xml_data_processor 모듈 로드 성공")
except ImportError as e:
    print(f"❌ xml_data_processor 모듈 로드 실패: {e}")
    print("xml_data_processor.py 파일이 현재 디렉토리 또는 상위 디렉토리에 있는지 확인하세요.")
    sys.exit(1)


def main():
    """메인 실행 함수"""
    print("="*70)
    print(" 🤖 SCARA 로봇 XML 데이터 처리 시작")
    print("="*70)
    
    # 데이터 디렉토리 확인
    data_dir = "data/xml_data"
    
    # 상대 경로로 데이터 디렉토리 찾기
    possible_paths = [
        data_dir,
        os.path.join("..", data_dir),
        os.path.join(".", "xml_data"),
        "xml_data",
        "."  # 현재 디렉토리
    ]
    
    actual_data_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            dat_files = [f for f in os.listdir(path) if f.endswith('.dat')]
            if dat_files:
                actual_data_dir = path
                print(f"📁 데이터 디렉토리 발견: {os.path.abspath(path)}")
                print(f"📄 .dat 파일 수: {len(dat_files)}개")
                break
    
    if not actual_data_dir:
        print("❌ .dat 파일을 찾을 수 없습니다.")
        print("다음 위치 중 하나에 .dat 파일을 배치하세요:")
        for path in possible_paths:
            print(f"   - {os.path.abspath(path)}")
        return
    
    # XML 데이터 처리기 초기화
    print(f"\n🔧 XML 데이터 처리기 초기화...")
    processor = XMLDataProcessor(actual_data_dir)
    
    # 파일 목록 확인
    dat_files = [f for f in os.listdir(actual_data_dir) if f.endswith('.dat')]
    print(f"처리할 파일 목록 (처음 5개):")
    for i, filename in enumerate(dat_files[:5]):
        print(f"   {i+1}. {filename}")
    if len(dat_files) > 5:
        print(f"   ... 및 {len(dat_files)-5}개 추가 파일")
    
    # 사용자 확인
    response = input(f"\n🚀 {len(dat_files)}개 파일을 처리하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("처리가 취소되었습니다.")
        return
    
    # 전체 파이프라인 실행
    print(f"\n⚡ 전체 데이터 처리 파이프라인 시작...")
    print("이 과정은 시간이 오래 걸릴 수 있습니다...")
    
    try:
        # 처리 실행
        result_df = processor.process_full_pipeline(
            file_pattern="*.dat",
            time_interval='5S',  # 5초 간격으로 샘플링 (성능 최적화)
            save_result=True
        )
        
        if result_df.empty:
            print("❌ 처리 결과가 비어있습니다.")
            return
        
        print(f"\n🎉 처리 완료!")
        print("="*50)
        
        # 결과 요약
        print("📊 처리 결과 요약:")
        print(f"   총 시간 포인트: {len(result_df):,}개")
        print(f"   총 특성(컬럼) 수: {len(result_df.columns)}개")
        
        if 'timestamp' in result_df.columns:
            time_range = result_df['timestamp'].max() - result_df['timestamp'].min()
            print(f"   시간 범위: {time_range}")
            print(f"   시작 시간: {result_df['timestamp'].min()}")
            print(f"   종료 시간: {result_df['timestamp'].max()}")
        
        # 주요 통계
        if 'health_score' in result_df.columns:
            avg_health = result_df['health_score'].mean()
            min_health = result_df['health_score'].min()
            max_health = result_df['health_score'].max()
            print(f"   평균 건강도: {avg_health:.1f}% (범위: {min_health:.1f}% ~ {max_health:.1f}%)")
        
        if 'anomaly_score' in result_df.columns:
            avg_anomaly = result_df['anomaly_score'].mean()
            max_anomaly = result_df['anomaly_score'].max()
            print(f"   평균 이상점수: {avg_anomaly:.3f} (최대: {max_anomaly:.3f})")
        
        if 'status' in result_df.columns:
            status_counts = result_df['status'].value_counts()
            print(f"   상태 분포:")
            for status, count in status_counts.items():
                percentage = count / len(result_df) * 100
                print(f"     - {status}: {count:,}개 ({percentage:.1f}%)")
        
        # 저장된 파일 정보
        csv_files = [f for f in os.listdir(actual_data_dir) if f.startswith('processed_robot_data_') and f.endswith('.csv')]
        if csv_files:
            latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(actual_data_dir, x)))
            file_path = os.path.join(actual_data_dir, latest_file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   저장된 파일: {latest_file}")
            print(f"   파일 크기: {file_size:.1f} MB")
        
        # 데이터 미리보기
        print(f"\n📋 데이터 미리보기 (처음 3행):")
        print(result_df.head(3).to_string())
        
        # 컬럼 목록
        print(f"\n📝 생성된 특성 목록:")
        sensor_cols = [col for col in result_df.columns if any(keyword in col for keyword in ['joint', 'pos', 'velocity', 'torque', 'error'])]
        engineered_cols = [col for col in result_df.columns if any(keyword in col for keyword in ['total_', 'mean_', 'max_', '_ma_', '_std_'])]
        target_cols = [col for col in result_df.columns if col in ['health_score', 'anomaly_score', 'status']]
        
        print(f"   센서 데이터 ({len(sensor_cols)}개): {sensor_cols[:5]}{'...' if len(sensor_cols) > 5 else ''}")
        print(f"   엔지니어링 특성 ({len(engineered_cols)}개): {engineered_cols[:5]}{'...' if len(engineered_cols) > 5 else ''}")
        print(f"   타겟 변수 ({len(target_cols)}개): {target_cols}")
        
        # 다음 단계 안내
        print(f"\n🚀 다음 단계:")
        print(f"1. AI 모델 훈련:")
        print(f"   from predictive_model import IoTPredictiveMaintenanceModel")
        print(f"   model = IoTPredictiveMaintenanceModel()")
        print(f"   model.train(result_df, epochs=50)")
        print(f"")
        print(f"2. 저장된 CSV 파일 직접 로드:")
        if csv_files:
            print(f"   df = pd.read_csv('{os.path.join(actual_data_dir, latest_file)}')")
        print(f"")
        print(f"3. 실시간 스트리밍에 통합:")
        print(f"   from kafka_streaming import StreamingManager")
        
        # 저장된 파일 경로 반환
        if csv_files:
            return os.path.join(actual_data_dir, latest_file)
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        print("오류 상세 정보:")
        import traceback
        traceback.print_exc()
        return None


def quick_test():
    """빠른 테스트 - 첫 번째 파일만 처리"""
    print("🧪 빠른 테스트 모드 (첫 번째 파일만 처리)")
    
    # 데이터 디렉토리 찾기
    data_dir = "."
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    
    if not dat_files:
        print("❌ 현재 디렉토리에 .dat 파일이 없습니다.")
        return
    
    first_file = dat_files[0]
    print(f"📄 테스트 파일: {first_file}")
    
    # 처리기 초기화
    processor = XMLDataProcessor(data_dir)
    
    try:
        # 단일 파일 파싱
        records = processor.parse_xml_file(first_file)
        print(f"✅ 파싱 완료: {len(records)}개 레코드")
        
        if records:
            # DataFrame 변환
            df = pd.DataFrame(records)
            print(f"📊 DataFrame 생성: {len(df)}행 x {len(df.columns)}열")
            
            # 샘플 데이터 출력
            print(f"\n📋 샘플 데이터:")
            print(df.head().to_string())
            
            # 태그명 확인
            unique_tags = df['tag_name'].unique()
            print(f"\n🏷️  발견된 태그 ({len(unique_tags)}개):")
            for i, tag in enumerate(unique_tags[:10]):
                print(f"   {i+1}. {tag}")
            if len(unique_tags) > 10:
                print(f"   ... 및 {len(unique_tags)-10}개 추가")
    
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        result_file = main()
        
        if result_file:
            print(f"\n💾 처리 완료된 데이터 파일: {result_file}")
            print("이 파일을 AI 모델 훈련에 사용할 수 있습니다!")
