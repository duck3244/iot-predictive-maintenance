"""
실제 XML 데이터 구조 분석 및 디버깅 스크립트
실제 .dat 파일의 태그명과 구조를 파악합니다.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from collections import Counter
import sys

def analyze_xml_structure(data_dir="."):
    """실제 XML 파일 구조 분석"""
    
    print("🔍 실제 데이터 구조 분석 시작")
    print("="*50)
    
    # .dat 파일 찾기
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not dat_files:
        print("❌ .dat 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 발견된 파일: {len(dat_files)}개")
    
    # 첫 번째 파일 상세 분석
    first_file = dat_files[0]
    print(f"📄 분석 대상: {first_file}")
    
    try:
        # 파일 읽기
        with open(os.path.join(data_dir, first_file), 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📏 파일 크기: {len(content):,} 문자")
        
        # 처음 1000자 출력
        print(f"\n📋 파일 내용 샘플 (처음 1000자):")
        print("-" * 30)
        print(content[:1000])
        print("-" * 30)
        
        # XML 구조 파싱
        if not content.strip().startswith('<?xml') and not content.strip().startswith('<root'):
            content = f"<root>{content}</root>"
        
        root = ET.fromstring(content)
        
        # 모든 태그명 수집
        all_tags = []
        tag_samples = {}
        
        for historical_data in root.findall('.//HistoricalTextData'):
            tag_name_elem = historical_data.find('TagName')
            tag_value_elem = historical_data.find('TagValue')
            timestamp_elem = historical_data.find('TimeStamp')
            status_elem = historical_data.find('Status')
            
            if tag_name_elem is not None:
                tag_name = tag_name_elem.text.strip() if tag_name_elem.text else ''
                tag_value = tag_value_elem.text if tag_value_elem is not None else ''
                timestamp = timestamp_elem.text if timestamp_elem is not None else ''
                status = status_elem.text if status_elem is not None else ''
                
                all_tags.append(tag_name)
                
                # 각 태그의 샘플 저장 (처음 만나는 경우만)
                if tag_name not in tag_samples:
                    tag_samples[tag_name] = {
                        'value': tag_value,
                        'timestamp': timestamp,
                        'status': status
                    }
        
        # 태그명 통계
        tag_counts = Counter(all_tags)
        
        print(f"\n📊 태그명 분석 결과:")
        print(f"   총 레코드 수: {len(all_tags):,}개")
        print(f"   고유 태그 수: {len(tag_counts)}개")
        
        print(f"\n🏷️  상위 20개 태그명과 빈도:")
        for i, (tag, count) in enumerate(tag_counts.most_common(20), 1):
            sample = tag_samples.get(tag, {})
            print(f"   {i:2d}. {tag}")
            print(f"       빈도: {count:,}회")
            print(f"       샘플값: {sample.get('value', 'N/A')}")
            print(f"       상태: {sample.get('status', 'N/A')}")
            print()
        
        # 태그명 패턴 분석
        print(f"🔍 태그명 패턴 분석:")
        
        # 공통 패턴 찾기
        patterns = {
            'scararobot': [tag for tag in tag_counts.keys() if 'scararobot' in tag.lower()],
            'position': [tag for tag in tag_counts.keys() if 'position' in tag.lower()],
            'velocity': [tag for tag in tag_counts.keys() if 'velocity' in tag.lower()],
            'torque': [tag for tag in tag_counts.keys() if 'torque' in tag.lower()],
            'error': [tag for tag in tag_counts.keys() if 'error' in tag.lower()],
            'actual': [tag for tag in tag_counts.keys() if 'actual' in tag.lower()],
            'command': [tag for tag in tag_counts.keys() if 'command' in tag.lower()],
        }
        
        for pattern_name, matching_tags in patterns.items():
            if matching_tags:
                print(f"   {pattern_name.upper()} 관련 태그 ({len(matching_tags)}개):")
                for tag in matching_tags[:5]:  # 처음 5개만 표시
                    print(f"     - {tag}")
                if len(matching_tags) > 5:
                    print(f"     ... 및 {len(matching_tags)-5}개 추가")
                print()
        
        # 추천 태그 매핑 생성
        print(f"💡 추천 태그 매핑:")
        print("recommended_mapping = {")
        
        for tag in tag_counts.most_common(30):  # 상위 30개 태그
            tag_name = tag[0]
            
            # 센서명 추천
            sensor_name = suggest_sensor_name(tag_name)
            if sensor_name:
                print(f"    '{tag_name}': '{sensor_name}',")
        
        print("}")
        
        return tag_counts, tag_samples
        
    except ET.ParseError as e:
        print(f"❌ XML 파싱 오류: {e}")
        print("\n🔍 원시 데이터 구조 확인:")
        lines = content.split('\n')
        for i, line in enumerate(lines[:20]):
            print(f"   {i+1:2d}: {line}")
        
    except Exception as e:
        print(f"❌ 분석 오류: {e}")
        import traceback
        traceback.print_exc()


def suggest_sensor_name(tag_name):
    """태그명으로부터 센서명 추천"""
    tag_lower = tag_name.lower()
    
    # 관절 번호 추출
    joint_num = None
    for i in range(1, 10):
        if f'j{i}' in tag_lower or f'ax{i}' in tag_lower or f'axis{i}' in tag_lower:
            joint_num = i
            break
    
    # 센서 타입 결정
    if 'actualposition' in tag_lower:
        return f'joint{joint_num}_actual_pos' if joint_num else 'actual_position'
    elif 'positioncommand' in tag_lower:
        return f'joint{joint_num}_cmd_pos' if joint_num else 'cmd_position'
    elif 'positionerror' in tag_lower:
        return f'joint{joint_num}_pos_error' if joint_num else 'position_error'
    elif 'actualvelocity' in tag_lower:
        return f'joint{joint_num}_velocity' if joint_num else 'velocity'
    elif 'actualtorque' in tag_lower:
        return f'joint{joint_num}_torque' if joint_num else 'torque'
    elif 'temperature' in tag_lower:
        return f'joint{joint_num}_temp' if joint_num else 'temperature'
    elif 'current' in tag_lower:
        return f'joint{joint_num}_current' if joint_num else 'current'
    
    return None


def create_corrected_mapping(tag_counts):
    """실제 데이터에서 발견된 태그들로 매핑 생성"""
    
    print("\n🔧 수정된 태그 매핑 생성 중...")
    
    corrected_mapping = {}
    
    for tag_name, count in tag_counts.most_common(50):  # 상위 50개
        sensor_name = suggest_sensor_name(tag_name)
        if sensor_name:
            corrected_mapping[tag_name] = sensor_name
    
    return corrected_mapping


def test_corrected_processor(data_dir="."):
    """수정된 프로세서로 테스트"""
    
    print("\n🧪 수정된 프로세서 테스트")
    print("="*40)
    
    # 먼저 데이터 구조 분석
    tag_counts, tag_samples = analyze_xml_structure(data_dir)
    
    if not tag_counts:
        print("❌ 태그 분석 실패")
        return
    
    # 수정된 매핑 생성
    corrected_mapping = create_corrected_mapping(tag_counts)
    
    print(f"\n📋 생성된 매핑 ({len(corrected_mapping)}개):")
    for original, mapped in list(corrected_mapping.items())[:10]:
        print(f"   '{original}' -> '{mapped}'")
    if len(corrected_mapping) > 10:
        print(f"   ... 및 {len(corrected_mapping)-10}개 추가")
    
    # 프로세서 수정 코드 생성
    print(f"\n🔧 xml_data_processor.py 수정 코드:")
    print("="*50)
    print("# 다음 코드를 xml_data_processor.py의 __init__ 메서드에서 self.tag_mapping 부분을 교체하세요:")
    print()
    print("self.tag_mapping = {")
    for original, mapped in corrected_mapping.items():
        print(f"    '{original}': '{mapped}',")
    print("}")
    
    return corrected_mapping


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XML 데이터 구조 분석")
    parser.add_argument("--dir", default=".", help="데이터 디렉토리 경로")
    parser.add_argument("--test", action="store_true", help="수정된 프로세서 테스트")
    
    args = parser.parse_args()
    
    if args.test:
        test_corrected_processor(args.dir)
    else:
        analyze_xml_structure(args.dir)
