#!/usr/bin/env python3
"""
MM-IMDb 멀티모달 영화 장르 예측 모델 데모 스크립트
Demo script for MM-IMDb multimodal movie genre prediction models

사용법:
python demo.py --mode [quick|full|xai|benchmark]
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='MM-IMDb 모델 데모 실행')
    parser.add_argument('--mode', choices=['quick', 'full', 'xai', 'benchmark'], 
                       default='quick', help='실행할 데모 모드')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='학습 에포크 수 (quick 모드용)')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='배치 크기')
    
    args = parser.parse_args()
    
    print("🎬 MM-IMDb 멀티모달 영화 장르 예측 모델 데모")
    print("=" * 50)
    print(f"실행 모드: {args.mode}")
    print(f"에포크 수: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print("=" * 50)
    
    # Jupyter 노트북 실행 안내
    if args.mode == 'quick':
        print("🚀 빠른 데모 실행 방법:")
        print("1. Jupyter 노트북 시작: jupyter notebook mmimdb_test.ipynb")
        print("2. 모든 셀을 순서대로 실행")
        print("3. 마지막에 run_quick_demo() 함수 호출")
        print("\n또는 명령어로 직접 실행:")
        print("python -c \"import subprocess; subprocess.run(['jupyter', 'notebook', 'mmimdb_test.ipynb'])\"")
        
    elif args.mode == 'full':
        print("🎯 전체 실험 실행 방법:")
        print("1. Jupyter 노트북 시작: jupyter notebook mmimdb_test.ipynb")
        print("2. 모든 셀을 순서대로 실행") 
        print("3. run_full_experiments() 함수 호출")
        print("⚠️  주의: 전체 실험은 GPU가 권장되며 몇 시간이 소요될 수 있습니다.")
        
    elif args.mode == 'xai':
        print("🔍 설명가능성 분석 실행 방법:")
        print("1. Jupyter 노트북 시작: jupyter notebook mmimdb_test.ipynb")
        print("2. 모든 셀을 순서대로 실행")
        print("3. run_xai_demo() 함수 호출")
        print("💡 Grad-CAM 및 Attention Map 시각화가 생성됩니다.")
        
    elif args.mode == 'benchmark':
        print("⚡ 성능 벤치마크 실행 방법:")
        print("1. Jupyter 노트북 시작: jupyter notebook mmimdb_test.ipynb")
        print("2. 모든 셀을 순서대로 실행")
        print("3. benchmark_models() 함수 호출")
        print("📊 추론 속도 및 메모리 사용량이 측정됩니다.")
    
    print("\n📋 추가 정보:")
    print("- 프로젝트 문서: README.md")
    print("- 데이터셋: MM-IMDb (자동 다운로드)")
    print("- 지원 모델: BERT, ResNet50, Cross-Attention Fusion 등")
    print("- GPU 메모리: 최소 8GB 권장")
    
    # 환경 확인
    try:
        import torch
        print(f"\n🔧 환경 정보:")
        print(f"- PyTorch: {torch.__version__}")
        print(f"- CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"- GPU: {torch.cuda.get_device_name(0)}")
            print(f"- GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    except ImportError:
        print("\n⚠️  PyTorch가 설치되지 않았습니다. requirements.txt를 확인하세요.")
    
    print("\n🎉 데모를 시작하려면 위의 안내를 따라주세요!")

if __name__ == "__main__":
    main()
