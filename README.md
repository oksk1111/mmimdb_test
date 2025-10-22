# MM-IMDb 멀티모달 영화 장르 예측 모델 개발 및 비교 실험

## 📑 프로젝트 개요

MM-IMDb 데이터셋을 활용한 멀티모달 융합 기반 영화 장르 예측 모델 개발 및 성능 비교 실험 프로젝트입니다.

### 🎯 연구 목표
- **주제**: 멀티모달 융합 기반 영화 장르 예측 모델 개발
- **데이터셋**: MM-IMDb (25,000편 영화, 포스터 이미지 + 줄거리 텍스트 + 23개 멀티라벨 장르)
- **핵심 기여**: Cross-Attention 기반 융합 모델 제안으로 이미지·텍스트 간 상호작용 정교화

## 🔬 실험 구성

### 비교 모델군
1. **텍스트 단일 모달**: BERT, RoBERTa
2. **이미지 단일 모달**: ResNet50, Vision Transformer (ViT)
3. **객체 탐지 기반**: YOLO Feature Extractor, Faster R-CNN Feature Extractor
4. **멀티모달 융합**: Early Fusion, Late Fusion, Attention Fusion
5. **제안 모델**: Cross-Attention Fusion
6. **고급 융합**: Object Detection Fusion (객체 탐지 특징 통합), GMU (Gated Multimodal Unit)

### 평가 지표
- Accuracy, Precision, Recall, F1-score, ROC-AUC, mAP

### 설명가능성 (XAI)
- Grad-CAM (이미지), Attention Map (텍스트)

## 🚀 사용 방법

### 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv mmimdb_env
# Linux/Mac
source mmimdb_env/bin/activate
# Windows
mmimdb_env\Scripts\activate

# 필수 라이브러리 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install torch torchvision transformers timm
pip install pandas numpy matplotlib seaborn scikit-learn
pip install pillow opencv-python tqdm
pip install jupyter notebook
pip install ultralytics  # YOLO 모델용 (선택사항)
```

### 데이터 준비
1. MM-IMDb 데이터셋 다운로드
2. `data/mmimdb/` 경로에 데이터 배치
3. 노트북의 경로 설정 확인

### 실험 실행
```bash
jupyter notebook mmimdb_test.ipynb
```

## ⚡ 빠른 시작

### 1단계: 환경 설정
```bash
# 저장소 복제
git clone <repository-url>
cd mmimdb_test

# 가상환경 설정
python -m venv mmimdb_env
source mmimdb_env/bin/activate  # Linux/Mac
# 또는 mmimdb_env\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2단계: 노트북 실행
```bash
jupyter notebook mmimdb_test.ipynb
```

### 3단계: 실험 실행 옵션

**빠른 데모 (추천)**:
```python
# 노트북에서 실행
run_quick_demo()  # 3 에포크, BERT 모델만
```

**전체 실험**:
```python
# 노트북에서 실행
run_full_experiments()  # 10 에포크, 모든 모델
```

**설명가능성 분석**:
```python
# 노트북에서 실행
run_xai_demo()  # Grad-CAM 및 Attention Map
```

**성능 벤치마크**:
```python
# 노트북에서 실행
benchmark_models()  # 추론 속도 및 메모리 사용량
```

## 📁 프로젝트 구조
```
mmimdb_test/
├── mmimdb_test.ipynb          # 메인 실험 노트북
├── demo.py                    # 데모 실행 스크립트
├── README.md                  # 프로젝트 설명서
├── requirements.txt           # 필요 라이브러리 목록
├── .gitignore                # Git 제외 파일 목록
├── data/                     # 데이터셋 저장 경로 (제외됨)
│   └── mmimdb/              # MM-IMDb 데이터셋
├── docs/                     # 문서 및 계획서
│   └── 논문계획서_요약.txt    # 연구 계획 요약
├── downloads/               # 다운로드 임시 파일 (제외됨)
├── results/                 # 실험 결과 (제외됨)
├── saved_models/           # 저장된 모델 (제외됨)
└── mmimdb_env/             # 가상환경 (제외됨)
```

## 📊 주요 기능

### 1. 데이터 처리
- MM-IMDb 데이터셋 로딩 및 전처리
- 이미지/텍스트 데이터 증강
- 70%/15%/15% 훈련/검증/테스트 분할

### 2. 모델 구현
- **단일 모달리티 모델**: BERT, RoBERTa, ResNet50, ViT
- **객체 탐지 기반 모델**: YOLO Feature Extractor, Faster R-CNN Feature Extractor
- **멀티모달 융합 모델**: Early/Late/Attention/Cross-Attention Fusion
- **고급 융합 모델**: Object Detection Fusion (객체 탐지 특징 통합)
- 체계적인 하이퍼파라미터 설정

### 3. 실험 및 평가
- 자동화된 모델 학습 및 검증
- 종합적인 성능 지표 계산
- 시각화 및 결과 분석

### 4. 설명가능성 분석
- Grad-CAM을 통한 이미지 중요 영역 시각화
- Attention Map을 통한 텍스트 토큰 중요도 분석

## 🔬 연구 방법론

### 데이터 전처리
- **이미지**: 224×224 리사이즈, 정규화, 데이터 증강
- **텍스트**: BERT 토크나이저, 최대 512 토큰

### 융합 전략
- **Early Fusion**: 특징 단계에서 결합
- **Late Fusion**: 예측 결과 결합  
- **Attention Fusion**: 어텐션 가중치 적용
- **Cross-Attention Fusion**: 모달리티 간 교차 어텐션 (제안)
- **GMU (Gated Multimodal Unit)**: 게이트 메커니즘으로 모달리티 간 정보 흐름 제어

### 학습 설정
- **손실 함수**: Binary Cross-Entropy with Logits
- **옵티마이저**: AdamW + Learning Rate Warm-up
- **평가**: 멀티라벨 분류 지표

## 🎯 모델 세부 사항

### 단일 모달리티 모델
- **BERT Classifier**: BERT-base-uncased 기반 텍스트 분류
- **RoBERTa Classifier**: RoBERTa-base 기반 텍스트 분류
- **ResNet50 Classifier**: 사전훈련된 ResNet50 기반 이미지 분류
- **ViT Classifier**: Vision Transformer 기반 이미지 분류

### 객체 탐지 기반 모델
- **YOLO Feature Extractor**: YOLOv5 백본을 활용한 특징 추출 및 분류
- **Faster R-CNN Feature Extractor**: Faster R-CNN 백본을 활용한 특징 추출 및 분류
- 객체 탐지 모델의 백본 네트워크만 활용하여 이미지 분류에 적용

### 멀티모달 융합 모델
- **Early Fusion**: 특징 레벨에서 텍스트와 이미지 특징을 직접 결합
- **Late Fusion**: 각 모달리티별로 독립적 학습 후 예측 결과를 가중 평균
- **Attention Fusion**: 어텐션 메커니즘으로 모달리티별 중요도 계산
- **Cross-Attention Fusion** (제안): 모달리티 간 교차 어텐션으로 상호작용 학습
- **Object Detection Fusion**: 객체 탐지 특징을 포함한 고급 멀티모달 융합
- **GMU (Gated Multimodal Unit)**: 게이트 메커니즘으로 모달리티 간 정보 흐름을 적응적으로 제어

## 📈 기대 성능

### 예상 성능 순서 (Macro F1-Score 기준)
1. **Cross-Attention Fusion** (제안 모델): 0.75+
2. **GMU (Gated Multimodal Unit)**: 0.73+
3. **Object Detection Fusion**: 0.72+
4. **Attention Fusion**: 0.70+
5. **Late Fusion**: 0.68+
6. **Early Fusion**: 0.66+
7. **BERT/RoBERTa** (텍스트 단일): 0.65+
8. **ResNet50/ViT** (이미지 단일): 0.55+
9. **YOLO/Faster R-CNN** (객체 탐지): 0.58+

## 🔧 기술적 특징

### 모델 설계 요소
- **다중 백본 지원**: 다양한 사전훈련 모델 활용
- **견고한 에러 처리**: 모델 로딩 실패시 대체 모델 자동 사용
- **모듈화된 구조**: 각 구성요소의 독립적 개발 및 테스트 가능
- **메모리 효율성**: 배치 처리 및 그래디언트 체크포인팅

### 실험 관리
- **자동화된 실험**: 모든 모델의 일괄 학습 및 평가
- **결과 추적**: 학습 곡선, 성능 지표, 모델 가중치 저장
- **시각화**: 성능 비교, 학습 곡선, XAI 분석 결과
- **재현 가능성**: 고정된 시드 및 설정으로 일관된 결과

## 📈 기대 성과
1. Cross-Attention 기반 융합 모델의 성능 우수성 검증
2. 멀티모달 접근법의 단일 모달리티 대비 성능 향상 확인
3. 설명가능성 관점에서의 모델 해석 및 분석 제공

## 🏆 기여점
- Cross-Attention 기반 멀티모달 융합 방법론 제안
- GMU(Gated Multimodal Unit) 기반 적응적 모달리티 융합 구현
- MM-IMDb 데이터셋에서의 체계적인 모델 비교 분석
- 설명가능한 AI 관점에서의 모델 해석 제공
- 객체 탐지 특징을 활용한 고급 멀티모달 융합 모델 구현

## 📚 참고 자료
- MM-IMDb 데이터셋: [Archive.org](https://archive.org/details/mmimdb)
- BERT: [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
- Vision Transformer: [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)
- Cross-Attention: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

## 🤝 기여 방법
1. 이슈 제기: 버그 리포트 또는 기능 요청
2. 풀 리퀘스트: 코드 개선 또는 새 기능 추가
3. 문서화: README 또는 코드 주석 개선
4. 테스트: 다양한 환경에서의 테스트 결과 공유

## 📈 향후 계획
- [ ] 더 많은 객체 탐지 모델 지원 (DETR, RetinaNet 등)
- [ ] 멀티스케일 이미지 특징 추출
- [ ] 텍스트 감정 분석 특징 통합
- [ ] 웹 인터페이스 개발
- [ ] Docker 컨테이너화
- [ ] MLflow를 이용한 실험 추적

## 📞 연락처
- 프로젝트 관리자: [이메일 주소]
- 이슈 트래커: [GitHub Issues URL]
- 위키: [GitHub Wiki URL]

## 📝 라이선스
이 프로젝트는 교육 및 연구 목적으로 작성되었습니다.
MIT License - 자세한 내용은 LICENSE 파일을 참고하세요.
