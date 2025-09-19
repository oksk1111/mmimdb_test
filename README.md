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
3. **객체 탐지 기반**: YOLO, Faster R-CNN
4. **멀티모달 융합**: Early Fusion, Late Fusion, Attention Fusion
5. **제안 모델**: Cross-Attention Fusion

### 평가 지표
- Accuracy, Precision, Recall, F1-score, ROC-AUC, mAP

### 설명가능성 (XAI)
- Grad-CAM (이미지), Attention Map (텍스트)

## 🚀 사용 방법

### 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv mmimdb_env
mmimdb_env\Scripts\activate

# 필요 라이브러리 설치
pip install torch torchvision transformers timm
pip install pandas numpy matplotlib seaborn scikit-learn
pip install pillow opencv-python tqdm
pip install jupyter notebook
```

### 데이터 준비
1. MM-IMDb 데이터셋 다운로드
2. `data/mmimdb/` 경로에 데이터 배치
3. 노트북의 경로 설정 확인

### 실험 실행
```bash
jupyter notebook mmimdb_test.ipynb
```

## 📁 프로젝트 구조
```
mmimdb_test/
├── mmimdb_test.ipynb          # 메인 실험 노트북
├── README.md                  # 프로젝트 설명서
├── requirements.txt           # 필요 라이브러리 목록
├── .gitignore                # Git 제외 파일 목록
├── mcp.json                  # MCP 설정 파일
├── data/                     # 데이터셋 저장 경로 (제외됨)
├── docs/                     # 문서 및 계획서
└── mmimdb_env/              # 가상환경 (제외됨)
```

## 📊 주요 기능

### 1. 데이터 처리
- MM-IMDb 데이터셋 로딩 및 전처리
- 이미지/텍스트 데이터 증강
- 70%/15%/15% 훈련/검증/테스트 분할

### 2. 모델 구현
- 단일 모달리티 모델 (BERT, RoBERTa, ResNet50, ViT)
- 멀티모달 융합 모델 (Early/Late/Attention/Cross-Attention)
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

### 학습 설정
- **손실 함수**: Binary Cross-Entropy with Logits
- **옵티마이저**: AdamW + Learning Rate Warm-up
- **평가**: 멀티라벨 분류 지표

## 📈 기대 성과
1. Cross-Attention 기반 융합 모델의 성능 우수성 검증
2. 멀티모달 접근법의 단일 모달리티 대비 성능 향상 확인
3. 설명가능성 관점에서의 모델 해석 및 분석 제공

## 🏆 기여점
- Cross-Attention 기반 멀티모달 융합 방법론 제안
- MM-IMDb 데이터셋에서의 체계적인 모델 비교 분석
- 설명가능한 AI 관점에서의 모델 해석 제공

## 📝 라이선스
이 프로젝트는 교육 및 연구 목적으로 작성되었습니다.
