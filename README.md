# 🚗 자동차 보험 사기 탐지 프로젝트

> 통계적 가설 검증과 해석 가능한 머신러닝(Interpretable ML)을 활용한 자동차 보험 사기 패턴 분석 및 탐지

## 📋 프로젝트 개요

자동차 보험 사기는 보험사에 막대한 재정적 손실을 초래하며, 이를 효과적으로 탐지하기 위해서는 단순한 예측 모델을 넘어 **"왜 사기로 의심되는지"를 설명할 수 있는 분석**이 필요함.

본 프로젝트는 Kaggle의 [Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data) 데이터셋(15,420건, 33개 변수)을 활용.

- **통계 검정 기반의 유의미한 변수 선별** (Chi-square, t-test, KS test)
- **비즈니스 가설 수립 및 데이터 기반 검증**
- **Explainable Boosting Machine(EBM)을 활용한 해석 가능한 사기 탐지 모델 구축**
- **Power BI 대시보드 연동을 위한 데이터 파이프라인 구성**

---


## 📊 데이터셋 부가 설명

| 구분 | 변수 | 설명 |
|------|------|------|
| **사고 일시** | Month, WeekOfMonth, DayOfWeek | 사고 발생 시점 |
| **청구 일시** | DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed | 보험 청구 접수 시점 |
| **차량 정보** | Make, VehicleCategory, VehiclePrice, AgeOfVehicle | 차량 제조사, 분류, 가격대, 연식 |
| **개인 정보** | Sex, MaritalStatus, Age, AgeOfPolicyHolder 등 | 피보험자의 인적 사항 |
| **보험 정보** | PolicyType, Deductible, BasePolicy, AgentType 등 | 보험 상품 및 계약 조건 |
| **사고 정보** | AccidentArea, Fault, PoliceReportFiled, WitnessPresent 등 | 사고 상황 및 조사 정보 |
| **타겟** | **FraudFound_P** | **0 = 정상, 1 = 사기** |

- 총 **15,420건** / 결측치 없음
- 사기 비율: 약 **6%** (923건) → 심각한 클래스 불균형

---

## 🔍 분석 파이프라인

### Step 1. 기초 EDA 및 데이터 이해

- 전체 변수의 분포 확인 (Plotly 기반 자동 시각화)
- 수치형 → 히스토그램, 범주형 → 카운트 플롯
- 데이터 타입 및 고유값 수 파악

### Step 2. 통계 검정 기반 변수 선별

변수 타입별로 적합한 통계 검정을 자동 적용하여, **FraudFound_P와 유의미한 관계**가 있는 변수를 선별함.

| 변수 조합 | 검정 방법 | 유의 기준 |
|-----------|-----------|-----------|
| 범주형 vs 범주형 | Chi-square Test | p < 0.05 |
| 수치형 vs 이진형 | t-test + KS Test | 두 검정 모두 p < 0.05 |
| 수치형 vs 다범주형 | ANOVA | p < 0.05 |

**선별된 주요 변수**: Sex, Age, AgeOfPolicyHolder, PastNumberOfClaims, VehiclePrice, AddressChange_Claim, BasePolicy 등

### Step 3. 보험 상품별 분리 분석

보험 상품(BasePolicy)에 따라 사기 패턴이 크게 다르므로, 상품별로 분리하여 분석.

| 상품 | 건수 | 사기 건수 | 사기율 |
|------|------|----------|--------|
| **All Perils** (종합보험) | 4,449 | 452 | **10.2%** |
| **Collision** (충돌보험) | 5,962 | 435 | **7.3%** |
| **Liability** (책임보험) | 5,009 | 36 | **0.7%** |

→ Liability는 사기 건수가 극히 적어 분석 대상에서 제외하고, **All Perils와 Collision**을 중심으로 분석.

### Step 4. 가설 기반 심화 분석

#### 가설 1: 고가 차량일수록 보험 사기 비율이 높다

차량 가격을 3구간(3만 미만 / 3~6만 / 6만 이상)으로 리매핑 후 분석한 결과

- **All Perils**: 비사기 중 고가차 비율 18.2% → 사기 중 고가차 비율 **23.2%** (+5%p)
- **Collision**: 비사기 중 고가차 비율 12.8% → 사기 중 고가차 비율 **19.3%** (+6.5%p)

**결론**: 보험 사기에 사용되는 차량은 상대적으로 고가이며, 이는 높은 보험금이 사기의 인센티브로 작용하는 것으로 해석됨.

#### 가설 2: 주소를 변경한 피보험자의 사기 비율이 높다

주소 변경 여부(change / no_change)로 단순화 후 분석

- **All Perils**: 주소 변경 O → 사기율 **15.5%** / 주소 변경 X → 사기율 **9.8%**
- **Collision**: 주소 변경 O → 사기율 **11.7%** / 주소 변경 X → 사기율 **7.0%**

주소 변경 그룹에서만 유의미한 추가 변수 발견

| 조건 | All Perils 사기율 | Collision 사기율 |
|------|-------------------|------------------|
| 주소 변경 + 차량 1대 | **30%** | **18%** |
| 주소 변경 + 차량 2대 | 9% | 7% |

또한 주소 변경 + 사기 그룹의 평균 자기부담금(Deductible)이 **449**로, 보험사기가 아닌 그룹(425)보다 높은 경향을 보임.

**결론**: 주소 변경 + 차량 1대 + 높은 자기부담금은 보험 사기의 주요 위험 시그널이라고 판단함.

### Step 5. 해석 가능한 머신러닝 모델링 (EBM)

#### 모델 선택: Explainable Boosting Machine (EBM)

- 일반화 가법 모형(GAM) 기반의 Glass-box 모델
- 각 변수의 기여도를 투명하게 설명 가능 (Global & Local Explanation)
- 보험 업무에서 "왜 이 건이 사기로 판단되었는지" 근거를 제시할 수 있음

#### 클래스 불균형 처리: SMOTENC

범주형 변수가 포함된 데이터에 적합한 SMOTENC를 사용하여 사기 클래스를 1,500건으로 오버샘플링하였음.

#### 모델 성능

| 상품 | Precision (사기) | Recall (사기) | F1 (사기) | Accuracy |
|------|------------------|---------------|-----------|----------|
| **All Perils** | 0.58 | 0.21 | 0.31 | 0.74 |
| **Collision** | 0.78 | 0.13 | 0.22 | 0.81 |

- Recall이 낮은 편이지만, **모델이 사기로 판단한 건의 신뢰도(Precision)는 확보**되었다고 판단함.

---

## 💡 핵심 인사이트 요약

1. **보험 상품에 따라 사기 패턴이 다르다** — All Perils(10.2%)가 가장 높고, Liability(0.7%)는 거의 없음
2. **고가 차량**일수록 사기에 활용될 가능성이 높음 — 높은 보험금이 인센티브
3. **주소 변경 이력**이 있는 피보험자의 사기율이 약 1.5~2배 높음
4. 주소 변경 + **차량 1대** + **높은 자기부담금** 조합은 강력한 사기 위험 시그널
5. EBM 모델로 개별 건에 대한 **사기 판단 근거를 투명하게 설명** 가능

---

## 🛠️ 기술 스택

| 구분 | 도구 |
|------|------|
| **언어** | Python |
| **데이터 처리** | Pandas, NumPy |
| **시각화** | Plotly, Seaborn, Matplotlib |
| **통계 검정** | SciPy |
| **머신러닝** | InterpretML (ExplainableBoostingClassifier) |
| **대시보드** | Power BI |

---


## 📈 향후 개선 방향

- **데이터 확보**: 현재 사기 데이터가 923건으로 부족하여 교차 분석의 모수가 제한적 → 사기 탐지 건수 누적 시 더 정밀한 패턴 분석 가능
- **모델 고도화**: Recall 개선을 위한 임계값(threshold) 조정, 앙상블 기법 적용
- **실시간 연동**: 신규 보험 청구 건에 대해 실시간으로 사기 위험도를 산출하는 파이프라인 구축
- **추가 변수 분석**: 시간 변수(사고~청구 간 소요일수), 지역 변수 등과의 교차 분석 확대


## 🧾 대시보드 

[PowerBI 대시보드](https://app.powerbi.com/view?r=eyJrIjoiZWFmNDY4MzktZjljYS00YjE0LTlkNGEtZjYzNGQ0OTUzZGQ4IiwidCI6Ijg4ZDQ0NWU1LWU5YjAtNGNkMy04MTVmLTQwZjhhMzAwOWI0MiIsImMiOjEwfQ%3D%3D) ⬅️ PowerBI 웹게시링크

[원본 다운로드](https://drive.google.com/file/d/1anMZda0bhGbTSkPnaTfwqVuNjhCbwKbh/view?usp=sharing)

[PDF](https://drive.google.com/file/d/1IvF3ayJLQykQXyMKk6SREgMf2D7jMUft/view?usp=sharing)
