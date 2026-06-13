# Titanic Feature Engineering Pipeline 과제

## 과제 주제
Titanic Dataset을 활용한 생존 여부 예측을 위한 Feature Engineering 파이프라인 비교 분석

## 데이터셋
- Dataset: Titanic - Machine Learning from Disaster 형식의 train.csv
- Target: `Survived`
- Rows: 891
- Columns: 12
- Problem Type: Binary Classification

## 실험 구성
| Experiment | Missing Value | Encoding | Scaling | Feature Selection |
|---|---|---|---|---|
| Base | None | None | None | None |
| Exp-1 | Mean | One-Hot | StandardScaler | X |
| Exp-2 | Median | Label/Ordinal | MinMaxScaler | O |
| Exp-3 | Most Frequent | One-Hot | RobustScaler | O |

## 사용 모델
- Logistic Regression
- Random Forest Classifier

## 평가 지표
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## 가산점 요소
- scikit-learn Pipeline 객체 활용
- GridSearchCV 적용
- Random Forest Feature Importance 시각화

## 실행 방법
```bash
pip install -r requirements.txt
python src/pipeline_experiment.py
```

또는 Jupyter Notebook에서 다음 파일을 실행한다.

```text
notebook/titanic_feature_engineering_executed.ipynb
```

## 제출 파일
- `report/titanic_feature_engineering_report.pdf`: 최종 보고서
- `notebook/titanic_feature_engineering_executed.ipynb`: 실행 완료 노트북
- `src/pipeline_experiment.py`: 재현 가능한 소스코드
- `results/*.csv`: 성능 비교표 및 분석 결과
- `figures/*.png`: EDA 및 Feature Importance 시각화
