# KNOW기반 직업 추천 알고리즘 경진대회
21.12.06 ~ 22.01.28<br>
- KNOW에서 제공되는 2017~2020년도에 해당하는 현직자들에 대한 설문지 정보를 활용하여 직업 추천 알고리즘 개발 및 직업과 연관성이 높은 설문지 문항 분석 및 영향변수 발굴이 목적<br>

### Data 개요
- 개별 항목
know_2017 : 일반업무활동<br>
know_2018 : 업무환경 및 흥미<br>
know_2019 : 지식 및 성격<br>
know_2020 : 업무수행능력 및 가치관<br>
- 공통 항목
직업군, 자격, 연봉, 성별, 나이 등<br>

### EDA (20일 정도 소요)
- 2018, 2019년도 데이터 중 column이 밀려서 입력되어 있는 데이터가 많아서 column의 data 범위를 벗어난 column들을 기준으로 다시 맞춤
- 텍스트로 구성되어 있는 column 중 category 가능한 column으로 자격증, 전공에 관련한 column을 선정하고 category화를 하기 위하여 이름을 통일 시켜줌 (KNOW_0000_train_scale.csv, KNOW_0000_test_scale.csv로 각각 저장)

### 분석 (Version 1, 5일정도 소요)
- 년도별 데이터를 각각 XGBoostClassifier, RandomForestClassifier, DecisionTreeClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, SVC를 사용하여 accuracy 스코어를 기준으로 성능을 측정하여 상위 5개의 모델은 선택해서 그 중 개별 성능 중 제일 높았던 XGBoost 모델과 Soft Voting Ensemble을 활용한 모델 두 가지를 제출
- 결과 : (public score_xgb = 0.49), (pubic score_voting = 0.43)
- 생각보다 점수가 너무 안나와서 EDA를 다시 실행

### 2번째 EDA (5일 정도 소요)
- label의 수가 너무 많아 label별 data 수를 확인해본 결과 평균 10개 제일 적은 경우 1개의 data가 분포하여 label별 data가 너무 부족하다고 판단하여 4개년도의 데이터를 합쳐서 분석하는 방향이 성능을 올릴 수 있을 것으로 생각 (6000개 가량의 label, 년도별 10000개 data)
- 공통 항목을 합치기 위해 다른 이름으로 지정되어 있는 column들을 같은 이름으로 통일 (know_total.csv로 저장)

### 분석 (Version 2, 10일 정도 소요)
- 4개년도의 데이터중 공통인 column만을 concat을 하여 Version 1의 과정을 다시 진행 (XGBoost 모델 제출)
- 4개년도의 데이터를 전부 concat해서 4개년도 각각에 해당하는 column만을 추출하여 Version 1의 과정을 다시 진행 (XGBoost 모델 제출)
- 결과 : (public score = 0.53)
- 4개년도의 data를 합쳐서 학습을 진행하다보니 시간이 너무 오래걸려 model 테스트가 어려워 많은 시도를 못해봄

### 추가
- cross validation을 통한 모델 학습 후 voting을 진행해 보고 싶었는데 cross validation 학습을 진행하는데 시간이 너무 오래 걸려 진행 못함
- 텍스트 column 중 중요한 column은 키워드를 자연어 모델을 통해서 추출하여 진행하면 성능이 오를 것으로 보임

### 마치며
- 처음 참여했던 이유는 데이터로 취준생들에게 직업추천을 해주고, 심리적 요인을 심리학과인 팀원과 feature engineering을 진행해서 추가하면 좋은 방향의 직업 추천으로 이어질 것으로 기대를 하였는데, 실상은 현직자의 설문조사를 통해서 현직자의 직업을 맞추는 직업 맞추기 task로 바뀌어서 아쉬움이 남음.
- 텍스트 category하는 것에 시간을 너무 많이 소요했고, 본인의 분석방법이 잘못된건지 데이터가 너무 컸던건지 분석시간이 너무 오래 걸려서 어려 시도를 못해본 것이 아쉽다.
- 첫 데이콘 출전이었는데 성적이 너무 안나와서 아쉽게 끝남