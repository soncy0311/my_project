# 35개 도로의 시간별 물류 통행량 예측
- Year_Dream Project (AI Connect 대회)
- 성적 final 점수 공동 8위 / 15팀

## EDA
- train data : 2020.01.01 00시 ~ 2020.05.24 23시
- test data  : 2020.05.17 00시 ~ 2020.05.31 23시 (17부터 23일의 data로 24일부터 31일 예측)
- 2020.02.29, 2020.03.30 data가 누락되어 있어서 우선 해당 날짜를 예측하는 모형을 만들어 예측한 후 진행 (Backcast: 7 days, Forecast: 1 days)
- 2020.02.06 18시, 2020.05.15 5시 data에 이상이 있어 전날과 비교하여 data 수정한 후 진행
- 하루 단위로 data를 불러오도록 Dataset, DataLoader 구축

## RNN, LSTM, GRU
- EDA 결과 1주일 간격으로 주기성을 보이는 것으로 판단하여 7일의 data로 7일의 data를 예측하는 model을 생성
- LSTM의 결과가 가장 높은 것으로 관측됨 Puplic Score (RMSE) : 8642

## N-Beats
- RNN 기반의 분석방법 외에 Linear Model을 통한 분석방법인 N-Beats를 활용해봄
- trend, seasonality, residual Block으로 구성하여 각각의 Block의 잔차를 예측하는 방식으로 진행
- LSTM에 비하여 Public Score는 낮았지만 그림으로 확인해본 결과 좀 더 정확한 것으로 판단
- Public Score (RMSE) : 8861

## 결론
- 시계열 데이터 분석에 N-Beats model을 직접 사용해본 좋은 경험이 되었다.
- N-Beats의 각각 Block의 beta값을 수식적으로 더 잘 이해했다면 편하게 model을 만들지 않았을까 생각한다.