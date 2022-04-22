# 🥈HappyWhale - Dolphin and Whale

# Contents

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🐬Task Description](https://github.com/thsckdduq/my_project/kaggle/happywhale#task-description-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🏆Project Result](https://github.com/thsckdduq/my_project/kaggle/happywhale#project-result-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🧐Collaborate Tools](https://github.com/thsckdduq/my_project/kaggle/happywhale#collaborate-tools-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🧐My Experiment](https://github.com/thsckdduq/my_project/kaggle/happywhale#my-experiment-1)**
<br>

# Task Description

### Subject https://www.kaggle.com/competitions/happy-whale-and-dolphin 
<br>
이번 대회의 주제는 돌고래와 고래의 사진으로 individual_id 분류하는 문제였습니다.  돌고래와 고래의 지느러미에 사람의 지문과 같이 각 개체를 분류할 수 있는 특징이 있다고 생각해 해당 문제를 object recognition 이론으로 접근하였습니다.

<br>
돌고래의 사진을 지느러미와 지느러미를 포함한 몸통으로 object detection 진행하고, 해당 이미지를 가지고 각 개체를 분류하는 모델을 생성하였습니다.
<br>

### Data

- 훈련 데이터 : 51033장의 이미지와 해당 이미지의 종과 individual_id

- 테스트 데이터 : 27916장의 이미지
<br>

### Metric

<img src="./img/metric.png" width='400px'/>
<br>

<img src="./img/metric_score.png" width='300px' height='180px' />
<br>

# Project Result

<div><img src=./img/rank.png?raw=true /></div>

- 은메달 47 등 / 1,613 팀

- Public LB Score: 0.85147 / Private LB Score: 0.81686

- Code : https://github.com/YDdreammaker/dl_whale_classification

- 솔루션은 [이곳](https://www.notion.so/Solution-c1be44608fc941bd9442495587a8f1e1)에서 확인하실 수 있습니다.
<br>

# Collaboration Tools
<table>
    <tr height="200px">
        <td align="center" width="350px">	
            <a href="https://www.notion.so/b47246b96c204ca38f96c45888919525?v=f2ab615cde7342c78d3761641a828e5c"><img height="180px" width="320px" src="./img/notion.png?raw=true"/></a>
            <br/>
            <a href="https://www.notion.so/b47246b96c204ca38f96c45888919525?v=f2ab615cde7342c78d3761641a828e5c">Notion</a>
        </td>
        <td align="center" width="350px">	
            <a><img height="180px" width="320px" src="./img/wandb.png?raw=true"/></a>
            <br />
            <a>WanDB</a>
        </td>
    </tr>
</table>

# My Experiment

- Annotation 작업을 통한 Yolo를 돌리기 위한 데이터 생성
- Inference Code 작성 (기존 코드 20분 가량 소요 -> 7분 소요되는 코드 구현)
    - 기존의 pandas로 이루어져 있던 코드를 numpy로 수정하여 소요 시간 단축
    - embedding 값으로 ensemble을 하기위한 코드 (Arcface Embedding값을 받아서 진행)
    - logit 값으로 ensemble 하기위한 코드 (ArcFace Cosine값을 받아서 진행)
- 종을 분류하기 위한 Model 실험
    - 해당 Task에서 Species를 통한 정답의 도출이 중요한 키워드라 생각하고 Species를 분류하는 모델을 위한 실험
    - Species를 분류하는 Model을 생성 후 Inference 과정에서 Species를 통한 Masking 후 Inference 하는 코드 작성 - Model이 정교화 되는 과정에서 Species를 통한 Masking이 효과가 없어짐
    - 마지막 Inference 과정에서 종별로 분포가 상이한 것을 확인하고 종별로 new_individual을 정하는 Threshold를 구하는 과정에 사용됨
- 종별로 label smoothing
    - 모델을 학습하는 과정에서 Species 별로 다양한 feature를 구분할 수 있으면 좋을 것 같다는 생각으로 Label Smoothing을 Species 별로 적용하여 학습 - 실제로 확인해본 결과 기존 ArcFace 모델과 학습한 내용이 매우 상이했음
    - 본래의 ArcFace Model과 Ensemble을 진행한 결과 성능이 소폭 상승
- Global Feature와 Local Feature를 학습하기 위한 방법 구현
