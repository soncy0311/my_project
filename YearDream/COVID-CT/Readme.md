# 폐 CT 사진으로 Covid 결과 추론
- Year_Dream Project (AI Connect 대회)
- 성적 final 점수 공동 5위 / 15팀

## Data Augmentation
- 원본 데이터 수 : 646 장
- grayscale, invert, horizontalflip을 진행
- 이유 : 이미지 확인 결과 흑백파일, 그리고 색상 반전이 되어있는 이미지가 많았다.
        그래서 이미지 파일 색상 invert를 진행, RGB로 인식되어 있으면 invert 과정에서 어려움이 있어서 grayscale 후 invert 진행
        폐의 경우 좌우 대칭으로 존재하므로 왼쪽의 이미지와 오른쪽을 이미지가 바뀌어도 같은 결과를 낼 것으로 생각하고 HorizonFlip을 진행하여 성능 향상
- 결과 : train data 2484, valid data 100 총 2584장으로 model 학습 진행

## ResNet
- ResNet50, ResNet101, VGG11로 학습을 진행, WanDB를 통해 확인한 결과 ResNet50이 가장 높은 결과를 도출해서 Resnet50을 사용
- v11 (l2 Regularization을 적용) : regularization을 적용하지 않고 학습했을때 train data에 너무 overfitting되는 경향을 보여 l2 regularization을 적용. validation score가 높은 반면 public 점수가 낮았다. 
- v13의 경우 : train data에 overfitting되는 경향을 보였지만, validation score가 무난하고, pubic score가 높았다.
- 결과 : v13을 제출. v13이 public score는 높았지만, v11이 final score가 압도적으로 높았다. 역시 validation score가 믿을만하다.