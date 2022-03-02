# 뉴스 기사의 댓글 중 편견, 혐오 표현이 포함된 댓글 식별
- Year_Dream Project (AI Connect 대회)
- 진행중

## Pretrained Model을 통한 학습
- KcElectra : 'beomi/KcELECTRA-base'
- KcBERT : 'beomi/kcbert-large', 'beomi/kcbert-base'
- KoBERT : 'skt/kobert-base-v1'
- RoBERTa : 'klue/roberta-large'
- KcElectra 및 KcBERT가 댓글을 위주로 해서 pretrained된 model이다 보니 좀 더 높은 점수를 보임

## 결론
- Naver에서 개발하는 Hyper Clova pretrained model을 찾지 못해서 사용해보지 못한 것이 아쉬움
- Kakao에서 개발한 KoGPT pretrained model은 GPU 용량 문제로 사용해보지 못한 점이 아쉬움