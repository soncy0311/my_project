### 확진률, 회복률에 대한 식품군 상관분석
- Data 출처 : Kaggle
- 내용
<pre><code>
데이터set : 국가별 섭취 식품군/ 비만/ 영양실조/ 코로나 확진률, 회복률, 사망률, 진행률 관련 데이터
분석내용 : 식품군이 확진률과 회복률에 어떤 영향을 끼치는지를 확인하기 위하여 상관분석을 실시
        회복률에 관련하여서는 상관관계가 크게 나타나지 않아 추가적으로 lightGBM, XGBoost를 통한 분류 모델링을 실시
        lightGBM, XGBoost의 classifier을 사용하기 위하여 회복률이 높은 그룹이 0.4정도의 비율이 되도록 회복률이 높은 그룹과 낮은 그룹으로 나누어 진행
분석결과 : 상관분석 결과 확진률과 식품군에 상관관계에 대해서는 어느정도 연관성이 있다고 보여진다.
        Vagetable Product-확진률의 상관계수가 약 -0.6으로 유의미한 음의 상관관계로 어느정도 확진률을 낮추는데 영향을 끼쳤다고 보인다.
        Animal Product-확진률의 상관계수가 약 0.6으로 유의미한 양의 상관관계로 어느정도 확진률을 높이는데 영향을 끼쳤다고 보인다.
         추가적으로 실시한 lightGBM, XGBoost의 결과 회복률-Pulses(씨앗류)의 연관성이 어느정도 있는것으로 보인다.
        하지만 데이터를 시각적으로 확인한 결과 어떤 영향을 미쳤는지 확인할 수 없었으므로 추가적인 인사이트를 도출할 필요가 있다고 보인다.
</code></pre>

### cal_nutrients.py 작성
- Data 출처 : khidi.or.kr 국민영향통계, kosis.kr
- 내용
<pre><code>
데이터set : 성별, 연령별 - 신장의 평균, 표준편차, 체중의 평균, 표준편차/ 성별, 연령별 - 일일 영양소 권장량 평균, 표준편차
계산내용 : 성별, 연령별 신장의 평균, 표준편차를 통해 Z값을 계산하고,
        일일 영양소 권장량의 평균과 표준편차를 활용하여 개인별 필요 영양소를 계산
        (데이터 셋의 크기가 100만개 가량 되어서 표준정규분포를 활용)
        그 후 활동량 별로 BMI 수치를 계산하는 비율을 적용하여 활동량에 따른 필수 영양소의 양을 도출
</code></pre>

## flask의 기본적인 구성
<pre><code>
GET, POST방식을 활용하여 값을 입력받고 출력하는 구조를 구성
한페이지에 form이 2가지로 POST가 2가지 일어날 경우 button의 name으로 분류하여 다른 동작이 일어나도록 구성
flask에서 html로 변수를 전송하는 방법으로 변수들을 전역변수로 지정하여 전체 페이지에서 사용 가능하도록 지정하고
GET 방식으로 페이지에 접근시에 render_template시에 페이지에 필요한 변수를 보내줌
</code></pre>

## food_lst POST to flask by java
<pre><code>
JavaScript에서 list flask로 넘겨받아 작업을 해야했는데 ajax와 js코드를 같이 사용할 수 없는 이슈가 발생하여
JavaScript 변수 값을 list에서 string으로 변환한 후 setAttribute를 활용하여 클릭 이벤트 발생시 해당 버튼의 value에 string값을 입력하여 POST 방식으로 전송
</code></pre>

## AWS 서버 동기화 및 mysql AWS 구동 작업
<pre><code>
https://github.com/thsckdduq/AWS.git 참조
+ mysql root 계정을 통하여 mysql 외부접근이 어려워서 새로운 유저를 생성하여 외부에서 flask를 통하여 mysql에 접근할 수 있도록 설정
개인 ec2 계정을 통하여 테스트를 위한 DB를 생성하여 개인 pc에서 mysql db를 통하여 test 할 수 있는 환경을 구축
</code></pre>

## login 기능
<pre><code>
mysql diet database에 User 테이블을 만들고 schema작성 pymysql, sqlalchemy를 통하여 DB에 연결하여 User 모델을 생성
회원가입 기능 : join 페이지에서 값을 입력받고, 입력받은 id값이 이미 DB에 존재할 경우 fail, 그외의 경우에 DB에
            pass word를 bcrypt 패키지를 활용하여 password를 인코딩, 암호화하여 DB에 저장하고 return redirect(url_for('login'))
로그인 기능 : login 페이지에서 값을 입력받고, 입력받은 id가 DB에 없을경우 아이디 확인문구 출력,
          id가 DB에 있을경우, 입력받은 password의 인코딩 값과 DB에 저장되어있는 password값이 같은지 확인하고 다를시에 password 확인문구 출력
</code></pre>

## js 코드 작성
<pre><code>
event 발생시의 함수를 작성하여 버튼 동작제어
visualization page : 코로나에 좋은 영향을 미치는 영양소, 안좋은 영향을 미치는 영양소를 보여줄때
                    mouseover event 발생시에 canvas 태그에 새로운 차트를 생성
                    mouseout event 발생시에 div의 canvas를 삭제하고 새로운 canvas를 생성
kit page : 섭취한 영양소 분석 차트에서 하나로 합쳐있던 graph를 코로나 대항 영양소/ 필수 영양소의 차트로 분리
          80% 이하로 섭취한 경우 영양소 부족으로 보고 이름앞에 ⛔️ 표시
          120% 이상으로 섭취한 경우 과다 섭취한 것으로 보고 이름앞에 🚨 표시
</code></pre>