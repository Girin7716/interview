# Used_car_prediction-ML<br>(중고차 가격 예측 머신러닝)

[깃허브](https://github.com/Girin7716/used_car_prediction-ML-)

### 1. 회귀분석의 네가지 방법

#### linear regression(선형 회귀)

#### decision tree(의사결정 트리)

- Classification과 Regression이 모두 가능한 supervised learning model 중 하나.
- 스무고개 하듯이 예/아니오 질문을 이어가며 학습함.
- 특정 기준(질문)에 따라 data를 구분하는 모델을 Decision Tree Model이라고 한다.
- 한번의 분기 때마다 변수 영역을 두 개로 구분.
- 지나치게 질문을 많이하면 Overfitting이 된다.
  - Decision Tree에서 아무 파라미터를 주지 않고 모델링하면 Overfitting이 된다.
- 가지치기(Pruning)
  - Overfitting을 막기 위한 전략
  - 트리에 가지가 너무 많으면 Overfitting이라고 본다.
  - 가지치기란 즉, 최대 깊이나 터미널 노드의 최대 개수, 혹은 한 노드가 분할하기 위한 최소 데이터 수를 제한하는 것.
    - min_sample_split : 한 노드에 들어있는 최소 데이터 수를 정해줌.
      - min_sample_split = 10 이라면 한 노드에 10개 데이터가 있다면 그 노드는 더 이상 분기를 하지 않는다.
    - max_depth : 최대 깊이를 지정해줌
      - max_depth = 4이면, 깊이를 4보다 크게 가지를 치지 않는다.
  - 가지치기에는 1) 사전 가지치기와 2) 사후 가지치기가 있지만, sklearn에서는 사전 가지치기만 지원.
- 

#### random forest(랜덤 포레스트)

- **Decision Tree** : 나무(Tree), 나무가 모여 숲(Forest)을 이룬다. 즉, Decision Tree가 모여 Random Forest를 구성. Decision Tree 하나만으로도 머신러닝을 할 수 있지만, Decision Tree의 단점은 훈련 데이터에 Overfitting이 되는 경향이 있다. 여러 개의 Decision Tree를 통해 Random Forest를 만들면 Overfitting이 되는 단점을 해결할 수 있다.
- 원리
  - ex> 건강의 위험도 예측
  - 성별, 키, 몸무게, 지역, 운동량, 흡연유무, 음주 여부,.... 등등 수많은 요소가 필요(feature가 30개라고 가정)
  - 이렇게 수많은 요소(Feature)를 기반으로 건강의 위험도(Label)을 예측한다면 Overfitting이 일어날 것이다.(트리의 가지가 많아질 것이니까)
  - 하지만, 30개의 Feature 중 랜덤으로 5개의 Feature만 선택해서 하나의 Decision Tree를 만들고, 또 30개 중 랜덤으로 5개의 Feature를 선택해서 또 다른 결정 트리를 만들고... 이렇게 계속 반복하여 여러 개의 Decision Tree를 만들 수 있다. Decision Tree 하나마다 예측 값을 내놓고, 여러 Decision Tree들이 내린 예측 값들 중 가장 많이 나온 값을 최종 예측값으로 정한다.
  - 다수결의 원칙을 따르는 것 처럼, 의견을 통합하거나 여러 가지 결과를 합치는 방식을 **앙상블(Ensemble)**이라고 한다.
  - 즉, 하나의 깊이가 깊은 Decision Tree를 만드는 것이 아닌, 여러 개의 작은 Decision Tree를 만드는 것이다.
  - 여러 개의 작은 Decision Tree가 예측한 값들 중 가장 많은 값(분류일 경우,) 혹은 평균값(회귀일 경우)을 최종 예측 값으로 정하는 것.
  - **문제를 풀 때, 한 명의 똑똑한 사람보다 100 명의 평범한 사람이 더 잘 푸는 원리.**

- **Bagging**
  - **Bootstrap Aggregating의 약자**
  - Voting과 달리 동일한 알고리즘으로 여러 분류기를 만들어 voting으로 최종 결정하는 알고리즘
  - 진행 방식
    1. 동일한 알고리즘을 사용하는 일정 수의 분류기 생성
    2. 각각의 분류기는 **Bootstrapping** 방식으로 생성된 Sample Data를 학습
       - Bootstrapping Sampling은 전체 데이터에서 일부 데이터의 중첩을 허용하는 방식
    3. 최종적을 모든 분류기가 Voting을 통해 예측 결정
- **랜덤 포레스트**
  - 랜덤 포레스트는 여러 개의 Decision Tree를 활용한 Bagging 방식의 대표적인 알고리즘
  - 장점
    - Decision tree의 쉽고 직관적인 장점을 그대로 가지고 있음.
    - 앙상블 알고리즘 중 비교적 빠른 수행 속도
    - 다양한 분야에서 좋은 성능
  - 단점
    - 하이퍼 파라미터가 많아 튜닝을 위한 시간이 많이 소요됨.
- 랜덤포레스트 하이퍼 파라미터 튜닝
  - n_estimators
    - Decision tree의 갯수 지정
    - default : 10
    - 무작정 tree 갯수를 늘릴면 성능 좋아지는 것 대비 시간 걸림
  - random_state
    - 지정한 값에 따라 전혀 다른 트리가 만들어져 같은 결과를 만들어야 하면 고정해야함
    - n_estimators로 만드는 트리 개수를 많이 지정하면 random_state 값의 변화에 따른 변동이 적음
  - max_features : 무작위로 선택할 Feature의 개수(일반적으로 default 사용)
    - max_features가 전체 Feature와 개수가 같다면, 전체 Feature 모두를 사용해 Decision Tree를 만든다. bootstrap 파라미터가 False이면 비복원 추출하기 때문에 그냥 전체 Feature를 사용해 Tree를 만듦.
    - 반면, bootstrap = True(default)이면, 전체 Feature에서 복원 추출해서 Tree를 만듦.
    - max_features 값이 크면 -> 랜덤 포레스트의 tree들이 매우 비슷해지고, 가장 두드러진 특성에 맞게 예측함.
    - max_features 값이 작으면 -> 랜덤 포레스트의 tree들이 서로 매우 달라짐. 이는, overfitting이 줄어드는 효과가 있다.

- 참고 : https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-5-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8Random-Forest%EC%99%80-%EC%95%99%EC%83%81%EB%B8%94Ensemble

#### SVM(Support Vector Machine)

- 주어진 data가 어느 카테고리에 속할지 판단하는 **이진 선형 분류 모델**.

- 

  <figure>
      <img src="https://blog.kakaocdn.net/dn/brs4hH/btqwLdV2jMB/ak8lLgNQKU2GaR6U9ZEDsk/img.png" alt="img" style="zoom: 50%;" />
      <figcaption>출처: Udacity</figcaption>
  </figure>

  - 위 그림에서 빨간 X와 파란 O를 구분하는 3개의 선 중 어떤 선이 가장 적절하게 두 데이털르 구분한 것일까? -> 가운데 선이 적절함
  - why? -> **Margin의 최대화**

- **Margin의 최대화**

  - <figure>
        <img src="https://blog.kakaocdn.net/dn/bh56rr/btqwJd3vppD/958jQaBiONdeq2frzmKbV0/img.png" alt="img" style="zoom:50%;" />
        <figcaption>출처: Udacity</figcaption>
    </figure>

  - 

  - Margin : 선과 가장 가까운 양 옆 데이터와의 거리

  - 서포트 벡터(Support Vector) : 선과 가장 가까운 포인트

  - 즉, Margin은 구분하는 선과 서포트 벡터와의 거리

  - Decision Boundary : 두 데이터를 구분하는 선

- **Robustness**

  - <figure>
        <img src="https://blog.kakaocdn.net/dn/Sl4oW/btqwNu9SajD/HrHtJFnhsNC8NbmOwi6Da0/img.png" alt="img" style="zoom:50%;" />
        <figcaption>출처: Udacity</figcaption>
    </figure>

  - 가운데 선이 Margin을 최대화 한다.

    - 왼쪽 선은 파란 서포트 벡터와의 거리는 길지만, 빨간 서포트 벡터와의 거리는 짧다.
    - 오른쪽 선은 빨간 서포트 벡터와의 거리는 길지만, 빨간 서포트 벡터와의 거리는 짧다.
    - 반면, 가운데 선은 빨간 서포트 벡터, 파란 서포트 벡터와의 거리가 같다.

  - 각 서포트 벡터와의 Margin을 최대화하는 방향으로 Decision Boundary를 결정해야 하므로, 가운데 선이 가장 적절한 Decision Boundary가 된다.

  - 이렇게 양 옆 서포트 벡터와의 **Margin을 최대화하면 robustness도 최대화가 된다.**

  - 로버스트(robust, 튼튼)하다 : 아웃라이어(outlier)의 영향을 받지 않는다는 뜻.

  - 왼쪽 선은 로버스트하지 않고, 가운데선은 로버스트하다.

    - 만약 빨간색 포인트에 noise가 생겨 지금보다 더 오른쪽에 빨간 포인트가 찍힌다면, 왼쪽 선은 빨간 포인트와 파란 포인트를 적절히 구분해주지 못한다.

  - 그러므로, Margin을 최대화하면 robustness도 최대화가 된다.

- **데이터의 정확한 분류**

  - <figure>
        <img src="https://blog.kakaocdn.net/dn/OJDbg/btqwMdgFIB2/HWESfwk8hMqcT8oJFHwMdk/img.png" alt="img" style="zoom:50%;" />
        <figcaption>출처: Udacity</figcaption>
    </figure>

  - 아래 선이 Decision Boundary에 적절하다.

  - 비록 윗 선이 Margin이 더 크지만, 윗 선은 데이터를 정확히 구분하지 못하기 때문에

  - SVM은 무작정 Margin을 크게 하는 Decision Boundary를 택하는 것이 아닌, 데이터를 정확히 분류하는 범위를 먼저 찾고, 그 범위 안에서 Margin을 최대화하는 Decision Boundary를 택하는 것.

- **Outlier 처리**

  - <figure>
        <img src="https://blog.kakaocdn.net/dn/bLtAW3/btqwMc29OS6/MVDPdXLrClHYpaJhirlmC0/img.png" alt="img" style="zoom:50%;" />
        <figcaption>출처: Udacity</figcaption>
    </figure>

  - 두 데이터를 정확히 구분하는 직선은 존재하지 않는다.

  - 이럴 땐 SVM이 어느 정도 outlier를 무시하고 최적의 Decision Boundary를 찾는다.

  - 위 그림에서는 빨간 포인트 사이에 섞인 파란 포인트는 outlier로 취급해서 무시하고 Margin을 최대화하는 구분선인 오른쪽 구분선이 적합하다고 판단한다.

- **커널 트릭(Kernel Trick)**

  - <figure>
        <img src="https://blog.kakaocdn.net/dn/dwqbXp/btqwMKL7pbB/L0N80aKnCmk6LvIILfTgEK/img.png" alt="img" style="zoom:50%;" />
        <figcaption>출처: Udacity</figcaption>
    </figure>

  - x,y 축으로 이루어진 좌표계에서는 빨간 포인트와 파란 포인트를 구분할 수 있는 linear line이 없다. 또한, outlier를 무시하고 그릴 수 있지도 않다.

  - 이럴 땐 차원을 바꿔 Decision Boundary를 구할 수 있다.

  - z = x^2 + y^2 이라하고, z와x로 이루어진 좌표계를 그린다. z는 원점으로부터 해당 데이터까지 떨어진 거리의 제곱과 같다. 따라서, 빨간 포인트는 원점으로부터 거리가 짧아 z가 작고, 파란 포인트는 원점으로부터 거리가 길어 z가 크다. z와x로 이루어진 좌표계는 linear line으로 구분할 수 있다. 즉, 선형으로 분류할 수 있게 된다.

  - **저차원 공간(low dimensional space)을 고차원 공간(high dimensional space)으로 매핑해주는 작업을 커널 트릭(Kernel Trick)이라고 한다.**

  - 위 예제처럼 x,y로 이루어진 저차원 공간은 linear separable하지 않다. 이를 Kernel Trick을 이용하여 고차원 공간으로 매핑한다. 고차원 공간에서는 저차원 공간에서 보다 쉽게 linear separable line을 구할 수 있다. 이 line을 다시 저차원 공간으로 매핑하면 non linear separable line이 구해진다. 처음부터 저차원 공간에서 non linear separable line을 구하려면 쉽지 않기 때문에, Kernel Trick을 활용하여 먼저 고차원 공간에서의 linear한 해를 구한 뒤 저차원 공간에서의 non linear한 해를 구한 것.

  - **머신러닝의 중요한 기법 중 하나**

- **sklearn SVM Parameter**

  - C

    - <figure>
          <img src="https://blog.kakaocdn.net/dn/qP3kl/btqwJdiaoO7/w0ax3gdKREWHs2pSCkQrVK/img.png" alt="img" style="zoom:50%;" />
          <figcaption>출처: Udacity</figcaption>
      </figure>

      

    - 초록색 구분선 : C가 큰 Decision boundary, 주황색 구분선 : C가 작은 Decision boundary

    - C가 크면 training 포인트를 정확히 구분

    - 반면 C가 작으면, smooth한 Decision boundary 그려줌.

    - 즉, C가 작다 == Margin이 커진다.

    - C는, training data를 정확히 구분할지 아니면 Decision Boundary를 일반화할지 결정.

    - **결론적으로 C가 크면 Decision Boundary는 더 굴곡지고, C가 작으면 Decision Boundary는 직선에 가깝다.**

  - Gamma(γ)

    - reach : Decision boundary의 굴곡에 영향을 주는 데이터의 범위.
    - Gammar가 작다 = reach가 멀다 = 대부분의 포인트가 Decision boundary에 영향을 준다. 따라서 Decision boundary와 가까이 있는 포인트 하나하나가  Decision boundary에 주는 영향이 상대적으로 작다. 그래서 선이 포인트 하나 때문에 구부러지지 않는다.
    - Gamma가 크다 = reach가 좁다. =  Decisoin boundary와 가까이 있는 포인트들만이 선의 굴곡에 영향을 준다. 멀리 있는 포인트는 영향이 없으므로 선과 가까이 있는 포인트 하나하나의 영향이 상대적으로 크다.
    - 

### Gradient Boosting Regressor VS RandomForestRegressor

참고 : https://biology-statistics-programming.tistory.com/49



