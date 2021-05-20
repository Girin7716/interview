# 딥러닝

참고 : https://github.com/boostcamp-ai-tech-4/ai-tech-interview/blob/main/answers/deep-learning.md



### AI, 머신러닝, 딥러닝의 차이

AI > 머신러닝 > 딥러닝

`인공지능`

- 인간이 가지고 있는 인식, 판단 등의 지적 능력을 모델링하여 컴퓨터에서 구현하는 것.

`머신러닝`

- 데이터를 기반으로 패턴을 학습하고 결과를 예측하는 알고리즘 기법.
- 조건이 복잡하고 규칙이 다양한 경우에, 데이터를 기반으로 일정한/숨겨진 패턴을 찾아내서 문제를 해결함.
- 단점 : 데이터에 매우 의존적. 즉, 좋은 품질의 데이터를 갖추지 못하면 머신러닝 수행결과도 좋지 않다.
- 분류
  - 지도학습 : 분류, 회귀, 추천시스템, 시각/음성 인지(DL), 텍스트분석(NLP,DL)
  - 비지도학습 : 클러스터링, 차원축소, 강화학습

`딥러닝`

- 여러 층을 가진 `인공신경망(Artificial Neural Network, ANN)`을 사용하여 머신러닝 학습을 수행하는 것으로, 심층학습이라고도 부른다.

`머신러닝과 딥러닝 차이점`

- 기존 머신러닝에서는 학습하려는 데이터의 여러 특징 중 `어떤 특징을 추출할지 사람이 직접 분석하고 판단`
- 딥러닝에서는 기계가 `자동으로 학습하려는 데이터에서 특징을 추출`하여 학습.
- 즉, 특징 추출에 사람이 개입 = 머신러닝 / 개입하지 않는다 = 딥러닝.
- 딥러닝의 데이터 셋, 학습시간 > 머신러닝의 데이터 셋, 학습시간
- 정형데이터 -> 머신러닝, 비정형데이터 -> 딥러닝



### Cost Function과 Activation Function이란?

`cost function`

- 모델은 데이터에 대해 현재 예측을 얼마나 잘하고 있는지 알아야 학습 방향을 어느 방향으로, 얼마나 개선할지 판단할 수 있다.
- 이 때, `예측 값과 데이터 값의 차이에 대한 함수`를 cost function(MSE, CrossEntropy 등) 이라고 한다.

`activation function`

- 데이터를 예측하기 위해 선형 모델 혹은 비선형 모델을 사용할 수 있다.
  - 선형 모델 -> 복잡한 데이터에 대해서 적절한 예측을 못함.
  - 비선형 모델 -> 복잡한 데이터에 대해서 적절한 예측을 함.
- 선형 모델 ----**activation function**(Sigmoid, ReLU 등)----> 비선형 모델
- 선형 모델은 깊게 쌓을 수 없다.(왜냐하면 깊게 쌓아도 하나의 층을 잘 튜닝한 것과 다르지 않기 때문에)
- 비선형 모델은 깊게 쌓을 수 있다.(왜냐하면 선형으로 만들었다가 비선형으로 만드는 작업을 계속 반복할 수 있기 때문에.). 이로 인해 모델은 복잡한 데이터에 대해 더 표현력이 좋아질 수 있다.
- 활성화 함수는 입력 값에 대해 더 높게 혹은 더 낮게 만들 수 있기 때문에 활성화 함수라고 불린다.



### Tensorflow, PyTorch 특징과 차이

| 구분        | Tensorflow         | PyTorch             |
| ----------- | ------------------ | ------------------- |
| 패러다임    | Define and Run     | Define by Run       |
| 그래프 형태 | 정적(Static graph) | 동적(Dynamic graph) |

- Define and RUN
  - 코드를 직접 돌리는 환경인 세션을 만들고, placeholder를 선언하고, 이것으로 계산 그래프를 만들고(Define)
  - 코드를 실행하는 시점에 데이터를 넣어 실행하는(Run) 방식
  - 장점: 계산 그래프를 명확히 보여주면서, 실행시점에 데이터만 바꿔줘도 되는 유연함
  - 단점: 이 자체로 비 직관적.
- Define by Run
  - 선언과 동시에 데이터를 집어넣고 세션도 필요없이 돌리면 되기때문에 코드가 간결하고, 난이도가 낮은 편.



### Data Normalization이란?

`Data Normalizatoin(데이터 정규화)`

- feature들의 분포(scale)를 조절하여 균일하게 만드는 방법
- 필요한 이유
  - 데이터 feature 간 scale 차이가 심하게 날 때, 큰 범위를 가지는 feature(ex>가격)가 작은 범위를 가지는 feature(ex> 나이) 보다 더 강하게 모델에 반영될 수 있기 때문에.
- 즉, data normalization은 모든 데이터 포인트가 동일한 정도의 scale(중요도)로 반영되도록 하는 역할을 수행.
- 장점
  - 학습속도 개선
  - noise가 작아지므로, overfitting 억제
  - 데이터를 덜 치우치게 만드므로, 좋은 성능을 보인다.



`Normalization`, `Standardization`은 모두 데이터의 범위(scale)을 축소하는 방법이다. 데이터의 범위 재조정이 필요한 이유는 `데이터의 범위가 너무 넓은 곳에 퍼져있을 때(scale이 크다면), data set이 outlier를 지나치게 반영`하여 overfitting이 될 가능성이 높기 때문이다.

`Normalization(정규화)`

- Batch Normalization
  - 적용시키려는 레이어의 통계량, 분포를 정규화시키는 방법
- Min-Max normalization
  - 모든 데이터 중에서 가장 작은 값을 0, 가장 큰 값을 1로 두고, 나머지 값들은 비율을 맞춰서 모두 0과 1 사이의 값으로 스케일링하는 방법이다. 모든 feature들의 스케일이 동일하지만, outlier를 잘 처리하지 못한다.

`Standardizatoin(표준화)`

- 표준화 확률변수를 구하는 방법이다. 이는 `z-score를 구하는 방법`을 의미.
- `z-score normalization`이라고도 부르기도 한다.
  - `z-score` : 관측값이 평균 기준으로 얼마나 떨어져있는지 나타낼 때 사용한다. 각 데이터에서 데이터 전체의 평균을 빼고, 이를 표준편차로 나누느 방식이다. outlier를 잘 처리하지만, 정확히 동일한 척도로 정규환 된 데이터를 생성하지 않는다.



### Activation Function 종류

`Sigmoid`

- 입력을 0~1 사이의 값으로 바꿔준다.
- 입력 값이 크거나 작을 때 기울기가 0에 가까워지는 `saturation(포화)` 문제가 있다.
  - 이는 `gradient vanishing` 문제를 야기하므로 요즘에는 activation function으로 잘 사용되지 않는다.
  - `cell saturation(셀 포화)` : 미분하게 거의 0
- 또한 값이 `zero-centered`가 아니기 때문에 입력값의 부호에 그대로 영향을 받으므로 `경사하강법` 과정에서 정확한 방향으로 가지 못하고 지그재그로 움직이는 문제가 있다.

`Tanh (하이퍼탄젠트)`

- 입력을 -1~1 사이의 값으로 바꿔준다.
- `saturation` 문제가 있다.

`ReLU`

- 입력이 양수면 그대로, 음수면 0을 출력
- 계산 효율과 성능에서 뛰어난 성능을 보여 가장 많이 사용되는 activation function.
- 양의 입력에 대해서는 `saturation` 문제가 발생하지 않는다.
- 음의 입력 값에 대해서는 어떤 업데이트도 되지 않는 `Dead ReLU` 문제가 발생한다.

`LeakyReLUd`

- `ReLU`와 마찬가지로 좋은 성능을 유지하면서 음수 입력이 0이 아니게 됨에 따라 `Dead ReLU` 문제를 해결



### Overfitting일 경우 어떻게 대처해야하는가?

`Regularization`

- `Generalization`이 잘 되도록 모델에 제약을 주며 학습을 하여 overfitting을 방지하는 방법
  - Early Stopping
    - training loss는 계속 낮아지더라도 validation loss는 올라가는 시점을 overfitting으로 간주하여 학습을 종료하는 방법
  - Parameter norm penalty(weight decay)
    - 비용함수에 제곱을 더하거나, 절댓값을 더해서 weight의 크기에 페널티를 부과하는 방법
  - Data augmentation
    - 훈련 데이터의 개수가 적을 때, 데이터에 인위적으로 변화를 주어 훈련 데이터의 수를 늘리는 방법
  - Noise robustness
    - 노이즈나 이상치같은 엉뚱한 데이터가 들어와도 견고한(robust) 모델을 만들기 위해 input data나 weight에 일부러 노이즈를 주는 방법
  - Label smoothing
    - 모델이 Ground Truth를 정확하게 예측하지 않아도 되게 만들어 주어 정확하지 않은 학습 데이터셋에 치중되는 경향(overconfident)을 막아주는 방법
  - Dropout
    - 각 계층 마다 일정 비율의 뉴런을 임의로 정해 drop 시키고 나머지 뉴런만 학습하도록 하는 방법
    - 매 학습마다 drop 되는 뉴런이 달라지기 때문에 서로 다른 모델들을 앙상블 하는 것과 같은 효과가 있다.
    - dropout은 **학습 시에만 적용**하고, 추론 시에는 적용x
  - Batch normalizatoin
    - activation function의 활성화 값 또는 출력값을 정규화하는 방법
    - 각 hidden layer에서 정규화를 하면서 입력분포가 일정하게 되고, 이에 따라 Learning rate을 크게 설정해도 괜찮아진다 결과적으로 학습속도가 빨라지는 효과가 있다.



### 하이퍼 파라미터란?

- 모델링할 때, **사용자가 직접 세팅해주는 값**

- 정해진 최적의 값이 없으며, 사용자의 선험적 지식을 기반으로 설정한다.

- 하이퍼 파라미튜 튜닝 기법에는 `Manual Search`, `Grid Search`, `Random Search`, `Bayesian Optimization` 등이 있다.

- | 종류                                            | 설명                                                         | 적용 시 고려사항                                           |
  | ----------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
  | 학습률<br />(Learning Rate)                     | gradient의 방향으로 얼마나 빠르게 이동할 것인지 결정하는 변수 | 너무 작으면 학습 속도가 늦고, 너무 크면 학습 불가          |
  | 손실 함수<br />(Cost Function)                  | 입력에 따른 기대 값과 실제 값의 차이를 계산하는 함수         | - 평균 제곱 오차<br />- 교차 엔트로피 오차                 |
  | 정규화 파라미터<br />(Regularization parameter) | Overfitting 문제 회피 위해 L1 또는 L2 정규화 방법 사용       | 사용하는 일반화 변수도 하이퍼 파라미터로 분류              |
  | 미니 배치 크기<br />(Mini-batch Size)           | 배치셋 수행을 위해 전체 학습 데이터를 나누는 크기            | 가용 메모리 크기와 epoch 수행 성능을 고려                  |
  | 훈련 반복 횟수<br />(Training Loop)             | 학습의 조기 종료를 결정하는 변수                             | 학습 효율이 떨어지는 시점을 적절히 판단                    |
  | 은닉층의 뉴런 개수<br />(Hidden Unit)           | 훈련 데이터에 대한 학습 최적화 결정 변수                     | 첫 Hidden Layer의 뉴런 수가 Input Layer보다 큰 것이 효과적 |
  | 가중치 초기화<br />(Weight Initialization)      | 학습 성능에 대한 결정 변수                                   | 모든 초기값이 0일 경우 모든 뉴런이 동일한 결과             |
  |                                                 |                                                              |                                                            |



파라미터 vs 하이퍼 파라미터

- 사용자가 직접 설정하면 하이퍼 파라미터(학습률, 배치 크기, 은닉층의 개수..)
- 모델 혹은 데이터에 의해 결정되면 파라미터(가중치, 편향...)



### 볼츠만 머신이란?

- 가시층(Visible Layer)와 은닉층(Hidden Layer), 총 2개의 층으로 신경망을 구성하는 방법.
- 모든 뉴런이 연결되어 있는 완전 그래프 형태
  - 제한된 볼츠만 머신(RBM)에서는 같은 층의 뉴런들은 연결되어 있지 않은 모양.
- 단층 구조, 확률 모델
- 분류나 선형 회귀 분석 등에 사용될 수 있다.
- DBN(Deep Belief Network)에서는 RBM들을 쌓아올려, 각 볼츠만 머신을 순차적으로 학습시킨다.
- ![img](https://github.com/boostcamp-ai-tech-4/ai-tech-interview/raw/main/images/adc/deep-learning/boltzmann_machine.PNG)



### sigmoid 보다 ReLU를 많이 쓰는 이유는?

- **기울기 소실 문제(Gradient Vanishing)** 때문.
- 기울기는 Chain Rule에 의해 국소적 미분값을 누적 곱 시키는데, sigmoid의 경우 기울기가 항상 0과 1 사이의 값이므로 이 값을 연쇄적 곱하게되면, 결국 0에 수렴할 수 밖에 없다. 반면 ReLU는 값이 양수일 때, 기울기가 1이므로 연쇄 곱이 1보다 작아지는 것을 어느 정도 막아줄 수 있다.
- 다만, ReLU는 값이 음수라면 기울기가 0이기 때문에, 일부 뉴런이 죽을 수 있다는 단점이 존재한다. 이를 보완한 activation function으로 Leaky ReLU가 있다.



### ReLU가 어떻게 곡선 함수를 근사하는가?

Multi-Layer의 activation으로 ReLU를 사용하게 되면 단순히 Linear 한 형태에서 더 나아가 Linear 한 부분 부분의 결합의 합성 함수가 만들어 지게 된다.

이 Linear한 결합 구간 구간을 최종적으로 보았을 때, 최종적으로 Non-linearity 한 성질을 가지게 된다.

학습 과정에서 BackPropagation 할 때에, 이 모델은 데이터에 적합하도록 fitting이 되게 된다.

ReLU의 장점인 gradient가 출력층과 멀리 있는 Layer 까지 전달된다는 성질로 인하여 데이터에 적합하도록 fitting이 잘되게 되고 이것으로 곡선 함수에 근사하도록 만들어 지게 된다.



### ReLU의 문제점

- **Dead Neurons(죽은 뉴런)**
  - ReLU는 결과값이 음수인 경우 모두 0으로 취급하는데, back propagtion시 기울기에 0이 곱해져 해당 부분의 뉴런은 죽고 그 이후의 뉴런 모두 죽게 된다.
  - 이를 해결한 Leaky ReLU는 값이 음수일 때 조금의 음의 기울기를 갖도록 하여 뉴런이 조금이라도 기울기를 갖도록 한다.
  - 또 다른 방법으로는 입력값에 아주 조금의 편향(bias)를 주어 ReLU를 왼쪽으로 조금 이동시키는 방법이다. 이렇게 되면 입력값은 모두 양수이므로 뉴런이 모두 활성화가 되어 뉴런이 죽지 않는다.
- **Bias Shift(편향 이동)**
  - ReLU는 항상 0 이상의 값을 출력하기 때문에, 활성화 값의 평균이 0보다 커 zero-centered하지 않다.
  - 활성화 값이 zero-centered되지 않으면 가중치 업데이트가 동일한 방향으로만 업데이트가 되서 학습 속도가 느려질 수가 있다.
  - 이를 해결하기 위해 Batch Normalization(배치 정규화)을 사용하여 결과값의 평균이 0이 되도록 하거나 zero-centered된 ELU, SeLU와 같은 활성화 함수를 사용한다.



### 학습모드의 설정

- **온라인 모드**
  - 각 데이터에 대해 가중치를 수정
  - N개 데이터 -> N 번의 가중치 수정
  - 오차 감소 속도가 빠르나 불안정적
- **배치 모드**
  - 모든 데이텅 ㅔ대한 오차를 합하여 가중치 수정
  - N개 데이터 -> 1번의 가중치 수정
  - 오차 감소 속도는 느리나 안정적
  - 1 epoch당 한번의 가중치 수정 발생
- **미니 배치 모드**
  - 데이터를 작은 부분집합으로 나누어 적용
  - N개 데이터를 M개의 그룹으로 묶음 -> M번의 가중치 수정
  - 데이터 규모가 큰 경우에 적용



### Model Setup

- 은닉노드의 수
  - 은닉노드 수가 많을수록 표현 가능한 함수가 다양해짐
  - 계산 비용과 일반화 성능을 고려(overfitting 고려)하여 문젱 맞게 적절히 조정
- 초기조건 설정
  - 초기 가중치 : 작은 범위의 실수 값을 랜덤하게 설정
  - 학습률 : 1 보다 작은 값에서 시작하여 조정
- Activation function
  - 은닉노드 : sigmoid, hyper-tangent, ReLU (전부 비선형 함수)
  - 출력노드
    - 목표출력값이 임의의 실수값 (회귀문제) : linear
    - 목표출력값이 클래스 라벨 (분류문제) : sigmoid, softmax



### Error(Loss) function

- **Squared error function**
  - 목표출력값이 연속한 실수값일 때 적절(회귀문제)
- **Cross Entropy error function**
  - 목표출력값이 1 또는 0의 값을 가질때 (분류문제)
  - 출력노드 actionvation function : softmax function을 주로 사용



### 정규화의 필요성

정규화를 안할 경우

- 신경세포에의 입력값이 크면 셀포화(학습이 잘안됨)가 발생할 가능성이 높아짐
- 학습이 어려움



### 편향(bias)는 왜 있는걸까?

편향(bias)는 activation function이 왼쪽 혹은 오른쪽으로 이동한다.

가중치(weight)는 activation function의 가파른 정도 즉, 기울기를 조절하는 반면, 편향(bias)는 **activation function을 움직임으로써 데이터에 더 잘 맞도록 한다.**



### 성능 계산

- 일반화 성능
  - 학습에 사용되지 않은 새로운 데이터에 대한 신경망 출력의 정확도
  - 평가방법 : test data set을 별도로 수집하여 오차계산
- 학습오차와 테스트오차의 관계
  - 학습이 진행되면 학습오차는 계속 감소
  - 테스트 오차는 일정 시점엠서 다시 증가
    - 과다 적합(학습오차에 포함된 noise까지 과대하게 학습 시킨거임.)
  - 즉, 학습오차가 계속 준다고해서 항상 좋은것은 아니다.
    - why? 우리가 원하는 건 학습오차가 아닌 "일반화 오차"이기 때문.



### Gradient Descent

- 어떤 함수의 극소점을 찾기 위해 gradient 반대 방향으로 이동해 가는 방법.
- 딥러닝에서는 Loss function을 최소화시키기 위해 파라미터에 대해 Loss function을 미분하여 그 기울기 값(gradient)을 구하고, 경사가 하강하는 방향으로 파라미터 값을 점진적으로 찾기 위해 사용된다.



### Gradient Descent의 문제점

- 적절한 step size(learning rate)
  - step size가 큰 경우 한 번 이동하는 거리가 커지므로 빠르게 수렴할 수 있다는 장점이 있다.
  - 하지만, step size를 너무 크게 설정해버리면 최소값을 계산하도록 수렴하지 못하고 함수 값이 계속 커지는 방향으로 최적화가 진행될 수 있다.
  - 한편, step size가 너무 작은 경우 발산하지는 않을 수 있지만 최적의 x를 구하는데 소요되는 시간이 오래 걸린다는 단점이 있다.
- local minima 문제
  - gradient descent 알고리즘을 시작하는 위치는 매번 랜덤하기 때문에 어떤 경우에는 local minima에 빠져 계속 헤어나오지 못하는 경우도 생긴다.



### MLP vs Deep Networks

- MLP는 hidden layer = 1, Depp Networks는 hidden layer >= 2
- Deep Networks의 장점
  - layer를 쌓는 것이 표현의 측면에서 더 효율적이다.
    - = 같은 개수의 파라미터로 더 다양한 형태의 함수로 표현하기 쉬워진다.
- Deep Networks의 단점
  - 학습이 느림 -> 층이 깊어지면 epoch의 횟수가 많아져야한다.
  - 일반화 성능이 낮아짐 -> 층이 많으면 표현할 수 있는 표현력이 커져 학습 data에 포함되어있는 noise까지 학습해서 일반화 성능이 낮아진다.



### Deep Learning의 어려움

- Local minima : 학습이 실패하는 경우
  - Gradient Discent의 근본적 문제
  - 오차가 충분히 작지 않은 local minima에서 학습이 멈춤
  - 대안
    - Simulated annealing : 파라미터 수정폭을 처음에 크게, 차차 줄여감
    - Stochastic(=온라인 러닝) gradient descent : 한번에 하나의 샘플만 사용
  - 참고
    - 온라인 러닝 -> 원래 정확한 error를 따라가는 것이 아닌 샘플하나로 error를 결정해서 전체 error와 약간 다른 값이므로 불안정적이다. = 이 불안정성이 local minima를 피해가는 효과 기대 가능.

- 느린 학습 속도의 문제
  - 학습이 느려지는 평평한 구간(plateau(플라토우))이 나타남 
    - **Plateau problem**
    - plateau는 local minimal가 아니다 -> why? -> 그 뒤에 error가 줄어드니까
  - 가중치 수정 폭은 gradient의 크기에 의존
    - gradient 크기가 점점 작아져서 학습이 느려짐
    - 이를, **Vanishing gradient problem**이라 한다.
- Overfitting
  - 



### Plateau의 발생 원인

- 오차 함수에서 plateau를 만드는 **saddle**이 무수히 많이 존재
- **saddle points**
  - 극대/극소가 아닌 **극점**



### Vanishing Gradient

- 출력층으로부터 오차신호가 입력층으로 내려오면서 점점 약해짐.
  - 즉, 출력층에서 back propagation하면서 입력층으로 오차신호가 내려오면서 약해진다.



### 느린 학습의 해결법

- Activation Function 바꾸기
  - activation function의 기울기가 작아지면 gradient가 감소
  - Sigmoid나 tanh대신 기울기가 줄어들지 않는 function 사용
    - ex> ReLU, Softplus, Leaky ReLU, PReLU, etc.
- Weight Initialization
  - `Cell saturation(셀 포화)`이 일어나지 않도록 작은 값으로
  - 각 뉴런들의 weight가 서로 달라지도록 랜덤하게
  - 각 뉴런들의 Activation function과 gradient의 variance가 서로 유사한 값을 가지도록
- Update rule with momentum(관성)
  - 이전의 움직임(momentum, 관성)을 반영
    - 학습속도 저하를 방지하거나 학습의 불안정성을 감소
- Nesterov accelerated gradient(NAG)
  - momentum을 쓰면서 합쳐서 기울기 계산.
- 적절한 학습률과 함께 update rule를 바꾸기(Changing update rule with adpative learning rate)
  - Weight마다 서로 다른 학습률
  - weight가 변화된 크기의 누적합을 활용
  - 변화폭이 큰 weight는 학습률을 감소시킴
  - RMSProp, AdaDelta
  - Adam(Adaptive Momentum), default임
    - RMSProp과 Momentum 방법의 결합
- 2차 미분 정보를 활용해서 Update rule 바꾸기(Changing Update rule using second order information)
  - Error function의 곡률 정보를 활용
  - 각 weight의 update term이 다른 weight의 update term에 영향
- Batch Normalization
  - 학습하는 동안 각 node의 activation function으로 들어가는 입력을 normalization
  - activation function에서의 입력 분포를 항상 일정하게 유지
  - Cell saturation과 covariate shift 현상을 방지
  - 즉, batch normalization을 사용해서 입력값이 강제적으로 유의미한 범위에 존재하도록 normalization을 함.



### Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유

- Local minima 문제가 사실은 고차원의 공간에서는 발생하기 힘든, 매우 희귀한 경우이기 때문이다. 실제 딥러닝 모델에서는 weight가 수도 없이 많으며, 그 수많은 weight가 모두 local minima에 빠져야 weight update가 정지되기 때문에 local minima는 큰 문제가 되지 않는다.
- 고차원 공간에서 모든 축의 방향으로 오목한 형태가 형성될 확률은 거의 0에 가깝다. 따라서, 고차원의 공간에서 대부분의 critical point는 local minima가 아니라 saddle point이다. 그리고, 고차원의 공간에서 설령 local minima가 발생한다 하더라도 이는 global minimum이거안 또는 global minimum과 거의 유사한 수준의 에러 값을 갖는다. 왜냐하면, critical point에 포함된 위로 볼록인 방향 축의 비율이 크면 클수록 높은 에러를 가지기 때문이다.(실험적 결과) local minima는 위로 볼록인 경우가 하나도 없는 경우이기 때문에 결과적으로 매우 낮은 에러를 갖게 될 것이다.
- 용어 정리
  - `critical point` : 일차 미분이 0인 지점 / local minima, global minima, local maxima, global maxima, saddle point
  - `local minimum` : 모든 방향에서 극소값을 만족하는 점
  - `global minimum` : 모든 방향에서 극소값을 만족하는 점 중 가장 값이 작은 점(정답)
  - `saddle point` : 어느 방향에서 보면 극대값이지만 다른 방향에서 보면 극소값이 되는 점



### Gradient Descent가 Local Minima 문제를 피하는 방법은?

**Momentum, NAG(Nesterov Accelerated Gradient), Adagard, Adadelta, RMSProp, Adam**

- `SGD`

  - Stochastic Gradient Descent
  - 하나 혹은 여러개의 데이터를 확인한 후에 어느 방향으로 갈지 정하는 가장 기초적인 방식.

- `Momentum`

  - 관성
  - 이전 gradient의 방향성을 담고있는 `momentum` 인자를 통해 흐르던 방향을 어느 정도 유지시켜 local minima에 빠지지 않게 만든다.
  - 즉, 관성을 이용하여, 학습 속도를 더 빠르게 하고, 변곡점을 잘 넘어갈 수 있도록 해주는 역할을 수행한다.

- `Nesterov Accelerated Gradient(NAG)`

  - momentum과 비슷한 역할을 수행하는 `Lock-ahead gradient` 인자를 포함하여, a 라는 `accumulate gradient`가 gradient를 감소시키는 역할을 한다.
  - momentum과 다른 점은, 미리 한 스텝을 옮겨가본 후에 어느 방향으로 갈지 정한다는 점이다.

- `Adagrad`

  - 뉴럴넷의 파라미터가 많이 바뀌었는지 적게 바뀌었는지 확인하고, 적게 변한건 더 크게 변하게 하고, 크게 변한건 더 작게 변화시키는 방법
  - `sum of gradient squares`(G<sub>t</sub>)를 사용하는데, 이는 gradient가 얼만큼 변했는지를 제곱해서 더하는 것이므로 계속 커진다는 문제가 발생
  - G<sub>t</sub>가 계속 커지면 분모가 점점 무한대에 가까워지게 되어, W 업데이트가 되지 않게 되어, 뒤로 갈수록 학습이 점점 안되는 문제점이 발생.

- `Adadelta`

  - `Exponential Moving Average(EMA)` 를 사용하여, Adagrad의 G<sub>t</sub>가 계속 커지는 현상을 막을 수 있다.
  - EMA는 현재 time step으로부터 `윈도우 사이즈만큼의 파라미터 변화(gradient 제곱의 변화)를 반영`하는 역할
  - 이전의 값을 모두 저장하는 것이 아닌, `이전 변화량에 특정 비율을 곱해 더한 인자`를 따로 두는 방식
  - Adadelta는 learning rate가 없다.

  

  ### 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?

  Gradient Descent 방식에서 local minima에 도달함은 증명되나, global minima에 도달하는 것은 보장되지 않는다.

  또한, 현재 지점이 global minima인지도 알 수 없다.

  딥러닝에서 다루는 문제가 convexity(집합 X에서 점과 점을 잡고 어떤 선을 그엇을 때, 그 선이 포함하는 점조차도 집합 X에 포함)를 만족하지 않기 때문.

  대신, local minima를 찾는다면, 그 지점이 곧 global minima일 가능성이 크다.(왜냐하면 고차원에서는 local minima가 곧 global minima일 가능성이 커서)

  고차원에서 saddle point가 아닌 완전한 local minimum이 발생하는 경우는 희귀해다.

  따라서 모든 방향에서 아래로 볼록인 local minima를 발견한다면, 그 지점이 바로 global minima일 가능성이 높다.

  

  ### Training set과 Test set을 분리하는 이유는?

  모델은 데이터에 대해 예측값을 만들고 정답과 비교하며 업데이트되면서 학습이 된다.

  그런데 학습 데이터에 대해서는 좋은 성능을 낸다 하더라도 본 적 없는 데이터에 대해서는 잘 대응하지 못하는 **overfitting**  문제가 생긴다면 좋은 모델이 아니다.

  이를 막기 위해 학습된 모델이 처음 보는 데이터에도 강건하게 성능을 내는지 판단하기 위한 수단으로 test set을 따로 만든다.

  

  ### Validation Set이 따로 있는 이유는?

  모델을 학습시키고 test data를 통해 모델의 일반화 성능을 파악하고, 다시 모델에 새로운 시도를 하고 test data를 통해 모델의 성능을 파악한다고 가정해보자.

  이 경우, 모델은 결국 test data에도 overfitting이 되어 다시 처음 보는 데이터를 주면 좋은 성능을 보장할 수 없다.

  이 문제를 막기 위해 validation set을 사용한다.

  validation set을 통해 모델의 성능을 평가하고 하이퍼파라미터 등을 수정하는 것이다.

  즉, train data로 모델을 학습시키고, valid data로 학습된 모델의 성능 평가를 하고 더 좋은 방향으로 모델을 수정한다. 그리고 최종적으로 만들어진 모델로 test data를 통해 최종 성능을 평가한다.

  <img src="https://blog.kakaocdn.net/dn/k4UQW/btqLjsEmxTP/UGalfLDhyaw5MF5GD5KjI1/img.png" alt="img" style="zoom:50%;" />

  - 출처 : https://modern-manual.tistory.com/19

결론

- Training Set으로 모델을 만든 뒤, Validatoin Set으로 최종 모델을 선택. 
- 최종 모델의 예상되는 성능을 보기 위해 Test Set을 사용하여 마지막을 ㅗ성능을 평가
- 그 뒤, 실제 사용하기 전에는 쪼개서 사용하였던 Training Set, Validation Set, Test Set을 모두 합쳐 다시 모델을 training 하여 최종 모델을 만든다.



### Test Set이 오염되었다는 말의 뜻은?

Test Data는 한 번도 학습에서 본 적 없는 데이터여야 한다.

그런데 Train Data가 Test Data와 흡사하거나 포함되기까지한다면 Test Data는 더이상 학습된 모델의 성능 평가를 객관적으로 하지 못한다.

이렇듯 Test Dat가 Train Data와 유사하거나 포함된 경우에 Test Set이 오염되었다고 말한다.



### Regularization이란?

모델의 overfitting을 막고 처음 보는 data에도 잘 예측하도록 만드는 방법을 Regularization(일반화)라고 한다.

대표적인 방법으로 dropout, L1-L2 Regularization 등이 있다.



### Batch Normalization의 효과는?

배치 정규화(Batch Normalization)은 학습 시 **미니배치 단위로 입력의 분포가 평균이 0, 분산이 1이 되도록 정규화**한다.

장점

- 기울기 소실/폭발 문제가 해결 -> 큰 학습률을 설정할 수 있어 학습속도가 빨라진다.
- 항상 입력을 정규화시키기 때문에 가중치 초깃값에 크게 의존하지 않아도 된다.
- 자체적인 규제(Regularization)효과가 있어 Dropout이나 Weight Decay와 같은 규제 방법을 사용하지 않아도 된다.



### Dropout의 효과는?

- **설정된 확률 p 만큼 hidden layer에 있는 뉴런을 무작위로 제거하는 방법**
- overfitting을 방지하기 위한 방법 중 하나.(정확히는 출력을 0으로 만들어 더이상의 전파가 되지 않도록 한다.)
- Dropout은 학습 때마다 무작위로 뉴런을 제거하므로 매번 다른 모델을 학습시키는 것으로 해석할 수 있다.
- 그리고 추론 시 출력에 제거 확률 p를 곱함으로써 앙상블 학습에서 여러 모델의 평균을 내는 효과를 얻을 수 있다.



### GAN에서 Generator 쪽에도 BN을 적용해도 될까?

일반적으로 GAN에서는 Generator(생성기)의 Output Layer에만 BN(Batch Normalizatoin)을 적용하지 않는다. 왜냐하면 Generator가 만든 이미지가 BN을 지나면 실제 이미지와는 값의 범위가 달라지기 때문.



### SGD, RMSProp, Adam에 대해서 아는대로 설명해봐

`SGD`

- Loss functoin을 계산할 때, 전체 train set을 사용하는 것을 Batch Gradient Descent라고 한다.
- 그러나 이렇게 계산을 할 경우 한번 step을 내딛을 때 전체 데이터에 대해 Loss function을 계산해야 하므로 너무 많은 계산량이 필요
- 이를 방지하기 위해 보통은 Stochastic Gradient Descent(SGD)라는 방법 사용
- 이 방법에서는 loss function을 계산할 때 전체 데이터(batch) 대신 데이터 한 개 또는 일부 조그마한 데이터의 모음(mini-batch)에 대해서만 loss function을 계산한다.
- **데이터 한 개를** 사용하는 경우 -> Stochastic Gradient Descent(**SGD**)
- **데이터의 일부**(mini-batch)를 사용하는 경우 -> mini-batch Stochastic Gradient Descent(**mini-batch SGD**)
- **일반적으로 통용되는 SGD는 mini-batch SGD이다.**
- 이 방법은 batch gradient descent 보다 다소 부정확할 수 있지만, 훨씬 계산 속도가 빠르기 때문에 같은 시간에 더 많은 step을 갈 수 있으며 여러 번 반복할 경우 보통 batch의 결과와 유사한 결과로 수렴한다.
- 또한, SGD를 사용할 경우 Batch Gradient Descent에서 빠질 local minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성도 있다.

`RMSProp`

- Adagard의 단점을 해결하기 위한 방법
- Adagrad의 식에서 gradient^2을 더해나가면서 구한 G<sub>t</sub> 부분을 합이 아니라 지수평균으로 바꿔서 대체한 방법
- 이렇게 대체할 경우 Adagrad처럼 G<sub>t</sub>가 무한정 커지지는 않으면서 최근 변화량의 변수간 상대적인 크기 차이는 유지할 수 있다.

`Adam`

- Adam(Adptive Moment Estimation)은 RMSProp와 Momentum 방식을 합친 것 같은 알고리즘이다.
- Momentum 방식과 유사하게 지금까지 계산해온 gradient의 지수평균을 저장하며, RMSProp와 유사하게 gradient^2의 지수평균을 저장한다.
- 다만, Adam에서는 m과 v가 처음에는 0으로 초기화되어 있기 때문에 학습의 초반부에서는 m<sub>t</sub>, v<sub>t</sub>가 0에 가깝게 bias 되어 있을 것이라고 판단하여 이를 unbiased 하게 만들어주는 작업을 거친다.



### SGD에서 Stochastic의 의미는?

SGD는 Loss function을 계산할 때 전체 train dataset을 사용하는 Batch Gradient Descent와 다르게 일부 조그마한 데이터의 모음(mini-batch)에 대해서만 loss function을 계산한다.

`Stochastic`은 **mini-batch가 전체 train dataset에서 무작위로 선택**된다는 것을 의미한다.



### 미니배치를 작게 할때의 장단점은?

- 장점
  - 한 iteration의 계산량이 적어지기 때문에 step 당 속도가 빨라진다.
  - 적은 Graphic Ram 으로도 학습이 가능하다.
- 단점
  - 데이터 전체의 경향을 반영하기 힘들다.
  - 업데이트를 항상 좋은 방향으로 하지만은 않는다.



### Momentum의 수식을 적어보기

Momentum 방식은 말 그대로 Gradient Descent를 통해 이동하느 ㄴ과정에 일종의 `관성`을 주는 것.

현재 Gradient를 통해 이동하는 방향과는 별개로, 과거에 이동했던 방식을 기억하면서 그 방향으로 일정 정도를 추가적으로 이동하는 방식.

![img](https://github.com/boostcamp-ai-tech-4/ai-tech-interview/raw/main/images/adc/deep-learning/wVsfex3Z6n.svg)

- v<sub>t</sub> : time step t 에서의 이동 벡터

![img](https://github.com/boostcamp-ai-tech-4/ai-tech-interview/raw/main/images/adc/deep-learning/NEC8eYVWpB.svg)

이때, ![img](https://render.githubusercontent.com/render/math?math=%5Cgamma)

는 얼마나 momentum을 줄 것인지에 대한 momentum term이다.



### Back Propagation coding

```python
# 가중치 매개변수의 gradient를 구함
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)	# x와 형상이 같은 배열을 새엇ㅇ
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val	# 값 복원
        
    return grad

for key in ('W1','b1','W2','b2'):
    network.params[key] -= learning_rate * gard[key]
```



### MNIST 분류기에서 CNN이 MLP 보다 성능이 좋은 이유

Convolution Layer는 receptive field를 통해 이미지의 위치 정보까지 고려할 수 있다는 장점이있다.

반면, MLP는 모드 Fully connected 구조이므로 이미지의 특징을 이해하는데 픽셀마다 위치를 고려할 수 없게 된다.

따라서, MNIST 분류기에서 MLP를 사용하면 CNN을 사용했을 때 보다 성능이 낮다.
