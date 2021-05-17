# Capstone1

깃허브 : https://github.com/hyunobae/Capstone_project1/tree/kihyun

### 1. HOG(Histogram of Oriented Gradients)

#### Gradient(구배)란?

- 공간에 대한 기울기
- 그냥 기울기, slope라고 생각하면 됨.
- 모든 방향에 대해 다 미분을 해야한다.
- ![img](https://t1.daumcdn.net/cfile/tistory/241B3347553343691E)
  - 위 예시는 3차원(x,y,z)이고 x,y,z 방향으로 미분해준 모습.
- 그럼 gradient를 구하면 뭐함?!
  - gradient는 기울기이고, gradient(기울기)의 방향은 바로 함수값이 y가 커지거나 작아지는 방향을 알 수 있다.
  - 이것은 최대점을 찾는 데 사용할 수 있다.

#### Feature Descriptor(형상 기술자)란?

- 유용한 정보를 추출하고 관계가 없는 정보를 버림으로써 **이미지를 단순화하는 이미지 또는 이미지 패치를 표현**한 것.
- 크기 폭* 높이 * 3(채널) 의 이미지를 형상 벡터/길이 n 의 배열로 변환.
  - ex> HOG 형상 기술자의 경우 입력 이미지의 크기는 `64*128*3`이고, 출력 형상 벡터의 길이는 3780이다.
- HOG feature descriptor에서, gradients의 방향의 분포(히스토그램들)가 특성으로 사용됨.



#### HOG(Histogram of Oriented Gradient)

- 일반적으로 보행자 검출이나 사람의 형태에 대한 검출 즉, Object Tracking에 많이 사용되는 Feature 중 하나.
- Image의 지역적인 Gradient를 해당 영상의 특징으로 사용하는 방법.

- HOG feature를 계산하는 순서
  - 영상에서 Gradient 계산
  - Gradient를 이용하여 Local Histogram 새엇ㅇ
  - Local Histogram을 이어붙여 1D vector 생성(HOG Feature)
- 영상은 기본적으로 pixel들의 집합
  - cell : 이런 pixel들을 몇개 묶어 소그룹을 만듦.
  - block : 이런 셀들을 몇개 묶음.
  - <img src="https://t1.daumcdn.net/cfile/tistory/265A064354880D2A24" alt="img" style="zoom:50%;" />
- 영상에서 gradient를 계산해야 함.
  - gradient를 계산하기위해 먼저 영상에서 edge를 계산
  - edge를 검출하기위해 아래와 같은 심플한 1D kernel 사용.
    - ![img](https://t1.daumcdn.net/cfile/tistory/2155B44354880D2B28)
    - => 양옆에 픽셀값을 빼준다.
  - x축 방향 edge와 y축 방향 edge를 계산했으면 이 두 값을 이용하여 orientation을 계산.
    - arctan 함수를 이용하면 두 edge에 의해 발생하는 orientation을 구할 수 있다.
  - 이렇게 구한 orientation을 그냥 다 사용하면 너무 많으니(histogram의 bin이 int 형태로 사용한다 해도 360개나 된다.), 적당히 quantization 한다.(40도 단위로 나누는 9개의 bin, 24도 단위로 나누는 15개 bin 등...)(그러면 orientation map을 만들어 낸거임)
  - 이렇게 구한 orientation map을 바탕으로 histogram으로 만들고 적당히 이어붙이면 HOG Feature가 된다.
- histogram은 cell 단위로 생성하고 얘를 이어붙이는 방식이 block의 움직임에 의해 결정된다.
- 각 cell 별로 histogram을 만드는데 위에서 orientation을 몇개의 bin으로 나눴는지가 histogram의 index가 된다.
- 해당 index에 속하는 pixel의 개수가 histogram의 높이가 된다.
- <img src="https://t1.daumcdn.net/cfile/tistory/2361ED4354880D2D17" alt="img" style="zoom:50%;" />
  <img src="https://t1.daumcdn.net/cfile/tistory/266A4D4354880D2D0C" alt="img" style="zoom:50%;" />
  - 이런식으로 모든 cell에 대하여 histogram 생성
- 이제 이런 histogram을 이어붙이는 방법에 대한 문제.
- block을 움직일때 overlap 하냐 마냐의 경우도 사용자의 선택에 따라 달라짐.
- 일반적으로 overlap하는 경우 아래의 그림과 같이 적당힌 overlap 하면서 shift한다.
- paper에서는 하나의 block 이 16*16 pixel로 이루어져있고, 8pixel 씩 overlap 하면서 shift 한다.
  - <img src="https://t1.daumcdn.net/cfile/tistory/266CC94354880D2B08" alt="img" style="zoom:33%;" />
  - <img src="https://t1.daumcdn.net/cfile/tistory/2161274354880D2C1A" alt="img" style="zoom:33%;" />
  - <img src="https://t1.daumcdn.net/cfile/tistory/23697E4354880D2D0D" alt="img" style="zoom:33%;" />



### 2. 왜 dlib를 사용했는가?

저희 프로젝트에서는 라즈베리파이에서 face detection을 한 프레임을 서버 컴퓨터로 가져와 해당 프레임을 바탕으로 모르는 얼굴을 분류해야 했습니다. 이때, 얼굴 인식으로 사용할 후보로 `Haar Cascade`와 `Dlib Face Detector`가 있었고, 성능적인 측면에서 `dlib`가 더 정확하게 face dectection을 하여 사용하게 되었습니다.(찾오보니까 비교대상이 더있다. Haar Cacade, Dlib, SSD, mtcnn, facenet-pytorch)

그러므로, 라즈베리파이에서 dlib을 사용해서 face detection을 수행하여 넘어온 frame을 바탕으로 다시 분류를 해야하기 때문에 저 또한 `dlib`을 사용했으며, `dlib`을 이용하여 얼굴 간의 유사성을 구해 모르는 얼굴의 방문자들을 분류할 수 있었습니다.



### 3. 구현 로직이 뭔가?

`face_recognition` 패키지의 `face_locations()`과 `face_encodings()` 함수는 `dlib`와 `numpy`에 쉽게 접근할 수 있도록 wrapping 해 놓은 함수이다.

#### 아는 사람의 얼굴과 비교

새로 인식된 얼굴을 기존에 아는 사람의 얼굴과 비교하는 알고리즘

1. 새로 얼굴이 인식되면, 아는 사람들의 `face_encoding`과의 거리를 구한다.
   - 이때 거리를 구할때 `Euclidean distance`를 사용함

2. 가장 가까운 사람과의 거리가 `similarity_threshold`보다 작다면 그 사람의 얼굴이라고 판단하고, 그 사람의 얼굴 DB에 얼굴에 추가한다.

3. 새로 추가된 얼굴을 포함하여 그 사람의 `face_encoding`을 업데이트 한다.(이때, 얼굴의 `face_encoding`의 평균으로 업데이트)

#### 새로운 사람 찾아내기

모르는 사람의 얼굴을 비교하여 새로운 사람을 찾아내는 것 알고리즘

1. 모르는 얼굴은 `unknown_faces`에 따로 저장
2. 새로 인식된 얼굴과 `unknown_faces`과의 거리를 구한다.
3. 가장 가까운 얼굴과의 거리가 `simliarity_threshold`보다 작으면 두 얼굴은 같은 사람의 얼굴이라고 판단하고, 새로운 사람을 만든다.
   - 이때, threshold값은 조사결과 0.5가 적절하다고 했지만, 동양인의 얼굴은 이에 대해서 적절한 성능이 나오지 않아 0.4로 낮추어 진행하니 좋은 분류가 되었다는 사실을 실험적인 결과로 알 수 있었다.
4. 가장 가까운 얼굴과의 거리가 `similarity_threshold`보다 크면 `unknown_faces`에 추가한다.

#### 작품 앞 머무른 시간 계산

각각의 라즈베리파이에서는 실시간으로 서버에 프레임을 보내고 있다. 그리고 각각의 라즈베리파이마다 디렉터리를 만들어서 해당 프레임들을 저장하고 있는 상태이다. 이때, 프레임들의 이름은 프레임이 들어온 순번이다.

이를 이용해서, 각 디렉터리 안에있는 프레임들의 만들어진 시각 정보를 가져와서 프레임들의 이름을 순번에서 `시분초`로 다시 바꾼 뒤, 분류 과정에서 프레임 이미지의 이름을 `시분초`로 바꾼 후 저장했다.

위 과정을 거치면, 각 라즈베리파이가 디렉터리 안에 프레임들은 같은 사람이라고 판단되어있는 프레임들끼리 디렉터리를 만들어서 그 안에 `시분초`의 이름을 가진 상태로 저장되게 된다.

그러면, 이를 통해서 각 프레임의 시각을 알 수 있으며, 초단위로 계산되어 정렬된  프레임들을 비교하면서 작품 앞에 특정 인물의 머무른 시간을 구할 수 있게 된다. 이때, n번째 프레임과 (n+1)번째 프레임의 시각 차이가 10초가 넘어간다면, 이는 해당 사람이 다른 작품에 갔다가 다시 왔다는 작품을 구경하러 왔다는 의미로 판단하고 다시 시간을 구하도록 구현했다.



### 4. dlib 란?

참고 : https://a292run.tistory.com/entry/Face-Recognition-with-Dlib-in-Python-1

`dlib`는 `OpenCV`와 유사하다. `dlib`는 주로 detection와 정렬 모듈을 사용하며, 이를 넘어 즉시 사용가능한 강력한 얼굴인식 `recognition` 모듈도 제공한다. 비록 `C++`로 작성되었지만 `Python` 인터페이스도 가지고 있다.

현대의 `Face recogntion pipline`은 탐지, 정렬, 표현, 검증의 4단계로 구성되며, 모든 단계가 `dlib`의 구현에 포함된다.

`Dlib`은 주로 `ResNet-34 model`에서 영감을 얻었다고한다. 일반적인 `ResNet` 구조를 수정하고 몇몇 레이어를 뺐다. 그리고 29개의 합성곱(convolition) 레이어로 구성된 신경망을 재구성했다. 이 신경망은 input으로 `150*150*3`의 크기를 받고 128차원 벡터로 얼굴 이미지를 표현한다.

그리고, `FaceScrub`과 `VGGFace2`를 포함하는 다양한 데이터셋으로 모델을 재훈련했다. 그리하여 `LFW(Labeled Faces in the Wild)`의 데이터셋으로 테스트했고 99.35%의 정확도를 얻었다.



### 5. Detection(검출)과 Recognition(인식)의 차이

#### Detection

- 얼굴 인식을 예를 들면, 카메라로부터 얼굴이라고 판단되어지는 영역을 찾아서 사각형이나 원 등으로 표시를 해주는 것은 `face detection`

#### Recognition

- 위 처럼 detection 된 face를 A라는 사람의 얼굴이고, B라는 사람의 얼굴이라고 식별(Identification)을 해주는 것이 `face recognition`이다.

즉, `detection`이라는 행위는 관심이 있는 객체와 관심이 없는 객체를 구별하는 것이 목적이고, 일반적으로 식별을 위해 우선적으로 수행되는 과정이라고 할 수 있다.

