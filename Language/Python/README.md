# Python

참고 : https://github.com/boostcamp-ai-tech-4/ai-tech-interview/blob/main/answers/python.md



### 목차

[TOC]

### 파이썬의 주요 특징은?

- **인터프리터 언어**
  - 파이썬은 인터프리터 언어이므로, 실행하기 전에 컴파일을 할 필요가 없다.
  - 인터프리터로 동작하는 **스크립트 언어**이므로 다른 컴파일 언어에 비해 다소 느리다.
    - 컴파일러가 코드를 기계어로 번역해서 실행가능 파일을 만드는 것에 비해, 인터프리터는 코드를 한줄 씩 실행시간마다 번역해서 실행하기 때문이다.
- **동적타이핑(Dynamic Typing)**
  - 파이썬은 실행시간에 자료형을 검사하므로, 선언할 때 변수 유형을 명시할 필요가 없다.
  - `typing`이란 프로그램 내에서 변수의 데이터 타입을 정하는 것을 말한다. 데이터 타입 지정은 정적 또는 동적 타이핑으로 분류되는데, 프로그램 컴파일 시에 변수의 타입을 체크하는 `C`, `C++`과 같은 언어는 정적 타입 언어, 프로그램 실행 시에 타입을 체크하는 `Python`은 동적 타입 언어이다.
- **객체 지향 프로그래밍(OOP)**
  - 클래스와 구성 및 상속을 함께 정의할 수 있따는 점에서 객체 지향 프로그래밍에 매우 적합.
- **일급객체(First-class citizen)**
  - 파이썬에서는 `함수`와 `클래스`는 일급 객체이다. 일급객체는 변수나 데이터 구조 안에 담을 수 있고, 매개변수로 전달이 가능하며, 리턴값으로 사용될 수 있다는 특징을 가지고 있다.



### What type of language is python? Programming or scripting?

파이썬은 정확하게는, `스크립트 언어`이다. 모든 스크립트 언어는 프로그래밍 언어로 볼 수 있으나, 모든 프로그래밍 언어는 스크립트 언어로 분류되는 것은 아니다. 따라서 `파이썬`은 `스크립트 언어`이자, `프로그래밍 언어`이다.

- 스크립팅(scripting/Scripting Language)
  - 스크립트 언어란 컴파일이 필요없이 실행될 수 있는 명렁어의 집합.
  - 스크립트 언어는 인터프리터를 사용하는데, 인터프리터는 컴파일 과정이 필요하지 않으며, 소스코드로 부터 바로 명령어를 해석할 수 있다.



### 인터프리터 언어(=스크립트 언어) 설명

인터프리터는 고급 언어로 작성된 원시코드 명령어들을 한번에 한 줄씩 읽어들여서 실행하는 프로그램이다. 인터프리터 언어는 runtime 전 기계 레벨 코드를 만드는 컴파일 언어와 다르게 소스코드를 바로 실행하는 언어이며, 파이썬은 인터프리터 언어에 해당한다.



### pep 8?

PEP(Python Enhancement Proposal)는 Python 코드를 포맷하는 방법을 지정하는 규칙 집합이다.

다른 사람과 원활하게 협엽하려면 공통된 스타일 공유가 필요하며, 일관성 있는 스타일은 나중에 수정하기도 쉽다. PEP8은 파이썬 코드를 어떻게 구성할 지 알려주는 스타일 가이드로서의 역할을 한다.



### Python에서 메모리 관리는 어떻게 하는가?

Python은 모든 것을 객체로 관리한다.

객체가 더 이상 필요하지 않으면 파이썬 메모리 관리자가 자동으로 객체에서 메모리를 회수하는 방식을 사용하므로, 파이썬은 `동적 메모리 할당` 방식을 사용한다고 말할 수 있다.

`힙(heapq)`은 동적할당을 구현하는데 사용된다. 힙을 사용하여 동적으로 메모리를 관리하면, 필요하지 않은 메모리를 비우고 재사용할 수 있다는 장점이 있다.

모든 파이썬 객체 또는 자료구조는 python private heap 공간에서 관리되며, 프로그래머는 이 공간에 접근할 수 없고, 대신 파이썬 인터프리터가 대신해서 관리한다.

파이썬 객체에 대한 힙 공간 할당은 `파이썬 메모리 관리자(Python Memory Manager)`에 의해 수행된다. 또한, 파이썬은 사용되지 않는 모든 메모리를 재활용하고 힙 공간에서 사용할 수 있도록 하는 `내장 Garbage Collector(GC)`를 가지고 있으며, Python 메모리 관리자에는 객체별 할당자가 있어 int, string 등과 같은 특정 객체에 대해 메모리를 명확하게 할당 할 수 있다.



### Python에서의 local variables과 global variables

- **지역 변수(Local Variance)**
  - 함수 내부에 선언된 변수.
- **전역 변수(Global Variance)**
  - 함수 외부 또는 전역 공간에 선언된 변수.
  - 프로그램의 모든 함수에서 전역변수에 접근할 수 있다.



### Is Python case sensitive?

파이썬은 대소문자를 구분하는 언어임.

- ex> `a`와 `A`는 다른 변수.



### ``__init__` 이 무엇인가?

`__init__`는 파이썬에서 특별하게 약속된 메서드 가운데 하나로, `초기화 메서드` 혹은 `생성자`라고도 한다. 이 메서드는 클래스의 새 개체/인스턴스가 생성될 때 메모리를 할당하기 위해 자동으로 호출되며, 그 객체가 갖게 될 여러 가지 성질을 정해준다. 모든 클래스에는 `__init__` 메서드가 있다.



### lambda function이란?

`람다 함수`는 `익명 함수(이름이 없는 함수)` 라고도 한다.

람다 함수는 `def` 키워드를 통해서 함수를 생성하는 리터럴 표기법을 **딱 한줄의 코드로 표현할** 수 있게 해주며, `lambda 인자 : 표현식`의 형식으로 표현한다.

람다함수는 결과 부분을 return 키워드 없이 자동으로 return 한다.

람다함수를 사용하면 코드가 간결해지고 메모리가 절약된다는 장점이 있다.

그러나, 함수에 이름이 없고, 저장된 변수가 없기 때문에 다시 사용하기 위해서는 다시 코드를 적어주거나, 람다함수를 변수에 담아주어야한다.

람다함수도 객체이기 때문에 정의와 동시에 변수에 담을 수는 있다.

재사용할 이유가 없다면 lambda 함수를 생성하여 넘겨주는 편이 좋다.



### Python에서 self는 무엇인가?

```python
class MyClass:
    def method(self):
        return 'instance method', self

obj = MyClass
print(obj.method())
```

`self`는 instance method의 첫 번째 인자이다. 메서드가 호출될 때, 파이썬은 `self`에 인스턴스를 넣고 이 인스턴스를 참조하여 인스턴스 메서드를 실행할 수 있게 된다.



### What does [::-1] do?

파이썬 sequence 자료형은 값이 연속적으로 이어진 자료형으로, **리스트, 튜플, range, 문자열**이 있다.

Sequence 자료형은 sequence 객체의 일부를 잘라낼 수 있는 **Slicing(슬라이싱)**이라는 기능을 쓸 수 있다.

슬라이싱은 `seq[star:end:step]` 처럼 쓸 수 있으며, `start`는 시작 인덱스, `end`는 끝 인덱스(범위에 포함하지는 않음), `step`은 인덱스 증감폭을 말한다. `step`이 양수이면 증가하고, 음수이면 감소한다.

`seq[::-1]`은 `start`와 `end`는 시작 인덱스와 끝 인덱스를 생략하였는데, 이럴경우 전체 sequence를 가져오며, 증감폭이 -1이므로 `end-1`부터 시작해 `start`순으로 요소를 가져온다.

즉, `seq[::-1]`은 sequence를 역전시킨다.



### iterator와 iterable 차이점

- `iterable 객체` : iter 함수에 인자로 전달 가능한, 반복 가능한 객체
- `iterator 객체` : iter 함수가 생성해서 반환하는 객체
  - next 함수로 Iterable 객체의 요소의 값을 차례대로 반환한다.
  - 만약 Iterable 객체를 모두 돌면 `StopIteration` 예외를 발생시킨다.
  - 다시 돌고 싶다면 iter 함수를 Iterator 객체를 다시 생성해야 한다.

- iterable 객체는 `iter` 함수에 인자로 전달 가능한, 반복 가능한 객체를 말한다.
  - ex> list, dictionary, set, string...
- iterable 객체를 `iter` 함수의 인자로 넣으면 iterable 객체를 순회할 수 있는 객체를 반환하는데, 이것이 iterator 객체이다. iterator 객체를 `next` 함수의 인자로 주면 iterable 객체의 요소의 값을 차례대로 반환한다. 만약 iterable 객체를 모두 순회했다면, `StopIteration` 예외를 발생시킨다. 만약 다시 순회를 하고 싶다면 `iter` 함수로 새로운 iterator 객체를 생성해주면 된다.



### pickling과 unpickling이 무엇인가?

우선 `직렬화(Serialization)` 와 `역 직렬화(Deserialization)`의 개념을 알아야한다.

- `직렬화`란 객체를 byte stream으로 변환하여 디스크에 저장하거나 네트워크로 보낼 수 있도록 만들어주는 것.
- `역 직렬화`란 반대로 byte steram을 파이썬 객체로 변환하는 것.

`pickle 모듈`은 파이썬 객체의 직렬화와 역 직렬화를 수행하는 모듈이다. 이때 파이썬 객체를 직렬화할 때를 `pickling`이라고 하며, 바이트 스트림을 역 직렬화할 때를 `unpickling`이라고 한다.



### Python에서 generators란?

`Generator`란 Iterator 객체를 간단히 만들 수 있는 함수를 말한다.

1) yield문과 함수 2) 표현식 형태로 만들 수 있다.

1. yield문과 함수

   - 제너레이터 함수 정의

   ```python
   def generator_list(value):
       for i in range(value):
           # 값을 반환하고 여기를 기억
           yield i
   ```

   - 제너레이터 객체 생성 및 next 함수로 호출

     ```python
     gen = generator_list(2)
     print(next(gen))	# 0
     print(next(gen))	# 1
     print(next(gen))	# StopIteration 에러 발생
     ```

2. 표현문

   - ```python
     value = 2
     gen = (i for i in range(value))
     print(next(gen))	# 0
     print(next(gen))	# 1
     print(next(gen))	# StopIteration 에러 발생
     ```

왜 리스트 대신 제너레이터를 사용할까?

- 리스트를 사용하면 리스트의 크기만큼 메모리에 공간이 할당.
- 반면, 제너레이터는 말 그대로 next 함수로 호출될 때 값을 생성하고 해당 값만 메모리에 올린다!
  - 즉, 메모리를 절약할 수 있다.
  - 그러므로, 작은 데이터라면 상관없지만, 큰 데이터에서는 제너레이터 사용이 필수이다.



### operators에서 `is`, `not`, `in`의 목적이 무엇인가?

- `is`

  - 객체 비교 연산자

  - 두 변수가 참조하는 객체의 id가 같을 경우 True 반환

  - ```python
    a = [1,2,3]
    b = a
    c = a.copy()
    
    print(a is b)	# True
    print(a is c)	# False
    ```

- `not`

  - 단항 논리 연산자
  - 뒤에 오는 boolean 값을 뒤집는다.

- `in`

  - 멤버 연산자
  - 요소 a와 시퀀스 b가 있는 지를 확인하고 싶을 때, `a in b`로 표현하고 결과로 boolean 값 반환

### 파이썬이 종료될때마다, 모든 메모리가 할당 해제 되지 않는 이유는?

다른 객체나 전역 namespace에서 참조되는 객체를 순환 참조하는 파이썬 모듈은 항상 해제되지는 않는다. 

또한 C 라이브러리가 예약한 메모리의 해당 부분을 해제하는 것은 불가능하다.

그러므로 파이썬 종료 시, 모든 메모리가 해제되지 않는다.



### Python에서 dictionary란?

`Dictionary`란 **key값과 그에 대응하는 value 값을 얻을 수 있는 collection**을 말한다.

데이터가 들어온 순서가 상관이 없고, 인덱싱이 되어 있어 key값으로 요소에 접근하여 value 수정이 가능하다.

하지만, key값은 고유 값이므로 key값 중복은 불가능.

key값을 list로 사용하면 변할 가능성이 있기 때문에 인터프리터에서 type error 발생시킨다.

1. key와 value로 구성된 구조를 가지는 collection으로, key에 따라서 value 값을 조회, 삭제, 수정, 삽입이 가능
2. 이는 hashing을 사용한다는 특징이 있다. 
3. hash table은 hashing을 사용하는데, hashing을 사용한다는 것은 객체가 가지고 있는 hashcode값에 따라서 key 테이블이 위치할 곳을 정한다는 것이다. hashcode 계산으로 해당 데이터가 위치한 곳을 빠르게 찾을 수 있기 때문에 많은 양의 데이터를 검색할 때 유리하다.
   하지만 주의도 필요하다. 이때 만약 다른 key값인데 hashcode 값이 동일한 경우가 있는데 이때는 해쉬 충돌이 일어날 수 있다. 이때는 hashcode를 담고있는 bucket이 연결리스트와 같은 구조로 다음 연결 리스트의 노드에 해당하는 key 값을 넣어주어 값을 저장한다.



### `*args`, `**kwargs`의 의미와 언제 사용하는가?

- `*args`는 함수에 전달되는 argument의 수를 알 수 없구나, list나 tuple의 argument들을 함수에 전달하고자 할 때 사용한다.

  - ex>

    ```python
    def name(*args):
        print(args)
    name("가","나","다","라")	# output: ('가','나','다','라')
    ```

- `**kwargs`는 함수에 전달되는 keyword argument의 수를 모르거나, dictionary의 keyword argument들을 함수에 전달하고자 할 때 사용한다.

  - ex>

    ```python
    def name(**kwargs):
        print(kwargs)
    name(ga="가",na="나",da="다",la="라")	
    # output: {'ga':'가', 'na':'나', 'da':'다', 'la':'라'}
    ```



### packages 와 modules

- `모듈`
  - 파이썬 코드를 논리적으로 묶어서 관리하고 사용할 수 있도록 하는 것
  - 보통 하나의 파이썬 `.py` 파일이 하나의 모듈이 된다.
  - 모듈 안에는 함수, 클래스, 혹은 변수들이 정의될 수 있으며, 실행 코드를 포함할 수 있다.
- `패키지`
  - 특정 기능과 관련된 여러 모듈을 묶은 것
  - 패키지는 모듈에 namespace를 제공
  - 패키지는 하나의 디렉토리에 놓여진 모듈들의 집합을 가리키는데, 그 디렉토리에는 일반적으로 `__init__.py`라는 패키지 초기화 파일이 존재한다.
  - 패키지는 모듈들의 컨테이너로서 패키지 안에는 또다른 서브 패키지를 포함할 수도 있다.
  - 파일시스템으로 비유
    - 패키지 = 디렉토리, 모듈 = 디렉토리 안 파일
  - <img src="https://github.com/boostcamp-ai-tech-4/ai-tech-interview/raw/main/images/adc/python/42_package.png" alt="img" style="zoom:50%;" />



### Python lists 대신 Numpy arrays를 사용할 때의 이점은?

`list`

- 효율적인 범용 컨테이너.
- 효율적인 삽입, 삭제 ,추가 및 연결을 지원
- list comprehenshion을 통해 쉽게 구성하고 조작 할 수 있다.
- 요소 별 덧셈 및 곱셈과 같은 `벡터화 된 연산`을 지원하지 않는다.
- 유형이 다른 객체를 포함할 수 있다는 사실은 Python이 모든 요소에 대한 유형 정보를 저장해야하며 작동 할 때 유형 디스패치 코드를 각 요소에 실행해야함을 의미.

`Numpy`

- 더 효율적, 편리
- 많은 벡터 및 행렬 연산, 불필요한 작업 피하기 가능
- 더 빠르며, `Numpy`, `FFT`, `convolution`, `빠른 검색`, `기본 통계`, `선형 대수`, `히스토그램`등이 많이 내장되어있다.



### Python은 OOPs 개념이 있는가?

- Python은 객체 지향 프로그래밍 언어이다.
- Python의 주요 OOP 개념에는 `Class`, `Object`, `Method`, `Inheritance(상속)`, `Polymorphism(다형성)`, `데이터 추상화`, `Encapsulation(캡슐화)`를 포함.



### Python에서 Multithreading을 어떻게 하는가?

파이썬에는 `Multithreading` 패키지가 있지만, 일반적으로 코드 속도를 높이기 위해 Multithread 패키지를 사용하는 것은 좋지 않다.

파이썬에는 `GIL(Global Interpreter Lock)`이라는 구조가 있다. GIL은 한번에 하나의 스레드만 실행할 수 있도록 한다. 스레드는 GIL을 획득하고 약간의 작업을 수행 한 다음 GIL을 다음 스레드로 전달한다. 이 작업은 매우 빠르게 수행되므로 사람의 눈에는 스레드가 병렬로 실행되는 것처럼 보일 수 있지만 실제로 동일한 CPU 코어를 사용하여 번갈아 가며 수행한다.



### Python libraries?

파이썬 라이브러리는 패키지의 모음이다.

- 주로 `Numpy`, `Pandas`, `Matplotlib`,`Scikit-learn(사이킷런)`



### Python에서의 monkey patching?

- 주로 테스트를 위해 많이 사용되는 방법

- 어떤 클래스나 모듈의 일부(함수나 변수 등)를 로컬에서 런타임으로만 instance를 통해 수정하는 방법

- ex>

  - `health.py` 파일의 `A` 클래스에 `a`라는 함수가 있는데, 다른 파일에서 `A`를 import하여 `a` 함수 대신 `new_a`를 할당하여 사용하는 방법.

    ```python
    from health import A
    
    A.a = new_a
    my_A = A()	# A 클래스 객체 할당
    my_A.a		# new_a가 동작
    ```

    

### Python에서의 Polymorphism(다형성)이란?

`다형성`

- 여러가지 형태를 가질 수 있는 능력
- 유지보수에 도움을 준다

파이썬에서는 `+` 연산이나 `len` 연산에 대해 생각해볼 수 있다. 이들은 여러 타입의 변수에 대해서도 동일한 기능을 제공하는데 `overriding`과 `overloading`을 통해 각기 달든 타입의 변수에도 반응하도록 다형성을 주었기 때문에 가능하다.



### Python에서의 encapsulation(캡슐화)란?

`캡슐화`

- 주요 변수나 함수를 외부로부터 보호하는 방법.
- 캡슐화를 통해 코드의 안전성 높임

파이썬에서는 클래스를 생각해 볼 수 있다. 클래스의 멤버 변수나 멤버 함수에 접근하기 위해서는 클래스에 대한 객체를 만들어야 한다. 객체를 통해 멤버에 접근하기 때문에 직접 변수를 손대는 것보다 데이터를 더 안전하게 지킬 수 있다.



### Python에서의 data abstraction 어떻게?

`데이터 추상화`

- 사용자에게 데이터의 주요 정보만 제공하여 구체적인 구현은 몰라도 사용할 수 있게 만드는 방법

파이썬에서는 `abstract class`를 통해 데이터 추상화를 할 수 있다.

`abstract class`를 사용하기 위해서는 `abc` 모듈을 import 하고 `metaclass=ABCClass`와 `@abstractmethod`를 사용

```python
from abc import *
class 추상클래스명(metaclass=ABCMeta):
	@abstractmethod
    def 추상메소드(self):
        pass
```



### Python에서 접근 제한자 사용가능?

파이썬은 다른 언어와 달리 `private`, `protected`, `public` 등 접근 제한자를 직접 명시하지 않고 `변수명`을 통해 접근 제어를 한다.

- `변수명__` : public (접두사 _ 가 없거나 접미사 _ 가 두 개 이상 인 경우)
- `_변수명` : protected
- `__변수명__` : private



### What does an object() do?

파이썬은 모든 것이 객체이다. 따라서 **기본적으로 object 클래스를 상속받고 있다.**

object()함수를 사용하면 새로운 기본 object 객체를 반환 받을 수 있다.



### Python에서의 map function이란?

`map` 함수는 iterable 한 객체의 모든 원소에 동일한 함수를 적용하는 기능이다.

- 첫 인자 : 적용할 함수
- 두번째 인자 : iterable 한 객체
- 그러면 iterable 한 map 객체 형태로 각 원소에 대해 함수가 적용된 묶음들이 담겨 나온다.

```python
int_arr = list(map(int,input().split()))
```



### python numpy가 lists보다 좋은가?

파이썬의 리스트는 각 원소들의 값을 직접 사용하지 않고 `원소들의 주소를 참조하는 방식`이기 때문에 원소들의 타입이 정해지지 않아 편리하지만 `메모리를 많이 사용하고 느리다는 단점이 있다.`

반면, `Numpy`는 C 기반으로 구현되어 `원소들의 타입을 미리 설정하여` 메모리를 적게 사용하고 빠르다. 또한, 행렬고 선형대수에 편리한 함수들을 제공한다는 장점도 있다.



### Python에서의 Decorators란?

함수를 인자로 받고 내부 함수에서 인자로 받은 함수를 사용하는 클래스나 함수가 있을 때, 인자로 사용할 함수를 간편하게 지정해주는 역할.

Decorator 사용 문법 : 인자가 될 함수 위에 `@외부함수이름`을 붙여주면 된다.

```python
import time

def make_time_checker(func):
	def new_func(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print('실행시간:',end_time - start_time)
        return result
    return new_func

@make_time_checker
def big_number(n):
    return n**n**n

@make_time_checker
def big_number2(n):
    return (n+1) ** (n+1) ** (n+1)
```

- Decorator를 통해 big_number와 big_number2 라는 서로 다른 함수를 make_time_checker가 인자로 받아 내부 함수에서 사용하고 있다.



### object interning이란?

파이썬에서는 모든 것이 객체 -> 변수들은 값을 바로 가지지 않고 값을 가진 주소를 참조하게 된다.

`object interning`은 자주 사용될, 즉 재활용될 object에 대해 매번 새로운 주소를 할당하는 것은 비효율적이므로, 하나의 주소에 값을 주고 그 주소를 재활용하는 작업을 말한다.



### `@classmethod`, `staticmethod`, `@property`란?

`@classmethod`

- 클래스 내부의 함수 중 `@classmethod`로 선언된 함수에 대해서는 클래스의 객체를 만들지 않고도 바로 접근이 가능하다. 하지만 함수의 첫 인자로 클래스를 받아서, 상속되었을 때 자식 클래스의 데이터를 따르는 특징이 있다.

`@staticmethod`

- `@staticmethod`는 `@classmethod`와 마찬가지로 클래스의 객체를 만들지 않고도 바로 접근할 수 있다. 하지만 클래스를 인자로 받지 않기 때문에, 상속되었을 때에도 자식 클래스의 데이터를 따르지 않고 처음 클래스에서 선언한 데이터대로 함수가 사용된다.

```python
class Language:
    default_language = "English"
    
    def __init__(self):
        self.show = '나의 언어는 ' + self.default_language
        
    @classmethod
    def class_my_language(cls):
        return cls()
    
    @staticmethod
    def static_my_language():
        return Language()
    
    def print_language(self):
        print(self.show)
        
class KoreanLanguage(Language):
    default_language = "한국어"
```

```python
from language import *
a = KoreanLanguage.static_my_language()
b = KoreanLanguage.class_my_language()

a.print_language()	# 나의 언어는 English
b.print_language()	# 나의 언어는 한국어
```

`@property`

- 객체지향 언어에서는 `캡슐화`를 위해 변수를 직접 지정하지 않고, 객체의 함수를 통해 지정하고 받아오는 `setter`, `getter` 함수를 사용한다. 파이썬에서는 이를 편하게 사용할 수 있도록 `@property`를 제공한다.
- `getter`가 될 함수에 `@property`를, `setter`가 될 함수에 `@변수.setter`를 붙이면 된다.

```python
class Test:
	def __init__(self):
        self.__color = "red"

    @property
    def color(self):
        return self.__color
    
    @color.setter
    def color(self.clr):
        self.__color = clr

if __name__ == '__main__':
    t = Test()
    t.color = "blue"
    
    print(t.color)
```
