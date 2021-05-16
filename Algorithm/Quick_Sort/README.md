# 퀵소트

### 개념

- 분할 정복(Divide and Conquer) :  문제를 작은 2개의 문제로 분리하고 각각 해결한 다음, 결과를 모아 원래의 문제를 해결하는 전략. (즉, 큰 문제들을 작은 문제들로 쪼개서 문제를 해결하는 방법.)
- unstable하다.
  - ex> 4,6,6,1,8,5 이런 data가 순서대로 들어올 때,
  - 4(1),6,(2)6(3),1(4),8(5),5(6)인데, 이를 퀵소트하면 
  - 1(4),4(1),5(6),6(3),6(2),8(5)로 정렬될 수 있으며, 6(3),6(2) 부분 때문에 unstable하다.



### 구현

원래알던 구현

```python
def quickSort(data, left, right):
    L = left
    R = right
    pivot = data[(left+right) // 2] # pivot을 배열의 가운데에 위치한 요소로 선택

    while L<=R:
        # pivot 왼쪽에는 pivot보다 작은 원소가 위치해야함, 만약 큰 원소가 있다면 L 변수로 그 원소를 알 수 있음.
        while data[L] < pivot:
            L += 1
        # pivot 오른쪽에는 pivot보다 큰 원소가 위치해야함, 만약 작은 원소가 있다면 R 변수로 그 원소 알 수 있음.
        while pivot < data[R]:
            R -= 1

        # L과 R이 역전되지 않고, 같은 경우가 아니라면 두 원소의 위치 교환.
        # 이를 통해서 pivot 기준으로 왼쪽 : 작은 원소, 오른쪽 : 큰 원소 위치
        if L <= R:
            if L != R:
                data[L], data[R] = data[R], data[L] # swap
            L += 1
            R -= 1
    # L과 R이 역전된 후에 pivot의 왼쪽과 오른쪽에는 정렬되지 않은 부분 배열이 남아있을 수 있다.
    # 이 경우, 남아 있는 부분 배열에 뒤해서 quickSort 수행
    if left < R:
        quickSort(data, left, R)
    if L < right:
        quickSort(data, L, right)

data = [14, 5, 10, 8, 9, 13, 15, 4, 13, 10]
quickSort(data, 0, len(data) - 1)
print(data)
```



간단하게 구현

```python
# 퀵소트
import random

def quick_sort(data):
    # data가 길이가 하나이하면 정렬되어있는 거임. 기저조건
    if len(data) <= 1:
        return data
	# pivot 랜덤하게 선택
    pivot = data[random.randrange(0,len(data))]
    # pivot보다 작은거, 같은거, 큰거 담을 리스트
    less_arr, equal_arr, big_arr = [],[],[]
    # 리스트를 보면서
    for i in data:
        # 검사하는 원소의 값과 pivot을 비교하면서 알맞는 리스트에 넣어주기
        if i < pivot:
            less_arr.append(i)
        elif i > pivot: big_arr.append(i)
        else: equal_arr.append(i)
	# 그러면, pivot에 대해서 작은거, 같은거, 큰 것들이 구분이 되어있고, 같은 값은 정렬이 되어있으니 그대로 반환하고, 나머지 둘은 정렬이 안되어있을 수도 있으니 퀵소트를 수행해준다.
    return quick_sort(less_arr) + equal_arr + quick_sort(big_arr)

data = [14, 5, 10, 8, 9, 13, 15, 4, 13, 10]
print(quick_sort(data))
```



### 시간복잡도

- O(NlogN), 단 최악의 경우 O(N^2)

  - 그러므로, O(N^2) (빅오 : 최악의 시간으로 표기), Θ(NlogN) (세타 : 빅오와 빅 오메가의 공통부분)

- 최악의 경우가 O(N^2)인 이유

  - 정렬된 배열에서 바로 pivot을 최솟값이나 최댓값으로 선택한 경우 가장 큰 시간이 소요.
  - ex> [1,2,3,4,5,6,7]
    - pivot : 1, left : 1, right : 7 로 선택하면, 7부터 1과 비교한다. 하지만, 모든 데이터가 1보다 크기 때문에 left는 움직이지 않고 right가 left에게로 와서 1은 자기자리로 돌아가게 된다.
    - 그 다음, 1이 정렬이 끝났으니, 2를 기준 값으로 위 과정 반복.
    - 이러한 과정은 배열 안 데이터 수(N)만큼 진행하게 되고 매번 N-1 개의 데이터들과 비교하므로 최악의 경우 O(N^2)이 된다.

- O(NlongN)의 복잡도를 가지는 다른 정렬알고리즘 보다도 더 빠르게 수행된다는 것이 증명되어있다.

  - 불필요한 데이터의 이동을 줄이고, 먼 거리의 데이터를 교환할 분만아니라, 한번 결정된 기준은 추후 연산에서 제외되는 성질을 가졌기 때문.

  

