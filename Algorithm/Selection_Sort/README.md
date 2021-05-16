# 선택 정렬(Selection Sort)

### 개념

- 해당 순서에 원소를 넣을 위치는 이미 정해져있고, 어떤 원소를 넣을지 선택하는 알고리즘
- 현재 위치에 저장될 값의 크기가 작냐, 크냐에 따라서 오름차순 정렬과 내림차순 정렬로 나뉜다.
- Unstable
  - ex> 5,4,5,2,3
  - 5(1),4(2),5(2),2(3),3(4)
  - 2(1),4(2),5(2),5(1),3(4) 가 되므로 unstable하다.(5(1)이 먼저 들어왔는데 5(2)보다 뒤에 있으니까)



### 구현

앞에서 부터 조건에 맞는 숫자를 찾아서 위치 고정

```python
# 선택 정렬
def selectionSort(data):
    for i in range(len(data)):
        standard = i
        for j in range(i+1,len(data)):
            if data[j] < data[standard]:
                standard = j

        data[standard], data[i] = data[i], data[standard]

        print("{} 단계 : {}".format(i+1,data))

selectionSort([7,6,2,4,3,9,1])
```



### 시간 복잡도

n + (n-1) + (n-2) + ... + 2 + 1 = n(n-1)/2 이므로 O(N^2)이다.