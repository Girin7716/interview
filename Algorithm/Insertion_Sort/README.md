# 삽입 정렬(Insertion Sort)

### 개념

- 2번째 원소부터 시작하여 그 앞의 원소들과 비교하여 삽입할 위치를 지정한 후, 원소를 뒤로 옮기고 지정된 자리에 자료를 삽입하여 정렬하는 알고리즘
- 최선의 경우, O(N)이며 기본적으로는 O(N^2)이다.

### 구현

앞에서부터 순차적으로 정렬함

```python
# 삽입 정렬
def insertionSort(data):
    for end in range(1,len(data)):
        to_insert = data[end]
        i = end
        while i > 0 and data[i-1] > to_insert:
            data[i] = data[i-1]
            i -= 1
        data[i] = to_insert
        print("{} 단계 : {}".format(end,data))

data = [6,5,3,1,8,7,2,4]
insertionSort(data)
print(data)
```



출력

```
1 단계 : [5, 6, 3, 1, 8, 7, 2, 4]
2 단계 : [3, 5, 6, 1, 8, 7, 2, 4]
3 단계 : [1, 3, 5, 6, 8, 7, 2, 4]
4 단계 : [1, 3, 5, 6, 8, 7, 2, 4]
5 단계 : [1, 3, 5, 6, 7, 8, 2, 4]
6 단계 : [1, 2, 3, 5, 6, 7, 8, 4]
7 단계 : [1, 2, 3, 4, 5, 6, 7, 8]
```

