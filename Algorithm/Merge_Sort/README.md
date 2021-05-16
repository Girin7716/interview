# 병합 정렬(Merge Sort)

### 개념

- 주어진 배열을 원소가 하나 밖에 남지 않을 때까지 계속 둘로 쪼갠 후, 다시 크기 순으로 재배열 하면서 원래 크기의 배열로 합친다.
  - ex> `[6,5,3,1,8,7,2,4]`
  - `[6,5,3,1][8,7,2,4]` -> divide
  - `[6,5][3,1][8,7][2,4]` -> divide
  - `[6][5][3][1][8][7][2][4]` -> divide
  - `[5,6][1,3][7,8][2,4]`  -> sort & merge
  - `[1,3,5,6][2,4,7,8]` -> sort & merge
  - [1,2,3,4,5,6,7,8] -> sort & merge

- 분할 정복(Divide and Conquer)
  - 요소를 쪼갠 후, 다시 합병시키면서 정렬해나가는 방식으로 쪼개는 방식은 퀵 소트와 유사
-  빠른 정렬로 분류되며, Quick Sort와 함께 많이 언급
- stable
- `list[:mid]` 이런식의 slicing은 간결한 코드를 작성할 수 있지만, slice를 할 때 배열의 복제가 일어나므로 메모리 사용 효율이 좋지 않다.



### 구현

```python
# 병합 정렬
def mergeSort(data):
    def sorting(low,high):
        # 종료조건
        if high-low < 2:
            return
		# divide
        mid = (low+high) // 2
        sorting(low, mid)
        sorting(mid, high)
        # sort & merge
        merge(low, mid, high)

    def merge(low,mid,high):
        temp = []
        l , h = low,mid
        while l < mid and h < high:
            if data[l] < data[h]:
                temp.append(data[l])
                l += 1
            else:
                temp.append(data[h])
                h += 1

        while l < mid:
            temp.append(data[l])
            l += 1
        while h < high:
            temp.append(data[h])
            h += 1

        for i in range(low,high):
            data[i] = temp[i - low]
    return sorting(0,len(data))

data = [13,5,1,3,7,9,5,4,3,10]
mergeSort(data)
print(data)
```





### 시간복잡도

- 평균 : O(NlogN)
- 최선 : O(NlogN)
- 최악 : O(NlogN)





