# 퀵소트
import random

def quick_sort(data):
    if len(data) <= 1:
        return data

    pivot = data[random.randrange(0,len(data))]
    less_arr, equal_arr, big_arr = [],[],[]
    for i in data:
        if i < pivot:
            less_arr.append(i)
        elif i > pivot: big_arr.append(i)
        else: equal_arr.append(i)

    return quick_sort(less_arr) + equal_arr + quick_sort(big_arr)

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
print(quick_sort(data))
data = [14, 5, 10, 8, 9, 13, 15, 4, 13, 10]
quickSort(data, 0, len(data) - 1)
print(data)