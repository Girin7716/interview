# 병합 정렬

def mergeSort(data):
    def sorting(low,high):
        if high-low < 2:
            return
        mid = (low+high) // 2
        sorting(low, mid)
        sorting(mid, high)
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