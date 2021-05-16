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