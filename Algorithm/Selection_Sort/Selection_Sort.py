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