# 버블 소트
def bubbleSort(data):
    for i in range(len(data)):
        for j in range(len(data)-i-1):
            if data[j] > data[j+1]:
                data[j],data[j+1] = data[j+1],data[j]
            print('{}단계 : {}'.format(i+1,data))

bubbleSort([6,2,4,3,7,1,9])