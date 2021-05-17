# 스택 2개로 큐 자료구조 구현하기

class Queue():
    # __변수명 : private / __변수명__ : public / _변수명 : protected
    __stack1 = []   # private
    __stack2 = []   # private
    def __init__(self):
        self.__stack1 = []
        self.__stack2 = []

    def push(self,data):
        self.__stack1.append(data)

    def pop(self):
        if self.__stack2:
            return self.__stack2.pop()
        else:
            while self.__stack1:
                self.__stack2.append(self.__stack1.pop())
            return self.__stack2.pop()

q = Queue()
q.push(3)
q.push(5)
print(q.pop())  # 3
q.push(6)
print(q.pop())  # 5
q.push(1)
q.push(2)
q.push(4)
print(q.pop())  # 6
print(q.pop())  # 1
q.push(5)
print(q.pop())  # 2
print(q.pop())  # 4
print(q.pop())  # 5