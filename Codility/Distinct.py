# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")



def solution(A):
    # write your code in Python 3.6
    A.sort()
    count = 1
    for i in range(1, len(A)):
        if A[i-1] != A[i]:
            count += 1
        else :
            continue
    print(count)


solution([2, 1, 1, 2, 3, 1])