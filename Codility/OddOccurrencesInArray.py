# 비어 있지 않은 리스트 A 가 있는데
# 홀수개의 정수 N 들을 담고있다.
# 이 N중에서 짝을 이룰 수 없는 정수가 하나 존재한다.
# 짝을 이룰 수 없는 정수를 출력해라.

# 예를 들어
# A = [ 9, 3, 9, 3, 9, 7, 9 ] 인경우 7이 반환되어야 한다.

import collections

def solution(A):
    # write your code in Python 3.6
    counted = collections.Counter(A)
    for key, val in counted.items():
        if val % 2 == 1:
            return key

