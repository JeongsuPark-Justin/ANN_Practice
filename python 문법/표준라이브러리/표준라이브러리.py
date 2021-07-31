# 내장함수
# itertools : 순열과 조합 라이브러리
# heapq : 힙 자료구조 제공 (우선순의 큐 기능 위해 사용 : 다익스트라)
# bisect : 이진탐색기능
# collections : 덱, 카운터 등 자료구조 포함
# math : 필수적인 수학적 기능 (팩토리얼, 제곱근, 최대공약수, 삼각함수 관련 함수)

# 내장함수
# sum
result = sum([1, 2, 3, 4, 5])
print(result)

# min(), max()
min_result = min(7, 3, 5, 2)
max_result = max(7, 3, 5, 2)
print(min_result, max_result)

# eval() 계산 결과를 수 형태로 반환
result = eval("(3+5)*7")
print(result)

# sorted 반복 가능한 객체가 들어왔을때 정렬된 결과를 반환
result = sorted([9, 1, 8, 5, 4])
reverse_result = sorted([9, 1, 8, 5, 4], reverse=True)
print(result)
print(reverse_result)

# 순열과 조합
# nPr, nCr

# 순열
from itertools import permutations

data = ['a', 'b', 'c']
result = list(permutations(data,3))
print(result)

#조합
from itertools import combinations
result = list(combinations(data, 2))
print(result)

# 중복순열
from itertools import product
result =  list(product(data, repeat=2)) # 두개를 뽑는 모든 순열 구하기 (중복 허용)
print(result)

# 중복 조합
from itertools import combinations_with_replacement
result = list(combinations_with_replacement(data,2)) # 두개를 뽑는 모든 조합 구하기 (중복 허용)
print(result)

# Counter : 등장 횟수를 세는 기능
from collections import Counter
counter = Counter(['red', 'blue', 'red', 'red',  'green', 'blue'])
print(counter['blue'])
print(counter['red'])
print(dict(counter)) # 사전 자료형으로 반환

import math
#  최소 공배수를 구하는 함수
def lcm(a,b):
    return a*b // math.gcd(a,b)
a=21
b=14

print(math.gcd(a, b)) # 최대 공약수
print(lcm(a, b)) # 최소 공배수