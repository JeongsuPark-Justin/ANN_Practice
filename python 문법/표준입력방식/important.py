# input() : 한 줄의 문자열을 입력받는 함수
# map() : 리스트의 모든 원소에 각각 특정한 함수를 적용할 때 사용
# ex) list(map(int, input().split()))
# ex) a, b, c = map(int, input().split())

n = int(input()) # 데이터의 개수 입력?>
data = list(map(int, input().split())) # 각 데이터를 공백을 기준으로 구분하여 입력
data.sort(reverse=True) # 내림차순으로 정렬
print(data)
print(n+3)

# 빠르게 입력받기
# sys 라이브러리에 정의되어있는 sys.stdin.readline() 이용
# 단 입력 후 엔터가 줄바꿈 기호로 입력되므로 rstrip() 를 함꼐 사용

import sys
# 문자열 받기
data = sys.stdin.readline().rstrip() # list 형으로 안나옴
print(data)