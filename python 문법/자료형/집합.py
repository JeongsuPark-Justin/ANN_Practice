# 중복 허용 안됨
# 순서 없음

# 리스트 혹은 문자열을 이용해서 초기화 할 수 있음 (이때  set() 활용)
# 중괄호 {} 안에 각 원소를 콤마를 기준으로 구분하여 삽입
# 데이터 조회 및 수정에 있어서 O(1) 시간에 처리 가능

data = set([1, 1, 2, 3, 4, 4, 5])

# OR

data = {1, 1, 2, 3, 4, 4, 5}

print(data) # 집합은 중복이 안됨

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}

# 합집합
print(a|b)
# 교집합
print(a&b)
# 차집합
print(a-b)

data = set([1, 2, 3])
print(data)

# 새로운 원소 추가
data.add(4)
print(data)

# 새로운 원소 여러개 추가
data.update([5, 6]) # 괄호 조심
print(data)

# 특정한 값을 가지는 원소 삭제
data.remove(3)
print(data)