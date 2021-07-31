# 언더바 사용하기

#코드 1 = 1부터 9까지 자연수를 더하기
summary = 0
for i in range(1,10):
    summary +=i

print(summary)

# 코드 2 = "Hello World"를 5번 출력하기

for _ in range(5):
    print("Hello world")

# 리스트 관련 메소드

a = [1, 4, 3]
print(a)

a.append(2) # 리스트에 원소 삽입
print("삽입 : ",a)

a.sort()
print("오름차순 정렬 : ",a)

a.sort(reverse=True)
print("내림차순 정렬 : ",a)

a = [4,3,2,1]

a.reverse()
print("원소 뒤집기 : ",a)

a.insert(2,3)
print("인덱스 2에 3 추가 : ",a)

# 특정 값인 데이터 개수 세기
print("값이 3인 데이터 개수 : ", a.count(3))

# 특정 값 데이터 삭제
a.remove(1)
print("값이 1인 데이터 삭제 : ",a)

a = [1,2,3,4,5,5,5]
remove_set = {3, 5} # 집합 자료형

# remove set에 포함되지 않은 값만을 저장
result = [i for i in a if i not in remove_set] # 중요
print(result)

