# 정수 N이 입력되면 00시 00분 00초부터 N시 59분 59초까지 모든 시각 중에 3이 하나라도 포함되는 모든 경우의 수를 구하는 프로그램을 작성하시오
# 세어야 하는 시각
# 00시 00분 03초
# 00시 13분 30초

# 완전탐색 문제 : 단순히 1씩 시각을 증가시키면서 3이 하나라도 포함되어있는지를 확인

N = int(input())
count=0
for i in range (N+1):
    for j in range(60):
        for k in range(60):
            # 매 시각 안에 '3' 이 포함되어있다면 카운트 증가
            if '3' in str(i) + str(j) + str(k): # '3' 은 문자열 = string 이기 때문에 i 를 str 로 바꿔줘야 함
                count +=1
print(count)