# 1이 될때까지 : 문제 설명
# 어떠한 수 N이 1이 될 때 까지 다음의 두 과정 중 하나를 반복적으로 선택하여 수행하려고 합니다.
# 단, 두번째 연산은 N 이 K로 나누어 떨어질때만 선택할 수 있습니다
# 1. N 에서 1을 뺍니다
# 2. N을 K로 나눕니다

# 문제 해결법 - 최대한 많이 나누어주자!

n, k = map(int, input().split()) # 공백을 기준으로 구분하여 입력 받기
result = 0

while True :
    target = (n//k) * k # N이 K로 나누어 떨어지는 수가 될때까지 빼기
    result += (n - target)
    n = target

    if n < k : # N이 K 보다 작을 때 (더 이상 나눌 수 없을 때) 반복문 탈출
        break
    result += 1 # K로 나누기
    n //= k
result += (n-1) # 마지막으로 남은 수에 대하여 1씩 빼기
print(result)