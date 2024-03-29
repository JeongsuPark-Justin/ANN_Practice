# 거스름돈 아이디어
# 당신은 음식점의 계산을 도와주는 점원입니다. 카운터에는 거스름돈으로 사용할 500원, 100원, 50원, 10원 짜리 동전이
# 무한히 존재한다고 가정합니다. 손님이게 거슬러 주어야 할 돈이 N원일 때 거슬러 주어야 할 동전의 최소 개수를 구하세요
# 단 거슬러 줘야 할 돈 N은 항상 10의 배수입니다

# if N = 1260
# 500원 2개, 100원 2개 50원 1개 10원 1개
# 큰 단위가 항상 작은 배수의 단위이므로 작은 단위의 동전들을 종합해 다른 해가 나올 수 없음 -> 정당함

n = 1260
count = 0

array = [500, 100, 50, 10] # 큰 단위의 화폐부터 차례대로 확인하기

for coin in array :
    count += n // coin # 해당 화폐로 거슬러 줄 수 있는 동전의 개수 세기
    n %= coin # n은 coin 으로 나눈 나머지 값

print(count)