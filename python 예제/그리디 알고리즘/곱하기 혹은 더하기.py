# 곱하기 혹은 더하기 : 문제 설명
# 0-9로 이루어진 문자열 S가 주어졌을 때 왼쪽부터 오른쪽으로 하나씩 모든 숫자 사이에 + 혹은 x 를 넣어 가장 큰 수를 구해라
# 20억 이하의 정수가 되도록 입력이 주어진다 (int 형)

data = input()
print(len(data))
result = int(data[0]) # 첫번째 문자를 숫자로 변경하여 대입

for i in range (1, len(data)): # 두 수 중에서 하나라도 0 혹은 1인 경우 곱하기 보다 더하기 수행
    num = int(data[i])
    if num <= 1 or result <= 1:
        result += num
    else :
        result *= num

print(result)
