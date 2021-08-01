# 문자열 재정렬
# 알파벳 대문자와 숫자로만 구성된 문자열이 입력
# 이때 모든 알파벳을  오름차순으로 정렬하여 이어서 출력한 뒤에 그 뒤에 모든 숫자를 더한 값을 이어서 출력
# 예를들어 K1KA5CB7 이라는 값이 들어오면 ABCKK13을 출력

data = input()
result = []
value = 0

# 문자를 하나씩 확인하며
for x in data:
    # 알파벳인 경우 리스트에 삽입
    if x.isalpha():
        result.append(x)
    # 숫자는 따로 더하기
    else:
        value += int(x)
# 알파벳을 오름차순으로 정렬
result.sort()

# 숫자가 하나라도 존재하는 경우 가장 뒤에 삽입
if value != 0:
    result.append(str(value))

print(''.join(result)) # join([]) = 리스트를 문자열로 바꿔줌 ex) _.join(['a', 'b', 'c']) -> abc
