
data = dict() # 사전 자료형 초기화 @@

data['사과'] = 'Apple'
data['바나나'] = 'Banana'
data['코코넛'] = 'Coconut'

print(data)

if '사과' in data:
    print("'사과'를 키로 가지는 데이터가 존재합니다")
else:
    print("'사과'를 키로 가지는 데이터가 대신 존재합니다")

key_list = data.keys()

value_list = data.values()
print(key_list)
print(value_list)

# 각 키에 따른 값을 하나씩 출력하는 명령
for key in key_list:
    print(data[key])