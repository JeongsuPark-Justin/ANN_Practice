# 함수 정의
# 1. 내장함수 : 파이썬이 기본적으로 제공하는 함수 print, input
# 2. 사용자 정의 함수 : 개발자가 직접 정의해서 사용할 수 있는 함수 (코딩테스트)

# def 함수명(매개변수) :
#     실행할 소스코드
#     return 반환 값 //함수에서 처리 된 결과를 반환

def add(a,b):
    return a+b

print(add(3,7))

def subtract(a,b):
    return a-b

print(subtract(7,3))

a = 10
def func():
    global a # 지역변수 쓰기 위한 명령어
    a+=1
    print(a)


func()

for i in range(10):
    func()

def operator(a, b):
    add_var = a+b
    subtract_var = a-b
    multiply_var = a*b
    divide_var = a/b
    return add_var, subtract_var, multiply_var, divide_var

print(operator(3,4))


# 람다 표현식 (이름없는 함수)

print((lambda a,b : a+b)(3,7))   # 한줄 정의 가능

array = [('홍길동', 50), ('이순신', 32), ('아무개', 77)]

def my_key(x):
    return x[1]

print(sorted(array, key=my_key))
print(sorted(array, key=lambda x: x[1]))