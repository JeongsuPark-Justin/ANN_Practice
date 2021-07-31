# 프로그램의 흐름을 제어하는 함수

x = int(input())
if x >= 10:
    print("x>=10")
else:
    print("x<10")


# if  조건문1 :
# elif 조건문 2 :
# else :

a = -1
if a >= 0:
    print("a는 양수입니다")
elif a >= -10:
    print("a는 -10보다 큽니다")
else:
    print("a는 죤나게 작습니다")

score = 85
if score >= 90:
    print("학점 A")
elif score >=80:
    print("학점 B")
elif score >=70:
    print("학점 C")
else :
    print("학점 F")

# 비교연산자 파일 확인
# X==Y : X와 Y가 서로 같을 때 참이다
# X!=Y : X와 Y가 서로 다를 때 참이다
# 논리 연산자도 확인2

if True or False:
    print("yes")

a = 15
if a<=20 and a>=0:
    print("yes")

# 기타연산자 (in , not in)

k = [1, 2, 3, 4, 5, 6, 7]
l = {5, 6, 7, 8, 9, 40}

if 5 in k and l :
    print("True")

# 조건문의 간소화
score = 86
result = "Success" if score >=80 else "Fail"
print(result)

# Example
