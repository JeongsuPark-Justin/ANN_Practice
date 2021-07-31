# 반복문 for문 while문
i = 1
result = 0

# While 문
while i <= 9:
    result += i
    i += 1

print(result)

i = 1
result =0
while i <= 9:
    if i % 2 == 1:
        result += i
    i += 1
print(result)

# for 문
# for 변수 in 리스트 :

array = [9, 8, 7, 6, 5]
for  x in array:
    print(x)

result = 0
for i in range(1, 10):
    result += i

print(result)

# continue  쓰기
result = 0
for i in range(1, 10):
    if i % 2 ==0:
        continue
    result +=  i
print(result)

# break  쓰기
i = 1
while True:
    print("현재 i의 값 : ", i)
    if i ==5:
        break
    i += 1

scores = [90, 85, 77, 65, 97]
for i in range(5):
    if scores[i] >= 80:
        print(i+1, "번 학생은 합격입니다.")

scores = [90,  85, 77, 65, 97]
cheating_student_list = {2, 4}

print('---------------------')
for  i  in range(5):
    if i +1 in cheating_student_list:
        continue
    if scores[i] >= 80:
        print(i+1, "번 학생은 합격입니다")


# 구구단

for i in range (1, 10):
    for j in range(1, 10):
        print(i, "x", j, "=", i*j)
   