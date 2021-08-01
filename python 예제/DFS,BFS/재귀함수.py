# 재귀함수(Recursive Function)란 자기 자신을 다시 호출하는 함수를 의미
# 단순한 형태의 재귀 함수 예제
# ' 재귀 함수를 호출합니다.' 라는 문자열을 무한히 출력합니다
#  어느 정도 출력하다가 최대 재귀 깊이 초과 메세지가 출력됩니다

"""
def recursive_function():
    print("재귀 함수를 호출합니다.")
    recursive_function()

recursive_function()

"""
# Stack 에 데이터를 넣었다가 빼는 효과

def recursive_function(i):
    # 10번째 호출을 했을 때 종료되도록 종료 조건 명시
    if i ==10:
        return
    print(i, '번째 재귀함수에서', i+1, '번째 재귀함수를 호출합니다')
    recursive_function(i+1)
    print(i, '번째 재귀함수를 종료합니다.')

recursive_function(1)

