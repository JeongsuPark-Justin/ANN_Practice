
data = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
data.sort()
print(data)

# 선택 정렬 -->> O(N^2)
#  처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 앞으로 보내는 것

array = [7,  5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(len(array)):
    min_index = i # 가장 작은 원소의 인덱스
    for j in range(i +1, len(array)):
        if array[min_index] > array[j]:
            min_index = j
    array[i], array[min_index] = array[min_index], array[i]  # 스와프

print(array)

# 삽입 정렬
# 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입
# 선택 정렬에 비해 구현 난이도가 높은 편이지만. 일반적으로 더 효율적임

array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range (1, len(array)):
    for j in range(i, 0, -1): # 인덱스 i 부터 1 까지 1 씩 감소하며 반복하는 문법
        if array[j] < array[j -1]: # 한칸씩 왼쪽으로 이동
            array[j], array[j -1] = array[j -1], array[j]
        else : # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
            break

print(array)

# 퀵 정렬  -->> 이상적인 경우 O(NlogN)을 기대 가능 // 최악의 경우 O(N^2)
# 기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
# 일반적인 상황에서 가장 많이 사용하는 정렬 알고리즘 중 하나
# 병합 정렬과 더불어 대부분의 프로그래밍 언어의 정렬 라이브러리의 근간이 되는 알고리즘
# 가장 기본적인 퀵 정렬은 첫번째 데이터를 기준 데이터(Pivot)으로 설정

array = [5, 7, 9,  0, 3, 1, 6, 2, 4, 8]
def quick_sort(array, start, end):
    if start >= end: # 원소가 1개인 경우 종료
        return
    pivot = start  # 피벗은 첫 번째 원소
    left = start + 1
    right = end
    while(left <= right):
        # 피벗보다 큰 데이터를 찾을 때까지 반복
        while(left <= end and array[left] <= array[pivot]):
            left += 1
        # 피벗보다 작은 데이터를 찾을 때 까지 반복
        while(right > start and array[right] >= array[pivot]):
            right -= 1
        if(left > right): # 엇갈렸다면 작은 데이터와 피벗을 교체
            array[right], array[pivot] = array[pivot], array[right]
        else :
            array[left], array[right] = array[right], array[left]
        # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
        quick_sort(array, start, right -1 )
        quick_sort(array, right +1 , end)

quick_sort(array, 0, len(array)-1)
print(array)


# 피벗을 좀 더 간단하게!
array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]
def quick_sort(array):
    # 리스트가 하나 이하의 원소만을 담고 있다면 종료
    if len(array) <= 1:
        return array
    pivot = array[0] # 피벗은 첫 번째 원소
    tail = array[1:] # 피벗을 제외한 리스트

    left_side = [x for x in tail if x <= pivot]  # 분할된 왼쪽 부분
    right_side = [x for x in tail if x > pivot]  # 분할된 오른쪽 부분

    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행하고, 전체 리스트 반환
    return quick_sort(left_side) + [pivot] + quick_sort(right_side)

print(quick_sort(array))

# 계수정렬
# 특정한 조건이 부합할 때만 사용 할 수 있지만 매우 빠르게 동작하는 정렬 알고리즘
# 계수 정렬은 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때 사용 가능
# 데이터의 개수가 N, 데이터(양수) 중 최대값이 K 일때  최악의 경우에도 수행 시간 O(N+K)를 보장한다.

array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]

# 최종 리스트에 각 데이터가 몇번씩 등장했는지 count 한다
# 인덱스 : 0 1 2 3 4 5 6 7 8 9
# 개  수 : 2 2 2 1 1 2 1 1 1 2

# 출력결과 : 0 0 1 1 2 2 3 4 5 5 6 7 8 9 9

# 모든 범위를 포함하는 리스트 선언 (모든 값은 0으로 초기화)
count = [0] * (max(array) + 1)

for i in range(len(array)):
    count[array[i]] += 1  # 각 데이터에 해당하는 인덱스의 값 증가

for i in range(len(count)):  # 리스트에 기록된 정렬 정보 확인
    for j in range(count[i]):
        print(i, end=' ')  # 띄어쓰기를 구분으로 등장한 횟수만큼 인덱스 출력