# 정수 N이 여러개 들어있는 array A 가 주어진다.
# 이때 A를 K번 회전 시키면 얻어지는 리스트를 구해라.

def solution(A, K):
    if not (A and K):
        return A
    K = K % len(A)
    return A[-K:] + A[:-K]

print(solution([3, 8, 9, 7, 6],3))