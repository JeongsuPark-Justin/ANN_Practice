# 금광
# n x m 크기의 금광이 있습니다. 금광은 1x1 크기의 칸으로 나누어져 있으며, 각 칸은 특정한 크기의 금이 들어있습니다.
# 채굴자는 첫번째 열부터 출발하여 금을 캐기 시작합니다. 맨 처음에는 첫번째 열의 어느 행에서든 출발할 수 있습니다.
# 이후에 m-1번에 걸쳐서 매번 오른쪽 위, 오른쪽, 오른쪽 아래 3가지 중 하나의 위치로 이동해야 합니다
# 결과적으로 채굴자가 얻을 수 있는 금의 최대 크기를 출력하는 프로그램을 작성하세요.

# 테스트 케이스 입력
for tc in range(int(input())):
    # 금광 정보 입력
    n, m = map(int, input().split())
    array = list(map(int, input().split()))
    # 다이나믹 프로그래밍을 위한 2차원 dp 테이블 초기화
    dp = []
    index = 0
    for i in range(1, m):
        for j in  range(n):
            # 왼쪽 위에서 오는 경우
            if i == 0: left_up = 0
            else: left_up = dp[i-1][j-1]
            # 왼쪽 아래서 오는 경우
            if i == n -1 : left_down = 0
            else: left_down = dp[i+1][j-1]
            # 왼쪽에서 오는 경우
            left = dp[i][j-1]
            dp[i][j] = dp[i][j] + max(left_up, left_down, left)
    result = 0
    for i in range(n):
        result = max(result, dp[i][m-1])

    print(result)