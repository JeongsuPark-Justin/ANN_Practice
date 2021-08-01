# 정수는 NxM 크기의 직사각형 형태의 미로에 갇혔습니다. 미로에는 여러마리 괴물이 있어 이를 피해 탈출해야 합니다.
# 정수의 위치는 (1,1)이며 미로의 출구는 (N,M)의 위치에 존재하며 한번에 한 칸씩 이동할 수 있습니다.
# 이때 괴물이 있는 부분은 0으로, 괴물이 없는 부분은 1로 표시되어 있습니다. 미로는 반드시 탈출할 수 있는 형태로 제시됩니다.
# 이때 정수가 탈출하기 위해 움직여야 하는 최소 칸의 개수를 구하세요.
# 칸을 셀 떄는 시작 칸과 마지막 칸을 모두 포함해서 계산합니다.

def bfs(x,y):
    from collections import deque
    # 큐 구현을 위해 deque 라이브러리 사용
    queue = deque()
    queue.append((x,y))
    # 큐가 빌때까지 반복하기
    while  queue:
        x, y = queue.popleft()
        # 현재 위치에서 4가지 방향으로의 위치 확인
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            # 미로 찾기 공간을  벗어난 경우 무시
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            # 벽인 경우 무시
            if graph[nx][ny] == 0:
                continue
            # 해당 노드를 처음 방문하는 경우에만 최단 거리 기록
            if graph[nx][ny] == 1:
                graph[nx][ny] = graph[x][y] +1
                queue.append((nx, ny))
    # 가장 오른쪽 아래까지의 최단거리 반환
    return graph[n -1][m -1]

# N, M을 공백을 기준으로 구분하여 입력 받기
n, m = map(int,input().split())
# 2차원 리스트의 맵 정보 입력 받기
graph = []
for i in range(n):
    graph.append(list(map(int, input())))

# 이동할 네가지 방향 정의 (상, 하, 좌, 우)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# BFS를 수행한 결과 출력
print(bfs(0,0))
