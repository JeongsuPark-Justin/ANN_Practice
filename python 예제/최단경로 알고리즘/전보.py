# 어떤 나라에는 N개의 도시가 있다. 그리고 각 도시는 보내고자 하는 메세지가 있는 경우, 다른 도시로 전보를 보내서 해당 메세지를 전송할  수 있다.
# 하지만 X라는 도시에서 Y라는 도시로 전보를 보내고자 한다면, 도시 X 에서 Y로 향하는 통로가 설치되어 있어야 한다.
# 예를 들어 X에서 Y로 향하는  통로는 있지만 Y에서 X로 향하는 통로가 없다면 Y는 X로 메세지를 보낼 수 없다.
# 또한 통로를 거쳐 메세지를 보낼 때는 일정 시간이 소요된다.
# 어느 날 C라는 도시에서 위급 상황이 발생했다. 그래서 최대한 많은 도시로 메세지를 보내고자 한다.
# 메세지는 도시 C에서 출발하여 각 도시 사이에 설치된 통로를 거쳐 최대한 많이 퍼져나갈 것이다.
# 각 도시의 번호와 통로가 설치되어 있는 정보가 주어졌을 떄, 도시 C에서 보낸 메세지를 받게 되는 도시의 개수는 총 몇개이며
# 도시들이 모두 메세지를 받는 데 까지 걸리는 시간은 얼마인지 계산하는 프로그램을 작성하시오

import  heapq
import sys
input = sys.stdin.readline
INF = int(1e9)

def dijkstra(start):
    q = []
    # 시작 노드로 가기 위한 최단 거리는 0으로 설정하여, 큐에 삽입
    heapq.heappush(q, (0, start))
    distance[start] = 0
    while q: # 큐가 비어있지 않다면
        # 가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
        dist, now = heapq.heappop(q)
        # 현재 노드가 이미 처리된 적이 있는 노드라면 무시
        if distance[now] < dist:
            continue
        # 현재 노드와 연결된 다른 인접한 노드들을 확인
        for i in graph[now]:
            cost = dist + i[1]
            # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

# 노드의 개수, 간선의 개수, 시작 노드를 입력받기
n, m, start = map(int, input().split())

# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n + 1)]
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n + 1)

# 모든 간선 정보를 입력 받기
for _ in range(m):
    a, b, c = map(int, input().split())
    # a 번 노드에서  b 번 노드로 가는 비용이 c라는 의미
    graph[a].append((b, c))

# 다익스트라 알고리즘을 수행
dijkstra(start)

# 도달할 수 있는 노드의 개수
count = 0
# 도달할 수 있는 노드중에서 가장 멀리 있는 노드와의 최단 거리
max_distance = 0
for d in distance:
    # 도달할 수 있는 노드인 경우
    if d != 1e9:
        count +=1
        max_distance = max(max_distance, d)

# 시작  노드는 제외해야 하므로 count - 1를 해서 출력
print(count - 1, max_distance)