
for i in range(5):
    for j in range(5):
        print('(', i, ',', j, ')', end='') # end = 공백 없애기
        print()

# 동 북 서 남
dx = [0, -1, 0, 1]
dy = [1, 0, -1, 0]

# 현재 위치
x, y = 2, 2

for i in range(4):
    # 다음 위치
    nx = x + dx[i]
    ny = y + dy[i]
    print(nx, ny)
