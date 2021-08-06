A = ['abc','def','hello world', 'hello', 'python']
B = 'Life is too short, You need python'.split()
print(A)
print(sorted(A, key=len)) # 문자열 개수 순으로 정렬
print(sorted(B, key=str.lower)) # 알파벳 순으로 정렬

print(5 in [1, 2, 3, 4, 5])