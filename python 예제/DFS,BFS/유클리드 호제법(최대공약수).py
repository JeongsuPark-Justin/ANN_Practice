# 두개의 자연수에 대한 최대 공약수를 구하는 알고리즘 - 유클리드 호제법
#  두 자연수 A, B에 대하여 (A>B) A를 B로 나눈 나머지를 R 이라고 하자
# 이때 A와 B의 최대공약수는 B와 R의 최대공약수와 같다

# 예시 GCD(192,162)

def gcd(a,b):
    if a % b ==0:
        return b
    else :
        return gcd(b, a%b)

print(gcd(192, 162))