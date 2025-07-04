---
title: "신경망"
layout: post
parent: deeplearning
has_children: false
nav_order: 403
last_modified_at: 2021-02-15T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - 신경망
---


# 신경망

> 신경망은 가중치 매개변수의 적절한 값을 데이터로 자동으로 학습

## 신경망의 예

![](https://t1.daumcdn.net/cfile/tistory/2117013E5928016429)

- 입력층 은닉층 출력층
  - 입력층 0층
  - 은닉층 1층(사람 눈에는 보이지 않는다)
  - 출력층 2층
```
      0 (b + w1x1+ w2x2 <=0)
y =
      1 (b + w1x1+w2x2 > 0)
```
- x1과 x2라는 두 신호를 입력받아  y를 출력하는 퍼셉트론

```
y = h(b + w1x1 + w2x2)
```
- 입력 신호의 충합이 h(x) 라는 함수를 거쳐 변환되어 y로 출력됨을 의미

```
        0 (x<=0)
h(x) = 
        1 (x>0)
```
- 결국 입력 값 x 가 0보다 크면 1을 돌려주고 그렇지 않으면 0을돌려주는 h(x) 함수가 나옴 
- 결과적으로 위 식 3개는 하는일이 동일하다

## 활성화 함수

> 위 h(x) 함수 같이 입력신호 총합을 출력신호로 변환하는 함수를 활성화 함수라 한다(activation function)

- a = b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>
  - 가중치가 달린 입력 신호와 편향의 총합 계산하고 이를 a 라한다
- y = h(a)
  - 위 a를 함수 h()에 넣어 y를 출력한다

> 퍼셉트론에서 활성화 함수는 임계값을 경계로 출력이 바뀐다 이를 계단 함수(step function)이라한다

### 시그모이드 함수

h(x) = 1 / (1 + exp(-x))

- exp(-x)는 e<sup>-x</sup>를 뜻하며 자연 상수 2.7182.. 의 값을 갖는 실수
  - 시그모이드 역시 단순 함수로 입력을 주면 아웃을줌
    - ex) h(1.0) = 0.731.., h(2.0) = 0.880

```python
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0
```

- 계단함수의 단순한 구현
- 인수x로 넘파이 배일열 받고싶어 아래와 같이 수정

```python
def step_function(x):
  y = x > 0
  return y.astype(np.int)
```
```python
import numpy as np
x = np.array([-1.0, 1.0, 2.0])
x # array([-1., 1., 2.])
y = x > 0
y # array([False, True, True], dtype =bool)
y = y.astype(np.int)
y # array([0,1,1])
```
- 넘파이 배열에 부등호 연산을 수행하면 원소 각각 에 bool 배열 생성
- int형 결과를 원하니 astype  으로 자료형 변환

### 시그모이드 구현

```python
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
```
- np.exp(-x)는 epx(-x) 수식에 해당
- 넘파이도 사용 가능

```python
x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
# array([0.26894142, 0.73105858, 0.88079708])
```
- 넘파이 브로드캐스트로 수행이 가능하다

#### 시그모이드 계단 함수 차이

![](https://sean-parkk.github.io/assets/images/DLscratch/3/Untitled%206.png)

- 가장 큰 차이는 매끄러움
- 공통점
  - 비선형 함수
    - 선형 함수는 f(x) = ax + b 일때 a와b가 상수인 한개의 직선 함수

- 신경망은 비선형 함수를 사용해야만함
  - 선형 함수는 층을 아무리 깊게 해도 은닉층이 없는 네트워크로 똑같이 만들수 있다

### ReLU함수
> 시그모이드 함수를 대신 주로 사용하는 함수
- ReLU : 입력이 0 을 넘으면 그입력을 그대로 출력하고 0 이하면 0을 출력하는 함수
```
        x (x > 0)
h(x) = 
        0 (x <= 0)
```
```python
def relu(x):
  return np.maximum(0,x) # 둘중 큰값 반환
```

## 다차원 배열 계산
> 넘파이의 다차원 배열을 사용한 계산법

### 다차원 배열

> n차원으로 나열하는 배열

```python
import numpy as np
A = np.array([1,2,3,4])
print(A)
# [1 2 3 4]
np.ndim(A)
# 1
A.shape
# (4,)
A.shape[0]
# 4

B = np.array([[1,2],[3,4],[5,6]])
print(B) # [[1 2] [3 4] [5 6]]
np.ndim(B) # 2
B.shape # (3,2)
```
```
1 2
3 4
5 6
```

- 3x2 배열로 처음 차원에 원소가3개
- 2차원 배열은 특히 행렬
  - 가로는 행, 세로는 열

### 행렬의 곱

**A**

|1|2|
|---|---|
|3|4| 

**B**

|5|6|
|---|---|
|7|8|

- 위 두 행렬의 A,B 곱  C는 아래와 같인 방법으로 계산
  - 1 * 5 + 2 * 7 = 19
  - 3 * 5 + 4 * 7 = 43

**C**

|19|22|
|---|---|
|43|50|

```python
A = np.array([[1,2], [3,4]])
A.shape # (2, 2)
B = np.array([[5,6], [7,8]])
B.shape # (2,2)
np.dot(A, B)
# array([[19,22], [43, 50]])
```
- np.dot 은1차원 배열이면 벡터, 2차원 배열이면 행렬곱 계산
- 두 행렬의 곱은 두 행렬의 대응 하는 차원의 원소 수를 일치 시켜야함
  - (3열 * 2행)  (2열 * 4행) = (3열 * 4행) 행렬 형태로 나옴
    - 3, 4로 결과의 모습
    - 2, 2가 동일해야 계산이 가능(다르면 계산 불가)
      - 행렬과 1차원 배열 계산도 가능 (3열 * 2행) (2열) = (3열)

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A.shape # (3, 2)
B = np.array([7, 8])
B.shape # (2,)
np.dot(A, B)
# array([[23, 53, 83]])
```

### 신경망 행렬 곱

|1|3|4|
|--|--|--|
|2|4|6|

```
x1        (1,2) y1
x2        (3,4) y2
          (4,5) y3  
```          
- 편향과 활성화 함수를 생략하고 가중치만 표시
```
X      W   =  Y
2    2 * 3    3
-일치-   -일치-
```
- 위 구현에서도 X와 W의 대응하는 차원의 원소수가 같아야 함

```python
X = np.array([1, 2])
X.shape # (2,)
W = np.array([[1, 3, 5], [2, 4, 6]])
#array[[1 3 5]
#      [2 4 6]]
W.shape # (2,3)
Y = np.dot(X,W)
# array([5, 11, 17])
```

### 3층 신경망 구현

- 입력층은 2개, 1번층(은닉층) 3개 2번층(은닉층) 2개 출력층은 2개 뉴런

W<sup>(1)</sup><sub>1 2</sub>
- (1): 1층의 가중치
- 1: 다음 층의 1번째 뉴런
- 2: 앞 층의 2번째 뉴런
- 1,2 반대로 쓰는 경우도 있으니 확인이 필요

- a<sup>(1)</sup><sub>1</sub> = w<sup>(1)</sup><sub>1 1</sub>x<sub>1</sub> + w<sup>(1)</sup><sub>1 2</sub>x<sub>2</sub> + b<sup>(1)</sup><sub>1</sub>
  - 위 식을 행렬 곱으로 간소화
    - A<sup>(1)</sup> = WX<sup>(1)</sup> + B<sup>(1)</sup>
      - 이 행렬은
      - A<sup>(1)</sup> = (a<sup>(1)</sup><sub>1</sub> a<sup>(1)</sup><sub>2</sub> a<sup>(1)</sup><sub>3</sub>)
      - X = (x<sub>1</sub> x<sub>2</sub>)
      - B<sup>(1)</sup> = [[(w<sup>(1)</sup><sub>1 1</sub> w<sup>(1)</sup><sub>2 1</sub> w<sup>(1)</sup><sub>3 1</sub>)], [w<sup>(1)</sup><sub>1 2</sub> w<sup>(1)</sup><sub>2 2</sub> w<sup>(1)</sup><sub>3 2</sub>]]

- 입력층에서 1층 신호 전달

```python

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

W1.shape # (2,3)
X.shape # (2,)
B1.shape # (3,)

A1 = np.dot(X, W1) + B1
# array([0.3, 0.7, 1.1])
Z1 = sigmoid(A1)
# array([0.57444252, 0.66818777, 0.75026011])
```

- 1층에서 2층으로 신호 전달

``` python
W2 = np.array([[0.1, 0.4], [0.2,0.5], [0.3,0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
# array([0.3, 0.7, 1.1])
Z2 = sigmoid(A2)
```

- 2층에서 출력층으로 신호 전달

``` python
def identity_functionm(x):
  return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # 혹은 Y = A3
```

## 출력층

- 신경망은 분류와 회귀 모두에 사용가능
- 어떤 문제냐에 따라 출력층에서 활성화 함수가 다르다
- 회귀에는 항등 함수
  - 입력데이터에서 수치를 예측하는 문제(사진 속 인물의 뭄무게 57.4kg 등을 예측)
- 분류에는 소프트 맥스
  - 분류 : 데이터가 어느 클래스에 속하느냐

### 항등 소프트 맥스

> 항등함수(identity function)은 입력을 그대로 출력

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F7o3ns%2FbtqvQDIyhq4%2FFYgVfbO6NaJrkc7y11f440%2Fimg.png)

- exp(x) 는 e<sup>x</sup>를 뜻하는 지수 함수(e는 자연상수)
- n은 출력층의 뉴런 수, y<sub>k</sub>는 그중 k번째 출력

```python
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) # 지수 함수
#array([ 1.34985881, 18.17414537, 54.59815003])
sum_exp_a = np.sum(exp_a) # 지수 함수의 합
print(sum_exp_a)
# 74.1221542101633

y = exp_a / sum_exp_a
# array([0.01821127, 0.24519181, 0.73659691])
```

```python
def softmax(a):
  exp_a = nmp.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  
  return y
```

### 소프트 맥수 함수 구현 주의사항

- 오버플로 문제 주의
- e<sup>10</sup> 은 20000 e<sup>100</sup> 은 0만 40개가 넘고e<sup>1000</sup>은 inf등으로 돌아온다
- 큰값의 나눗셈은 값이 불안해진다

```python
a = np.array([1010, 1000, 990]) # 소프트맥스 함수의 계산
np.exp(a) / np.sum(np.exp(a)) # 제대로 계산되지 않음
# array([nan, nan, nan]) 
c =  np.max(a)
a- c
# array([0, -10, -20])
np.exp(a-c) / np.sum(np.exp(a-c))
# array
```

```python
def softmax(a):
  c = np.max(a)
  exp_a = nmp.exp(a - c)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  
  return y
```

### 소프트맥스 특징
- 출력은 0 ~ 1 사이의 실수
- 출력의 총합은 1
  - 확률로 해석 가능


## 손글씨 숫자 인식

- 이미 학습된 매개변수를 사용하여 학습과정 생략후 추론과정만 구현
- 추론과정을 신경망의 순전파(forward propagation) 라고도 한다
- 피클(pickle) 파이썬 기능으로 저장해둔 파일을 로드하여 실행 당시의 객체를 즉시 복원가능하다

### 신경망 추론처리
 - 입력층 뉴런은 784
   - 28 * 28 이미지 크기
 - 출력층 뉴런은 10
   - 이미지는 0~9 까지 숫자를 나타내기 때문

### 배치처리

- X(100 * 784) W1(784 * 50) W2(50 * 100) W3(100 * 10) Y(100 * 10)
  - 784 입력 이미지 100개를 동시에 처리 입력데이터를 묶음으로 처리하는것을 배치라고한다
