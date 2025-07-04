---
title: "신경망 학습"
layout: post
parent: deeplearning
has_children: false
nav_order: 404
last_modified_at: 2021-02-22T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - 신경망학습
---


# 신경망학습

> 학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는것을 말한다

## 데이터에서 학습

- 가중치 매개변수의 값을 데이터를 보고 자동으로 결정

### 데이터 주도 학습

- 기계학습은 데이터에서 잡을 찾고 데이터에서 패턴을 발견하고 데이터로 이야기를 만드는 것
- 데이터가 없으면 아무것도 할수없다

### 훈련데이터와 시험 데이터

- 훈련데이터 : 학습하며 최적의 매개변수를 찾는다
- 시험데이터 : 앞서 훈련한 모델의 실력을 평가
- "범용능력"을 제대로 평가하기 위해 훈련과 시험 데이터를 구분
* 오버피팅을 주의(한 데이터셋에 지나치게 최적화된 상태)

## 손실함수(loss function)
> 신경망은 하나의 지표를 기준으로 최적의 매개변수값을 탐색하며 지표는 손실함수라고 한다.
### 오차제곱합

![](https://blog.kakaocdn.net/dn/R0Ope/btqCPzxBk4O/TziWb03jyywpcl0PewQKW1/img.png)

- y<sub>k</sub> 는 신경망의 출력
- t<sub>k</sub> 는 정답 레이블
- k 데이터 차원수

```python
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]
```

- y 는 소프트 맥수 함수의 출력
- t 는 정답 레이블로 정답 위치는1 그외는 0
  - 원-핫 인코딩: 한 원소만1 그외 0 표기법 
```python
def sum_squares_error(y,t):
  return 0.5 * np.sum((y-t)**2)
```
- 위식 그대로 구현
```python
sum_squares_error(np.array(y), np.array(t))
# 0.09750000000000003
y = [0.1,0.05,0.1,0.5,0.05,0.1,0.0,0.1,0.0,0.0]
sum_squares_error(np.array(y), np.array(t))     
# 0.5475
```
- 처음 경우 0.0975.. 로 오차율이 낮고 그다음 0.5 오차울로 오차율 차이를 알수있다

### 교차엔트로피 오차(cross entropy error CEE)

![](https://user-images.githubusercontent.com/10937193/58108191-577e3080-7c26-11e9-8b54-097fec3e5f0e.png)

- log는 밑이 e인 자연로그
- y<sub>i</sub>는 신경망의 출력
- t<sub>i</sub>는 정답레이블
  - 원-핫 인코딩으로 정답만 1 그외는 0
- 정답일 때 추정(t가 1일때의 y)의 자연로그 계산하는식
  - 정답 레이블 2가 정답이라고 할때 신경망 출력이 0.6이라면 오차는 -log0.6 = 0.51 이된다
  - 같은 조건에서 신경망 출력이 0.1 이라면 -log0.1 = 2.30이 된다

```python
def cross_entropy_error(y,t):
  delta = 1e-7 # 0 을 입력시 마이너스 무한대를 방지하기위한 아주 작은값을 만듬
  return -np.sum(t * np.log(y+delta))
```
```python
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
cross_entropy_error(np.array(y), np.array(t))
# 0.510825457099338
y = [0.1,0.05,0.1,0.5,0.05,0.1,0.0,0.1,0.0,0.0]
cross_entropy_error(np.array(y), np.array(t))     
# 2.302584092994546
```

### 미니배치 학습

- 60000장의 이미지 중 100장을 무작위로  뽑아 학습하는 방법

### 손실 함수 설정이유

- 정확도를 사용하지 않고 손실함수를 사용하는 이유는
  - 미분의 역활에 주목하면 해결
  - 최적의 매개변수(가중치와 편향)를 탐색할때 손실함수 값을 가능한 한 작게 하는 값을 찾음
  - 이때 매개변수의 미분(기울기)를 계싼하고 갱신하는 과정 반복
- 신경망을 학습할 때 정확도를 지표로 삼아서는 안된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이되기 때문이다.

## 수치 미분

### 미분

- 10분에서 2km 씩 달렸을때 속도는 2/10 = 0.2km/분 즉 1분에 0.2km 만큼의 속도로 뛰었다고 해석 가능하다
  - 이는 10분간 평균속도를 나타낸다 이때 미분은 특정 시간의 변화량을 의미한다.
    - 달리고 10초 후 속도 등

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/731355a21e3ca457060e3043c7088b6b7d38890c)

- 좌변은 f(x)의 x에 대한 미분(x에 대한 f(x)의 변화량)을 나타내는 기호
- 시간의 작은 변화 x를 한없이 0에 가깝게한다는 의미를 lim로 나타낸다

```python
#나쁜 구현 예
def numerical_diff(f, x):
  h = 10e-50
  return (f(x+h) - f(x)) / h
```
- 개선점
  1. 0.000..1의 형태에 0이 50개인 값 h이다 이 방식은 반올림 오차 문제를 일으킨다
    - 10<sup>-4</sup> 정도의 값이 좋은 결과를 얻는다고 알려져 있다
  2. f의 차분(임의 두점에서의 함수 값들의 차) 
    - x+h와 x사이의 함수 f의 차분을 계산하고 있지만 (x+h)와 x사이의 기울기에 미분과 구현값은 일치하지는 않는다.
    - 이는 h를 무한히 0 으로 좁히는 것이 불가능해서 그렇다
```python
#개선
def numerical_diff(f, x):
  h = 1e-4 # 0.0001
  return (f(x+h) - f(x-h)) / (2*h)
```

### 수치 미분의 예

> y = 0.01x<sup>2</sup> + 0.1x

```python
def function_1(x):
  return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1) # 0에서 20까지 0.1간격의 배열x를 만든다(20포함)
y = function_1(x)
numerical_diff(function_1, 5)
# 0.19999...
numerical_diff(function_1, 10)
# 0.29999...
```
- 미분값이 x에 대한f(x)의 변화량 즉 기울기다
- f(x) = 0.01x<sup>2</sup>+0.1x 의 해석적 해는 0.02x+0.1
  - x가 5일때와 10일 떄의 진정한 미분은 0.2, 0.3이다 앞 수식 근사치와 오차가 매우 작음을 알수있다

### 편미분

> f(x<sub>0</sub>, x<sub>1</sub>) = x<sup>2</sup><sub>0</sub> + x<sup>2</sup><sub>1</sub>

```python
def function_2(x):
  return x[0]**2 + x[1]**2 # return np.sum(x**2)
```

1. x<sub>0</sub> = 3 x<sub>1</sub> = 4 일때 x<sub>0</sub> 편미분   ∂f/∂x<sub>0</sub> 를 구하라

```python
def function_tmp1(x0):
  return x0*x0 + 4.0**2.0

numerical_diff(function_tmp1, 3.0)
# 6.00000000000378
```

## 기울기

- 편미분 x0과 x1은 변수별로 따로 계산한다
- 동시에 계산하고고싶을때 모든 변수의 편미분을 벡터로 정리한 것을 기울기라고한다

```python
def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

  for idx in range(x.size):
    tmp_val = x[idx]
    # f(x+h) 
    x[idx] = tmp_val+h
    fxh1 = f(x)

    #f(x-h)
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val # 값 복원
  return grad
```

- 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향이다

### 경사법

- 기계학습 문제 대부분 학습 단계에서 최적의 매개변수를 찾는다
- 그러나 매개변수 공간이 광대하여 최소값 짐작이 힘들때 기울기를 이용해 최솟값을 찾으려는것이 경사법이다
- 경사법은 최솟값 최대값을 찾느냐에 따라 이름이 다르다(경사하강법, 경사 상승법)

![](https://media.vlpt.us/images/jakeseo_me/post/1d5481d5-c66d-4c92-86d7-b4c8c83c8c60/gradient_descent_method_equation.png)

- η 기호 에타는 갱신하는 양을 나타낸다 (신경망 학습에서 학습률)

```python
def gradient_descent(f, init_x, lr=0.01, step_num = 100):
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x
```
- 경사 하강법 구현은 위와 같다
- 경사법으로 f(x0, x1) = x<sup>2</sup><sub>0</sub> + x<sup>2</sup><sub>1</sub> 의 최소값을 구하라

```python
def function_2(x):
  return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x, 0.1, 100)
# array([-6.11110793e-10, 8.14814391e-10])
```

### 신경망에서의 기울기

- 가중치 매개변수에 대한 손실 함수의 기울기
- 형상이 2X3, 가중치가 W, 손실함수가 L인 신경망은
  - ∂L/∂W 와 같다
  
W  = 
|W<sub>11</sub>|W<sub>12</sub>|W<sub>13</sub>|
|--|--|--|
|W<sub>21</sub>|W<sub>22</sub>|W<sub>21</sub>|

∂L/∂W  = 
|∂L/∂W<sub>11</sub>|∂L/∂W<sub>12</sub>|∂L/∂W<sub>13</sub>|
|--|--|--|
|∂L/∂W<sub>21</sub>|∂L/∂W<sub>22</sub>|∂L/∂W<sub>23</sub>|

- ∂L/∂W 의 각 원소는 각각의 원소에 관한 편미분이다
  - 1행 1번째 원소 ∂L/∂W<sub>11</sub> 은 W<sub>11</sub>을 변경했을떄 손실 함수L이 얼마나 변화하느냐를 나타낸다

## 학습 알고리즘

- 전제
  - 신경망에는 가중치와 편향이 있고 훈련데이터 적응하도록 조정하는 과정을 학습이라한다
- 1단계 - 미니배치
  - 훈련 데이터 중 일부를 무작위로 선별하여 손실 함수 값을 줄인다
- 2단계 - 기울기 산출
  - 미니배치의 손실 함수 값을 줄이기 위해 가중치 매개변수의 기울기를 구한다. 기울기는 손실함수의 값을 가장 작게하는 방향을 제시한다
- 3단계 - 매개변수의 갱신
  - 가중치 매개변수를 기울기방향으로 아주 조금 갱신한다
- 4단계 - 반복
  - 1~3단계 반복