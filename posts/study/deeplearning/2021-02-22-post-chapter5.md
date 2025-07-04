---
title: "오차역전파법"
layout: post
parent: deeplearning
has_children: false
nav_order: 405
last_modified_at: 2021-02-22T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - 신경망학습
---


# 오차역전파법

> 수치미분은 구현하기 쉽지만 계산시간이 오래걸린다는 단점이 있다 이에 효율적인 오차역전파법을 공부한다

## 계산 그래프

- 100원 짜리 사과2개를 샀다. 지불 금액을 구하라 단 소비세 10%가 부과된다.
  1. 사과100 -> 곱하기 -> 곱하기 -> 지불금액
  2. 사과의 개수 2
  3. 소비세 1.1
  - 사과 개수와 곱해서 200, 이후 소비세와 곱하고 220


- 100원 짜리 사과2개, 150원짜리 귤 3개를 샀다. 지불 금액을 구하라 단 소비세 10%가 부과된다.

```
사과의 개수 2 ->
                  X 200
사과 가격 100 -> 
                          + 650  
                                   X 715
귤 가격 150   -> 
                  X 450
귤의 개수 3   ->
소비세 1.1                      ->
```
1. 계산 그래프를 구성
2. 왼쪽에서 오른쪽으로 계산을 진행(순전파)
  - 반대는 역전파

### 국소적 계산

- 계산을 국소적으로 간단하게 진행한다

### 덧셈 노드 역전파

- z = x + y 라는 미분은 ∂z/∂Lx = 1 ∂z/∂Ly = 1로 해석적으로 계산가능
  - 덧셈 노드 역전파: 입력 값을 그대로 흘려보낸다
  - 10 + 5 = 15 이면 역전파는 1.3 + 1.3 = 1.3 으로 그대로 보낸다
    - 입력 신호를 그대로 출력할뿐이므로 다음 노드로 전달한다

### 곱셈 노드 역전파
- z = xy 의 미분은  ∂z/∂Lx = y  ∂z/∂Ly = x
- 10 * 5 = 50 은 1.3 = 6.5 * 13 으로 역전파 계산이된다

### 앞 문제 역전파 

```
사과의 개수 2(110) ->
                  X 200(1.1)  
사과 가격 100(2.2) -> 
                          + 650(1.1)  
                                   X 715 (1)
귤 가격 150(3.3)   -> 
                  X 450(1.1)  
귤의 개수 3(165)   ->
소비세 1.1                   (650)->
```

- 사과는 2.2의 역전파 값을 갖는다
  - 이는 1원이 올랐을떄 2.2원이 오른다는 의미(정확히는 2.2배만큼 오른다는 뜻)

## 연쇄법칙

- 국소적인 미분을 오른쪽에서 왼쪽으로 전달한다
- 국소적 미분을 전달하는 원리는 연쇄법칙에 따른것
### 연쇄법칙이란
  - 합성함수 z = t<sup>2</sup> 가 t = x + y 로 구성
  - 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다

  1. z = t<sup>2</sup> 가 t = x + y
  2. x 에 대한 z의 미분 ∂z/∂x = ∂z/∂t * ∂t/∂x 로 나타낼수있다(연쇄법칙)
  3. ∂z/∂t = 2t, ∂t/∂x = 1
  4. ∂z/∂x = ∂z/∂t * ∂t/∂x = 2t * 1 = 2(x + y)


### 계산 그래프의 역전파
- y = f(x)  의 역전파는  E(∂y/∂x)  <- E
- 계산절차
1. 신호 E에 노드의 국소적 미분 ∂y/∂x 을 곱한 후 다음 노드로 전달
2. 

## 계층 구현

- forward와 backward 공통 인터페이스로 진행

```python
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    return out
  def backward(self, dout):
    dx = dout * self.y #x와 y를 바꾼다
    dy = dout * self.x
    return dx, dy

class AddLayer:
  def __init__(self):
    pass

  def forward(self, x, y):
    out = x + y
    return out
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy
```
## 활성화 함수 계층 구현학기

> 계산 그래프를 신경망에 적용, ReLU, Sigmoid 계층 구현

### ReLU 계층

```   
      x  (x>0)
y = { 
      0  (x <= 0)


            1 (x > 0)
∂y/∂x   = { 
            0 (x <= 0)
```
* x > 0

x              y
→              →
       relu
←              ←
∂L/∂y        ∂L/∂y

* x <= 0

x              y
→              →
       relu
←              ←
0             ∂L/∂y


```python
class Relu:
  def __init__(self):
    self.mask = None

  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out

  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# array([[ 1. , -0.5], [-2. ,  3. ]])
mask = (x <= 0)
mask
# array([[False,  True], [ True, False]])
```
- Relu 클래스는 mask 라는 인스턴스 변수를 갖는다
  - mask: True, False 로 구성된 넘파이 배열
    - 순전파 입력 X의 원소의 값이 0이하인 인덱스는 True, 그 외는 False

> ReLU 계층은 스위치와 같다. 순전파 떄 전류가 흐르면 ON 아니면OFF로한다. 역전파 떄는 스위치가 ON이라면 전류가 그대로 흐르고 OFF이면 흐르지 않는다.

### sigmoid

y = 1/(1 + exp(-x))

- 순정파
  1. 곱하기 노드: x * -1 
  2. exp 노드: exp(-x) 
  3. 더하기 노드 : 1 + exp(-x)
  4. 나누기 노드 : 1/(1 + exp(-x))
  - x => * -1 => exp => + 1 => / 
  - y = 1/(1 + exp(-x))
- 역전파
  1. 나누기 노드: ∂y/∂x = -1/x<sup>2</sup> = -y<sup>2</sup>
    - 상류 값에 -y<sup>2</sup> 값을 곱하여 전달 -∂L/∂y*y<sup>2</sup>
  2. 더하기 노드: 그대로 전달 -∂y/∂x * y<sup>2</sup>
  3. exp 노드: y = exp(x) 수행 미분 값 ∂y/∂x = exp(x)
    - 상류 값에 순전파 떄 출력을 곱해 하류로 전달(exp(-x))
    - -∂y/∂x * y<sup>2</sup>exp(-x)
  4. 곱하기 노드 : 순전파 값을 서로 바꿔 곱
    - 위 예는 -1 곱함 ∂y/∂x * y<sup>2</sup>exp(-x)

## affine/Softmax 계층 

### affine 계층

> X, W, B 가 행렬로 계산하는 계층

- X(2,), W(2,3)행렬을 곱하고 B(3,)를 더해 Y(3,)를 구할때 
  - X * W + B = Y
- 역전파
  - ∂L/∂Y(3,) 입력시
  - +노드는 그대로 ∂L/∂Y(3,)
  - 곱하기 X노드는 ∂L/∂Y(3,) * W<sup>T</sup>(전치행렬)(2,3)

### softmax with Loss 계층

- 계층 마지막에 Softmax-with-Loss 노드를 둡니다. 
- Softmax는 신경망 학습할때 필요한 계층
- CEE도 올수 있다

### 신경망 학습의 전체 그림

- 전체: 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 "학습"이라고 한다. 
1. 미니배치: 훈련 데이터 중 일부를 무작위로 가져온다. 선별한 데이터를 미니배치라고한다. 미니 배치의 손실 함수 값을 줄이는 것이 목표
2. 기울기 산출: 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다.
3. 매개변수 갱신: 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.
4. 반복: 1~3단계를 반복한다.

- 이 중 2번 기울기 산출에서 오차역전파법을 사용한다. 수치미분은 구현은 쉽지만 계산이 오래걸렸다. 