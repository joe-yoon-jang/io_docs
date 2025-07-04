---
title: "퍼셉트론"
layout: post
parent: deeplearning
has_children: false
nav_order: 402
last_modified_at: 2021-02-09T16:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - perceptron
  - 퍼셉트론
---

# 퍼셉트론

- 고대 화석같은 알고리즘 
- 신경망(딥러닝)의 기원이 되는 알고리즘

## 퍼셉트론이란

![](https://raw.githubusercontent.com/haedal-with-knu/H-A.I/master/Lecture/Img/p_op.png)

x1, x2 입력신호에 가중치를 곱한다 w1x1, w2x2 이 총합이 정해진 한계를 넘어설 때만 1을 출력한다(뉴런이 활성화한다라고 표현하기도 함). 이 한계를 임계값이라고 하며 θ(세타)로 나타낸다

## 논리회로
### AND 게이트

|x1|x2|y|
|---|:---:|---:|
|0|0|0|
|1|0|0|
|0|1|0|
|1|1|1|

> 진리표 "x1,x2 입력신호와 y출력 신호의 대응표"

- 위 표대로 동작하는 w1,w2,θ 값을 정함
    -  (0.5, 0.5, 0.7)
    -  (0.5, 0.5, 0.8)
    -  (1, 1, 1)

### NAND 게이트
>and 게이트와 같은 not and
### OR 게이트
>and 게이트와 같은 or

## 간단한 구현

```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
```

### 가중치와 편향

![2](https://raw.githubusercontent.com/haedal-with-knu/H-A.I/master/Lecture/Img/p_op.png)

- 위 식에 편향 b를 추가
- b + w1x1 + w2x2


```python
import numpy as np

x = np.array([0,1]) # 입력
w = np.array([0.5, 0.5]) # 가중치
b = -0.7 # 편향
w*x # array([0.,0.5])
np.sum(w*x) # 0.5
np.sum(w*x) + b # -0.199999.. 대략 -0.2 부동소수점 수에 의한 연산 오차
```

```python
def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

- theta 가 편향 b로 치환
- 편향은 w1,w2와 기능이 다르다
- w1,w2
    - 각 입력신호가 결과에 주는 영향력(중요도) 조절 하는 매개변수
- b
    - 편향은 뉴런이 얼마나 쉽게 활성화 하느냐를 조정하는 매개변수

## 퍼셉트론의 한계

### xor 게이트 구현 불가능

> xor 진리표

|x1|x2|y|
|---|:---:|---:|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|0|

- 퍼셉트론은 선형만 나타낼수있다
- 비선형 (xor같은)을 할수없다

### 다중 퍼셉트론

> 층을 쌓아 다중 퍼셉트론을 구현

- AND, NAND, OR 게이트 조합하여 XOR게이트

|x1|x2|s1|s2|y|
|---:|---:|---:|---:|:---:|
|0|0|1|0|0|
|1|0|1|1|1|
|0|1|1|1|1|
|1|1|0|1|0|

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

1. 0층의 두 뉴런이 입력 신호를 받아 1층의 뉴런으로 신호를 보낸다
2. 1층의 뉴런이 2층의 뉴런으로 신호를 보내고 2층의 뉴런은 y를 출력한다

* 단층 퍼셉트론으로 표현하지 못하면 층을 하나 늘려 구현