---
title: "학습 관련 기술들"
layout: post
parent: deeplearning
has_children: false
nav_order: 406
last_modified_at: 2021-03-16T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - 신경망학습
---


# 학습 관련 기술
- 가중치 매개변수의 최적값을 탐색하는 최적화 방법, 가중치 매개변수 초깃값, 하이퍼파라미터 설정방법등을 알려주는 파트
## 매개변수 갱신
- 최적화(optimization): 신경망 목적은 손실함수의 값을 가능한 한 낮추는 매개변수를 찾는 것. 이는 곧 매개변수의 최적값을 찾는 문제이며, 이러한 문제를 푸는 것을 최적화(optimization)이라 함
- 확률적 경사 하강법(SGD) : 매개변수의 기울기(미분)을 이용 매개변수의 기울기를 구하고 기울어진 방향으로 매개변수 값을 갱신하는 일을 반복하여 최적의 값을 찾는다

### 최적화 모험가 이야기(SGD)
- 제약2개(눈을가리고, 지도가없이)를 갖고 가장 깊은 곳(낮은곳)을 찾는 모험가
- 모험가는 발바닥으로 바닥의 기울기는 알수있음
- 가증 크게 기울어진곳으로 계속이동하다보면 가장낮은곳으로 도착할수도있다(SGD)

### 확률적 경사 하강법SGD

- W ← W - η(∂L/∂W)
- W: 갱신할 가중치 매개변수
- ∂L/∂W: W에 대한 손실 함수의 기울기
- η : 학습률을 의미하는데 실제로는 0.01이나 0.001같은 미리 정해진 값을 사용
- ← : 우변의값으로 좌변을 갱신한다는 뜻

```py
class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr
  def update(self, params, grads):
    for key in params.keys():
      params[key] -= self.lr * grads[key]
```

- lr: 학습률(learning rate)를 뜻함
- params, grads: 딕셔너리 변수 params[w1]과 같은 가중치 매개변수와 기울기를 저장하고있다

```py
network = TwoLayerNet(...)
optimizer = SGD()
for i in range(10000):
  ...
  x_batch, t_batch = get_mini_batch(...) # 미니배치
  grads = network.gradient(x_batch, t_batch)
  params = network.params
  optimizer.update(params, grads)
  ...
```

- 위 코드는 의사코드로 동작은 하지않는다.
- optimizer로 최적화 모듈을 지정하고 매개변수 갱신을 수행하게 함으로 매개변수와 기울기만 넘겨주고 맡길수있다.
- SGD와 같이 클래스를 모듈화 하여 사용하면 좋다
  - 예를들어 momentum 역시 update(params, grads) 라는 공통 메서드로 구현하여 optimizer = Momentum()으로만 변경하여 SGD를 momentum으로 변경할수 있다

### SGD의 단점

![sgd1](../../assets/images/220318.png)

- 이 수식은 x축 방향으로 늘인 타원모양으로 기울기가 나타난다

![sgd2](../../assets/images/220319.png)

- 위 수식은 최소값은 0,0 이나 대부분의 위치가 최수값을 가르키지 않는다.

![sgd3](../../assets/images/220320.png)

- 따라서 탐색을 하게되면 멍청하게 값을 갱신하게된다
- 비등방성(anisotropy)함수(방향에 따라 성질 즉 기울기가 달라지는 함수)에서 탐색 경로가 비효율적
- 근본 원인은 기울어진 반향이 최소값과 다른방향을 가리키고있다

### 모멘텀

- 모멘텀(momentum)은 운동량을 뜻하는 단어
- v  ← αv - η(∂L/∂W)
- W ← W + v
- W: 갱신할 가중치 매개변수
- ∂L/∂W: W에 대한 손실함의 기울기
- η: 학습률
- v: 물리에서 말하는 속도(velocity)에 해당한다
  - 초기화 때는 아무런 값을 담지 않고 update가 호출 될때 매개변수와 같은 구조의 데이터를 딕셔너리 변수로 저장한다.
- αv: 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할을 한다.(α는 0.9 등으로 설정한다)

```py
class Momentum:
  def __init__(self, lr=0.01, momentum=0.9):
    self.lr = lr
    self.momentum = mementum
    self.v = None
  def update(self, params, grads):
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)
    
    for key in params.keys():
      self.v[key] = self.momentum*self.v[keys] - self.lr*grads[key]
      params[key] += self.v[key]

```
- 모멘텀은 지그재그가 아닌 공이 그릇 바닥을 구르듯 값을 갱신한다(지그재그 정도가 덜하다)
  - 이는 x축의 힘은 아주 작지만 방향은 변하지 않아서 한방향으로 일정하게 가속하기 떄문
  - 거꾸로 y축의 힘은 크지만 위아래로 번갈아 받아서 상층하여 y축 방향의 속도는 안정적이지 않다

![momentum](../../assets/images/220321.png)

### AdaGrad

![AdaGrad1](../../assets/images/220322.png)

- 신경망 학습에서는 학습률 값(수식η) 이 중요하다
  - 너무 크면 발산하여 학습이 제대로 이뤄지지않고 너무작으면 학습시간이 너무 길어진다.
- 학습률을 정하는 효과적 기술 학습률 감소(learning rater decay)가 있다.
  - 처음에는 크게 학습하다가 조금씩 작게 학습
- h : 기존 기울기 값을 제곱하여 계속 더한다
- ⊙ 는 element wise multiplecation 이다(한국말로는 행렬의 원소별 곱셈)

- AdaGrad는 과거의 기울기를 제곱하여 계속 더해간다. 따라서 학습을 진행할수록 갱신강도가 약해진다. 이는 무한히 학습하면 갱신량이 0이되어 갱신되지 않게된다. 이를 개선하기 위해 RMSProp이라는 방법이 있다.
  - RMSProp 는 과거의 모든 기울기를 균일하게 더해가는 것이 아니라 오래된 기울기는 서서히 잊고 새로운 기울기를 크게 반영한다. 이를 지수이동평균(Exponential Moving Average EMA)라 한다.

![AdaGrad2](../../assets/images/220323.png)

- AdaGrad에 의한 최적화 갱신 경로

### Adam 
- 모멘텀은 공이 그릇 바닥을 구르듯한 움직임을보이고 AdaGrad는 매개변수의 원소 마다 적응적으로 갱신정도를 조정한다. 이 두방법을 융합한 방법이 Adam이다.
- 하이퍼 파라미터의 "편향보정"도 진행된다.
- Adam은 하이퍼 파라미터를 3개 설정한다. 
  - 지금까지의 학습률 α
  - 일차 모멘텀용 계수 β<sub>1</sub>
  - 이차 모멘텀용 계쑤 β<sub>2</sub>

![Adam](../../assets/images/220324.png)


### 어느 갱신 방법을 이용할 것인가?

- 정답은 없다.
- 케이스 별로 다르다.

## 가중치의 초기값
- 신경망 학습에서 특히 중요한 것이 가중치의 초깃값
- 권장 초기값 및 실험을 통해 실제로 신경망 학습이 신속하게 이뤄지는 모습을 확인가능하다

### 초깃값 0

- 오버피팅을 억제해 범용 성능을 높이는 테크닉인 가중치 감소 기법
  - 가중치 매개변수의값이 작아지도록 학습하는 방법. 작게 하여 오버피팅이 일어나지 않게한다
  - 그렇다면 모든 매개변수를 작게 주는게 유리하다
  - 하지만 모든 매개변수가 0이면 학습이 이루어지지 않는다.(정확히는 매개변수가 같은값이면 안된다) 이는 오차역전법에서 확인한 가중치 값 갱신이 동일하게 발생하기 떄문이다.

### 은닉충의 활성화값 분포

- w = np.random.randn(node, node) * 1
  - 시그모이드 함수는 출력이 0에 또는 1에 가까워지자 그미분은 0에 다가간다. 이는 활성화 값이 0과 1에 치우쳐 분포하게된다.. 그래서 데이터가 0과1에 치우쳐 분포하게 되면 역전파의 기울기 값이 점점 작아지다가 사라지고 이것이 기울기 소실(gradient vanishing)이라 알려진 문제이다.
- w = np.random.randn(node, node) * 0.01
  - 0.5 부군에 집중되어 활성화 값들이 치우치게된다. 이는 표현련 관점에서 큰문제로 다수의 뉴런이 거의 같은 값을 출력하고있어 뉴런이 여러개인 의미가 없어지게된다.
- Xavier 초기값은 활성화 함수가 선형인 것을 전제한다.
- sigmoid 함수와 tanh 함수는 좌우 대칭이라 중앙 부근이 선형인 함수로 볼 수 있어 위 초기값이 유효하다.
- ReLU는 He 초기값을 사용한다.

```py
import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1 / (1 + np.exp(-1))
    
x = np.random.randn(1000, 100) # 1000개의 데이터
node_num = 100                 # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5          # 은닉층이 5개
activations = {}               # 이 곳에 활성화 결과(활성화값)를 저장

for i in range(hidden_layer_size):
  if i != 0:
    x = activations[i-1]
  w = np.random.randn(node_num, node_num) * 1
  a = np.dot(x, w)
  z = sigmoid(a)
  activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
  plt.subplot(1, len(activations), i+1)
  plt.title(str(i+1) + '-layer')
  plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
```

![gradient vanishing](../../assets/images/220325.png)

- 5개 층이 존재하고 각 층 뉴런은 100개
- 활성화 함수로 시그모이드를 사용하여 각층의 결과를 activations 변수에 저장한다
- 표준편차가 1인 정규분포를 이용하여 활성화 값들의 분포변화를 관찰한다.
- 시그모이드 함수는 출력이 0 또는 1에 가까워지면 그미분은 0에 다가간다
  - 그래서 데이터가 0과 1에 치우쳐 분포하게 되면 역전파의 기울기 값이 작아지다 사라진다.
  - 이것이 기울기 소실이라 알려진 문제이다.(층을 깊게 하는 딥러닝에서 기울기 소실은 더 심각한문제이다)

```py
  #가중치 표준편차를 0.01로 바꿔 반복
  #w = np.random.randn(node_num, node_num) * 1
  w = np.random.randn(node_num, node_num) * 0.01
```

![vygusfur](../../assets/images/220326.png)

- 0.5 부근에 집중되었다
- 0,1 로 치우치지않았으니 기울기 소실 문제는 없으나 활성화 값들이 치우쳤다는 큰 문제가 있다
  - 뉴런이 많아도 의미가없이 같은 값을 출력하는 "표현력을 제한"하는 문제

```py
  #가중치 표준편차를 0.01로 바꿔 반복
  #w = np.random.randn(node_num, node_num) * 1
  #w = np.random.randn(node_num, node_num) * 0.01
  w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
```

![xavier](../../assets/images/220327.png)

- xavier: 논문에서 권장하는 가중치 초깃값
  - 각 층의 활성화 값들을 광범위하게 분포시킬 목적으로 가중치의 적절한 분포로 앞 계층 노드가 n개라면 표준편차가 1/√n 인 분포를 사용한다는 결론
  - xavier 는 선형을 전제로 이끈 결과(sigmoid, tanh) 반면 ReLU를 이용할때는 He초깃값을 사용을 권장

## 배치 정규화
- 각 층이 활성화를 적당히 퍼뜨리도록 강제 해보는 방법

### 배치 정규화 알고리즘

![batchnorm](../../assets/images/220328.png)

- 2015년 제안된 방법
- 배치 정규화 장점
  - 학습을 빨리 진행할 수 있다.
  - 초기값에 크게 의존하지 않는다
  - 오버피팅을 억제한다
- 각층에서 활성화 값이 적당히 분포되도록 조정하여 배치 정규화 계층을 신경망에 삽입

## 바른 학습
- 오버피팅
  - 주요 원인
    - 매개변수가 많고 표현력이 높은 모델
    - 훈련 데이터가 적음
  - 가중치 감소(weight decay)
    - 가중치가 큰 항목에 큰 페널티를 부과하여 오버피팅을 억제하는 방법
  - 드롭아웃
    - 뉴런을 임의 삭제하면서 학습 하는 방법

## 적절한 하이퍼파라미터 값 찾기
- 하이퍼파라미터
  - 각 층의 뉴런 수, 배치 크기, 매개변수 갱신 시의 학습률과 가중치 감소 등

### 검증데이터
- 하이퍼파라미터 성능평가할 떄는 시험 데이터를 사용해서 안된다.
- 하이퍼파라미터 조정용 데이터를 일반적으로 검증데이터(validation data)라고 부른다.
  - 훈련 데이터: 매개변수 학습
  - 검증 데이터: 하이퍼파라미터 성능 평가
  - 시험 데이터: 신경망의 범용 성능 평가

### 하이퍼파라미터 최적화
- 핵심은 하이퍼파라미터의 최적 값이 존재하는 범위를 조금씩 줄여간다.
- 10<sup>-3</sup> ~ 10<sup>3</sup> 사이 값으로 지정 (로그 스케일)
- 정리하면 다음 순서로 진행한다
  0. 하이퍼파라미터의 값의 범위를 설정
  1. 설정된 범위에서 하이퍼파라미터 값을 무작위로 추출한다.
  2. 1단계에서 샘플링한 하이퍼파라미터 값을 사용하여 학습하고, 검증 데이터로 정확도를 평가한다.(단 에폭은 작게)
  3. 1단계와 2단계를 특정 횟수(100회등) 반복하며, 그 정확도의 결과를 보고 하이퍼파라미터의 범위를 좁힌다.

## 정리
- 매개변수 갱싱 방법은 SGD, 모멘텀,  AdaGrad, Adam 등이 있다.
- 가중치 초깃값을 정하는 방법은 매우 중요하다.
- 가중치의 초깃값으로는 Xavier, He가 효과적이다.
- 배치 정규화를 이용하면 학습을 빠르게 진행하고 초깃값에 영향을 덜받는다
- 오버피팅을 억제하는 정규화 기술로 가중치 감수와 드롭아웃이 있다
- 하이퍼파라미터 값 탐색은 최적 값이 존재할 법한 범위를 점차 좁히면서 하는 것이 효과적이다  