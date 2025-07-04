---
title: 표현식과 문"
layout: post
parent: deeplearning
has_children: false
nav_order: 410
last_modified_at: 2021-12-26T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - GAN
  - pytorch
---

# 경쟁하며 학습하는 GAN

## GAN 기초

- 적대적 생성 신경망
- GAN은 생성(generative)을 하는 모델
- GAN은 적대적(adversarial)으로 학습. 가짜 이미지를 생성하는 생성자(generator)와 판별자(discriminator)가 번갈아 학습하며 경쟁적 학습 진행
- GAN은 인공싱경망 모델
- 비지도학습 방식

![0](../../assets/images/3333.png)

- 생성자와 판별자는 지폐 위조범과 경찰관 비유를 흔히 한다
  - 지폐 위조범(생성자)와 경찰(판별자)는 위조지폐와 진짜를 감별하기위해 노력한다.
  - 이런 경쟁구도속에서 위조범과 경찰의 능력이 발전한다


## GAN으로 새로운 패션 아이템 생성하기

### 생성자와 판별자 구현

- Sequential 클래스 : 신경망을 이루는 각 층에서 수행할 연산들을 입력받아 차례대로 실행하는 역할
    - 파이토치 모듈 init과 forward 함수를 동시 정의와 같다
- 생성자는 실제 데이터와 비슷한 가짜 데이터를 만들어내는 신경망
    - 정규분포로부터 뽑은 64차원 무작위 텐서를 입력받아 행렬곱(Linear)과 황성화 함수(ReLu, Tanh) 연산을 실행
    

```py
# 생성자 Generator
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),    
    nn.Linear(256, 256),
    nn.ReLU(),        
    nn.Linear(256, 784),
    nn.Tanh()
)
```    
- 판별자
    - 이미지 크기인 784 차원의 텐서를 입력받는다
    - 784차원의 텐서가 생성자가 만든 가짜 이미지인지 실제 이미지인 구분하는 분류 모델
    - ReLU가 아닌 Leaky ReLU 활성화 함수를 사용한다
        - 양의 기울기만 전달하는 ReLU와 다르게 약간의 음의 기울기도 다음층에 전달
        
```py
# 판별자 Discriminator
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)
```        

 ### GAN 학습 구현
 
 - to() 함수로 모델을 학습에 쓰일 장치로 보낸다
 - 생성자와 판별자 학습에 쓰일 오차함수와 최적화 알고리즘도 각각 정의해준다
 - 레이블이 가짜 진자 2가지 이므로 이진 교체 엔트롶피를 사용하고 Adam 최적화 함수를 이용해 학습한다

```py
D = D.to(DEVICE)
G = G.to(DEVICE)

criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
``` 

- GAN을 학습시키는 반복문

```py
total_step = len(train_loader)
for epoch in range(EPOCHS):
    print('epoch', epoch)
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
```

- 생성자가 만든 데이터는 가짜 레이블을 부여
- Fashion MNIST 데이터넷 진짜와 판별자 신경망에 입력한다
- real_labels 텐서는 ones() 함수로 1로만 이루어진 텐서를 생성
- fake_labes는 zeros()로 0으로만 이루어진 텐서를 생성

```py
real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
```

- 판별자는 이미지를 보고 진짜를 구분하는 학습
- 실제 이미지를 판별자 신경망에 입력해 결괏값과 진짜 레이블 사이의 오차를 계산

```py
outputs = D(images)
d_loss_real = criterion(outputs, real_labels)
real_score = outputs
```
- 생성자의 동작을 정의
- 정규분포로부터 생성한 무작위 텐서를 입력받아 실제 이미지와 차원이 같은 텐서 배출

```py
z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
fake_images = G(z)
```
- 판별자가 가짜 이미지를 가짜로 인식하는지 알아보기 위해 생성자가 만들어낸 fake_images를 판별자에 입력
- 결괏값과 가짜 레이블 사이의 오차를 계산


```py
outputs = D(fake_images)
d_loss_fake = criterion(outputs, fake_labels)
fake_score = outputs
```
- 실제 데이터와 가짜 데이터를 가지고 낸 오차를 더해줌으로 판별자 신경망의 전체 오차를 계산한다
- 역전파 알고리즘과 경사하강법을 통하여 판별자 신경망을 학습시킨다
- zero_grad() 함수로 생성자와 판별자의 기울기를 초기화한다

```py
d_loss = d_loss_real + d_loss_fake
d_optimizer.zero_grad()
g_optimizer.zero_grad()
d_loss.backward()
d_optimizer.step()
```
- 판별자가 학습을 통해 성장했다
- 이제 생성자 학습시킬 차례이다
- 생성자의 결과물을 다시 판별자에 입력시켜 real_labels사이의 오차를 최소화하는 식으로 학습을 진행


## cGAN 으로 생성 제어하기

- 이미지 무작위 생성외에 사용자가 원하는 이미지를 생성하는 기능 제공
- 이미지 생성 과정에서 생성하고픈 레이블 정보를 추가로 넣어 원하는 이미지가 나오게끔 모델을 수정한다

### cGAN으로 원하는 이미지 생성하기

- 생성자와 판별자에 레이블 정보가 들어간다
- 앞선 GAN은 여러 종류의 패션아이템 중 무엇을 생성해라 라는 지시 로직이 없다
- 즉 GAN은 사용자가 원하는 패션아이템을 생성하는 능력이 없다
- 이를 보완해서 출력할 아이템의 종류를 입력받아 이미지를 생성하는 모델이 조건부 GAN(conditional GAN)이다

![0](../../assets/images/cgan.png)

- 기본적으로 GAN 모델과 동일하다
- C(레이블) 정보가 생성자와 판별자에 들어간다

### 조건부 생성자와 판별자

- 생성자와 판별자가 하나의 입력이 아닌 레이블정보까지 2가지 입력을 받는다
- 무작위 텐서(z)의 크기를 100으로 정한다
- 10은 레이블에 관한정보로 110을 입력받는다

```py
# 생성자 (G)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    def forward(self, z, labels):
        c = self.embed(labels)
        x = torch.cat([z, c], 1)
        return self.model(x)
```
- inplace=True설정은 입력을 복사하지 않고 바로 조작한다는 의미(메모리 사용을 효율적으로만드나 backward 가 불가능하다)
- 파이토치 cat 함수는 두 벡터를 이어붙이는 연산을 실행한다


```py

# 판별자 (D)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        c = self.embed(labels)
        x = torch.cat([x, c], 1)
        return self.model(x)
```
- 판별자에게도 레이블정보를 전달하기 위해 이미지 크기에 10을더해준다 (784 + 10)


### cGAN 학습 구현

```py

# 모델 인스턴스를 만들고 모델의 가중치를 지정한 장치로 보내기
D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)

# 이진 교차 엔트로피 함수와
# 생성자와 판별자를 최적화할 Adam 모듈
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr =0.0002)
g_optimizer = optim.Adam(G.parameters(), lr =0.0002)
```

- 모델의 인스턴스를 만든다
- 모델의 가중치를 지정한 장치로 보낸다

```py

total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
        
        # '진짜'와 '가짜' 레이블 생성
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        # 판별자가 진짜 이미지를 진짜로 인식하는 오차 계산 (데이터셋 레이블 입력)
        labels = labels.to(DEVICE)
        outputs = D(images, labels)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
    
        # 무작위 텐서와 무작위 레이블을 생성자에 입력해 가짜 이미지 생성
        z = torch.randn(BATCH_SIZE, 100).to(DEVICE)
        g_label = torch.randint(0, 10, (BATCH_SIZE,)).to(DEVICE)
        fake_images = G(z, g_label)
        
        # 판별자가 가짜 이미지를 가짜로 인식하는 오차 계산
        outputs = D(fake_images, g_label)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 판별자의 오차 계산
        d_loss = d_loss_real + d_loss_fake
        
        # 역전파 알고리즘으로 판별자 모델의 학습을 진행
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # 생성자가 판별자를 속였는지에 대한 오차 계산(무작위 레이블 입력)
        fake_images = G(z, g_label)
        outputs = D(fake_images, g_label)
        g_loss = criterion(outputs, real_labels)

        # 역전파 알고리즘으로 생성자 모델의 학습을 진행
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

1. labels 표기를 해주고 진짜와 가짜 레이블을 만든다
2. 오차를 계산하고 데이터셋의 레이블을 입력해 판별자가 이미지와 레이블의 관계를 학습하게 한다
3. 무작위 텐서를 생성하는데 torch.randint로 0과 10사이의 값을 가진 '배치x1'크기의 텐서를  g_label이라고 하나만든다
4. z와 g_label을 생성자에 입력해 g_label과 생성물의 관계성을 학습하도록 한다
5. 이렇게 만들어진 가짜 이미지를 판별자에게 입력하여 가짜라고 인식하는 오차를 계산한다
6. 진짜와 가짜 이미지 그리고 각각의 레이블을 보고 계산한 판별자의 총 오차를 계산하고 역전파 알고리즘으로 판별자의 학습을 진행한다
7. 생성자도 아까 생성해둔 z와 g_label로 다시 이미지를 생성하고 판별자를 속이는지에 대한 오차를 계산하고 이때는 생성자와 판별자 둘다 g_label을 두번쨰 인자로받는다
