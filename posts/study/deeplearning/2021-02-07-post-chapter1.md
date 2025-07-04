---
title: "헬로 파이썬"
layout: post
parent: deeplearning
has_children: false
nav_order: 401
last_modified_at: 2021-02-08T16:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
---


# 헬로 파이썬

## 파이썬이란
- 유명 딥러닝 프레임워크 카페, 텐서플로 체이너 테아노 등
- 책은 아나콘다로 https://anaconda.com/distribution

## 넘파이

딥러닝 구현 시 배열이나 행렬 계산등 메서드 제공

```py
import numpy as np
```
- element-wise 원소별
- element-wise product 원소별 곱셈

```python
A = np.array([[1,2],[3,4]])
A.shape
# (2,2)
A.dtype
# dtype('int64')
```

- 행렬 형상은 shape, 자료형은 dtype

### 브로드캐스트

> 넘파이의 형상이 다른 배열끼리 계산 [[1,2],[3,4]] * 10 은 [[1,2],[3,4]] * [[10,10],[10,10]] 으로 확대하여 계산하고 이를 브로드캐스트라고함

> [[1,2],[3,4]] * [10,20] 은 [[1,2],[3,4]] * [[10,20],[10,20]] = [[10,40],[30,80]]

### matplotlib

데이터 시각화 라이브러리