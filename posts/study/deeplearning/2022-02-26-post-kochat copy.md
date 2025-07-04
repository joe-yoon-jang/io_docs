---
title: "Kochat"
layout: post
parent: deeplearning
has_children: false
nav_order: 411
last_modified_at: 2022-02-27T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - 신경망학습
  - chatbot
---

# [Kochat]

- 한국어 전용 챗봇 프레임워크
- 자신만의 딥러닝 챗봇 애플리케이션 제작 지원
-  [RASA]는 개발자가 직접 소스코드를 수정할 수 있기 때문에 다양한 부분을 커스터마이징 할 수 있다 [RASA]를 보고 제작 하게되었다

```py
# 1. 데이터셋 객체 생성
dataset = Dataset(ood=True)

# 2. 임베딩 프로세서 생성
emb = GensimEmbedder(model=embed.FastText())

# 3. 의도(Intent) 분류기 생성
clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),                  
    loss=CenterLoss(dataset.intent_dict)                    
)

# 4. 개체명(Named Entity) 인식기 생성                                                     
rcn = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

# 5. 딥러닝 챗봇 RESTful API 학습 & 빌드
kochat = KochatApi(
    dataset=dataset, 
    embed_processor=(emb, True), 
    intent_classifier=(clf, True),
    entity_recognizer=(rcn, True), 
    scenarios=[
        weather, dust, travel, restaurant
    ]
)

# 6. View 소스파일과 연결                                                                                                        
@kochat.app.route('/')
def index():
    return render_template("index.html")

# 7. 챗봇 애플리케이션 서버 가동                                                          
if __name__ == '__main__':
    kochat.app.template_folder = kochat.root_dir + 'templates'
    kochat.app.static_folder = kochat.root_dir + 'static'
    kochat.app.run(port=8080, host='0.0.0.0')
```

## About Chatbot
### 챗봇의 분류

- 비목적 대화를 위한 Open domain 챗봇
    - 잡담등을 수행하는 챗봇(심심이 등)(Chit-chat)
- 목적 대화를 위한 Close domain 챗봇
    - 금융상담봇, 식당예얏봇 등 Goal oriented 챗봇
- 시리나 빅스비 같은 인공지능 비서, 인공지능 스피커들은 특수 기능과 잡담도 잘해야하므로 2개 도메인 모두 포함되어있다

![3](../../assets/images/about-chatbot.jpg)

### 챗봇의 구현

- 통계 기반과 딥러닝 기반의 챗봇으로 분리한다.
- Kochat 딥러닝 기반의 챗봇

#### Open domain  챗봇
- 대부분 End to End 신경망 기계번역 방식(Seq2Seq)로 구현된다
- Seq2Seq: 한문장을 다른 문장으로 변환/번역 하는 방식 
    - 구글 [Meena] 같은 모델로 사람과 근접한 수준으로 대화가능

### Close domain 챗봇

- Close domain 챗봇은 대부분 Slot Filling 방식으로 구현
- Slot Filling: 미리 기능을 수행할 정보를 담는 슬롯을 먼저 정의한 다음 어떤 슬롯을 선택할지 정하고 슬롯을 채워나가는 방식
    - 인텐트와 엔티티라는 개념이 필요하다

![2](../../assets/images/close-domain-01.jpg)

#### 인텐트(의도) 분류하기: 슬롯 고르기

> 수요일 부산 날씨 어떨까?

1. 날씨 API 슬롯 지역 날짜
2. 미세먼지 API 슬롯
3. 맛집 API 슬롯
4. 여행지 API 슬롯

- 4가지 정보제공 기능 중 어떤 기능을 실행해야하는지 알아채야한다
- 이를 인텐트 분류라한다

#### 폴백 검출하기: 모르겠으면 모른다고 말하기

- 위 4가지 발화의도 안에서만 말할 것이라는 보장은 없다
- 입력 단어들의 임베딩인 문장 벡터와 기존 데이터셋에 있는 문장 벡터들의 Cosine 유사도를 비교한다
- 인접 클래스와의 각도가 임계치 이상이면 Fallback 이고 그렇지 않으면 인접 클래스로 데이터 샘플을 분류
![0](../../assets/images/fallback.png)

#### 엔티티(개체명) 인식하기: 슬롯 채우기

- 개체명인식 (Named Entity Recognition)
  - 인텐트 분류 이후 API를 호출하기 위한 파라미터를 찾는다 
  - 날씨 API의 실행을 위한 파라미터가 "지역", "날씨"면 사용자의 입력 문장에서 "지역"에 관련된 정보와 "날씨"에 관련된 정보를 찾아 해당 슬롯을 채운다
  - 슬롯을 채우고 API로 외부로부터 정보를 제공받는다 
  - API로부터 결과가 도착하면, 미리 만들어둔 템플릿 문장에 해당 실행 결과를 삽입하여 대답을 만들고, 사용자에게 response 한다

![1](../../assets/images/response.jpg)

## 데이터셋

- 학습시킬 데이터셋 추가
- Intent와 Entity 데이터셋 필요
- intent
  - 파일로 구분 weather.csv 등
- Entity
  - 라벨로 구분
- 파일 샘플
    - weather.csv
    - travel.csv
- [BIOES] 태깅을 사용하여 라벨링
```
question,label
날씨 알려주세요,O O
월요일 인제 비오니,S-DATE S-LOCATION O
군산 날씨 추울까 정말,S-LOCATION O O O
곡성 비올까,S-LOCATION O
내일 단양 눈 오겠지 아마,S-DATE S-LOCATION O O O
강원도 춘천 가는데 오늘 날씨 알려줘,B-LOCATION E-LOCATION O S-DATE O O
전북 군산 가는데 화요일 날씨 알려줄래,B-LOCATION E-LOCATION O S-DATE O O
제주 서귀포 가려는데 화요일 날씨 알려줘,B-LOCATION E-LOCATION O S-DATE O O
오늘 제주도 날씨 알려줘,S-DATE S-LOCATION O O
... (생략)
```

```
question,label
어디 관광지 가겠냐,O O O
파주 유명한 공연장 알려줘,S-LOCATION O S-PLACE O
창원 여행 갈만한 바다,S-LOCATION O O S-PLACE
평택 갈만한 스키장 여행 해보고 싶네,S-LOCATION O S-PLACE O O O
제주도 템플스테이 여행 갈 데 추천해 줘,S-LOCATION S-PLACE O O O O O
전주 가까운 바다 관광지 보여줘 봐요,S-LOCATION O S-PLACE O O O
용인 가까운 축구장 어딨어,S-LOCATION O S-PLACE O
붐비는 관광지,O O
청주 가을 풍경 예쁜 산 가보고 싶어,S-LOCATION S-DATE O O S-PLACE O O
... (생략)
```

#### OOD 데이터셋

- Out of distribution
  - 분포 외 데이터셋을 의미한다
  - OOD 데이터셋이 없어도 이용하는데 문제는 없다
  - 단 귀찮은 몇몇 부분을 효과적으로 자동화 할 수 있다(주로 Fallback Detection threshold)



[kochat]: https://github.com/hyunwoongko/kochat
[rasa]: https://rasa.com/
[Meena]:https://ai.googleblog.com/2020/01/towards-conversational-agent-that-can.html
[BIOES]: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)