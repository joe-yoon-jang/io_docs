---
layout: post
title: 파운데이션 모델을 넘어서
parent: LLM
has_children: false
nav_order: 887
---

# 파운데이션 모델을 넘어서

## 7.2 사례연구: VQA
- 시각적 질문-답변visual question-answering(VQA): 이미지와 자연어 모두에 대한 이해와 추론이 필요한 어려운 작업

### 텍스트 프로세서: DistillBERT
- DistilBERT: 속도와 메모리 효율성을 위해 최적화된 인기 있는 BERT 모델의 증류(distilled) 버전. 사전 훈련된 모델은 지식 증류(Knowledge Distillation)를 사용하여 더 작고 효율적인 모델로 지식을 전달
    - 지식증류: 머신러닝에서 모델의 지식을 전달하는 방법으로 모델의 크기는 줄이되 중요한 부분을 남기는 방법

### 이미지 프로세서: ViT
- 비전트랜스포머Vision Transformer(ViT): 이미지 이해를 위해 특별히설계된트랜스포머기반 아키텍처. 이 모델은 이미지에서 관련된 특징을 추출하기 위해 셀프 어텐션Self-Attention 메커니즘을 사용.
- 모델이 사전훈련과 동일한이미지 전처리단계를사용 하는 것이 좋다. 이렇게 함으로 모델이 새로운 이미지셋을 더 쉽게 학습할 수 있다.
    - 장점
        1. 사전 훈련과의 일관성: 사전 훈련 중에 사용된 것과 동일한 형식과 분포의 데이터를 사용하면 성능이 향상 되고 수렴 속도가 빨라질 수 있다.
        2. 사전 지식 활용: 모델은 이미 대규모 데이터셋에서 사전 훈련되었기 때문에 이미지에서 의미 있는 특징을 추출하는 방법을 이미 학습. 동일한 전처리 단계를 사용하면 모델이 이 사전 지식을 새 데이터셋 에 효과적으로 적용.
        3. 개선된 일반화: 사전 훈련과 일관된 전처리 단계를 사용 시 새 데이터에 대해 더 잘 일반화될 가능성이 높다.
    - 단점
        1. 제한된 유연성: 동일한 전처리 단계를 재사용하면 모델이 새로운 데이터 분포나 새 데이터셋의 특성에 적 응하는 능력이 제한, 이는 최적의 성능을 위해 다른 전처리 기술이 필요.
        2. 새 데이터와의 불일치: 경우에 따라 새 데이터셋은 기존의 전처리 단계에 잘 맞지 않는 고유한 속성이나 구조를 가질 수 있다. 이로 인해 전처리 단계가 해당 속성이나 구조에 맞게 조정되지 않는다면 성능이 떨어질 수 있다.
        3. 사전 훈련 데이터에 대한 과적합: 동일한 전처리 단계에 지나치게 의존하면 모델이 사전 훈련 데이터의 특 성에 과적합 가능, 이는 새롭고 다양한 데이터셋에 대한 일반화 능력을 감소.

### 텍스트 인코더: GPT-2
- 대량의 텍스트 데이터에 대해 사전 훈련된 오픈 소스 생성 언어 모델. 약 40GB의 데이터로 사전 훈련
- 세 가지 모델의 조합(텍스트 처리를 위한 DistilBERT, 이미지 처리를 위한 ViT, 텍스트 디코딩을 위한 GPT-2 )은 7장 멀티모달 시스템의 기초

### 7.2.2 은닉 상태(딥러닝 은닉층hidden layer 상태) 투영과 융합
- 각각의 모델은 입력을 받으면 출력 텐서를 생성. 반드시 같은 형식으로 되지는 않으며, 차원 수가 다를 수도 있다.
- 불일치를 해결하기 위해, 선형 투영(projection) 계층을 사용 텍스트와 이미지 모델의 출력 텐서를 공유 차원 공간에 투영. 이를 통해 텍스트와 이미지 입력에서 추출된 특징 들을 효과적으로 결합.

### 7.2.3 크로스-어텐션(Cross-Attention): 이것은 무엇이며 왜 중요한가요
- 멀티모달 시스템이 텍스트와 이미지 입력 사이의 상호작용 및 생성하고자 하는 출력 텍스트 학습 메커니즘. 
- 기본 트랜스포머 아키텍처의 핵심 구성 요소
    -  시퀀스-투-시퀀스 모델의 특징
- 셀프-어텐션self-attention 계산과 매우유사
- 크로스-어텐션은 입력 시퀀스(텍스트와 이미지를 모두 입력하므로 결합된 시퀀스)가 키key와 값value 입력

### 어텐션의 쿼리, 키 그리고 값
- 어텐션 메커니즘의 세 가지 내부 구성 요소인 쿼리Query, 키Key, 그리고 값Value
- 쿼리는 어텐션 가중치를 계산 하기 위한 토큰을 나타내며, 키와 값은 시퀀스내의 다른 토큰을나타냅니다. 어텐션점수는쿼 리와 키 사이의 내적을 취하고, 정규화 계수로 조정한 다음 값에 곱하여 가중합을 생성함으로 써 계산
    - 즉 쿼리는 어텐션 점수에 따라 다른 토큰에서 관련 정보를 추출하는 데 사용. 
    - 키는 어떤 토큰이 쿼리와 관련이 있는지 확인하는 데 도움
    - 값은 해당 정보를 제공
- 크로스-어텐션에서 쿼리, 키, 값 행렬은 약간 다른 목적으로 사용
    - 쿼리는 한 모달리티(예: 텍스트)의 출력
    - 키와 값은 다른 모달리티(예: 이미지)의 출력
    - 크로스-어텐션은 다른 모달리티를 처리할 때 한 모달리티의 출력에 얼마나 중요도를 둘 것인지 결정하는 어텐션 점수 계산
    - 어텐션 점수는 쿼리와 키 사이의 내적을 취하고 정규화 계수로 조정하여 계산 

```py
# 텍스트 인코더 모델을 불러오고 해당 설정에서 은닉 상태 크기(은닉 유닛의 수)를 출력
print(AutoModel.from_pretrained(TEXT_ENCODER_MODEL).config.hidden_size)
# 비전 트랜스포머 아키텍처를 사용하여 이미지 인코더 모델을 불러오고
# 해당 설정에서 은닉 상태 크기를 출력 
print(ViTModel.from_pretrained(IMAGE_ENCODER_MODEL).config.hidden_size)
# 인과적 언어 모델링을 위한 디코더 모델을 불러오고
# 해당 설정에서 은닉 상태 크기를 출력 
print(AutoModelForCausalLM.from_pretrained(DECODER_MODEL).config.hidden_size)
# 768 # 768 # 768
```

- 이 경우 모든 모델이 동일한 은닉 상태 크기를 가지고 있어 투영이 필수는 아니다

### 7.2.4 맞춤형 멀티모달 모델

- 새로운 파이토치 모듈을 생성할 때(우리가 지금 하고 있는 것), 정의해야 할 주요 메서드는 생성자(init)
- 이는 세 가지 트랜스포머 모델을 인스턴스화하고, 훈련 속도를 높이기 위 해 계층을 고정시킬 수도 있습니다(이에 대한 자세한 내용은 8장에서 다룰 것입니다). 
- 입력을 받아 출력과 손실 값을 생성하는 forward 메서드가 있습니다(손실은 오류와 같은 것이 며, 값이 낮을수록 더 좋습니다).

- forward 메서드는 다음 입력을 받습니다.
    - input_ids: 텍스트 토큰의 입력 ID를 포함하는 텐서입니다. 이 ID들은 입력 텍스트를 기반으로 토크나
이저에 의해 생성. 텐서의 형태는 [batch_size, sequence_length]
    - attention_mask: input_ids와 같은 형태의 텐서로, 어떤 입력 토큰을 주목해야 하는지(값 1), 무시해야 하는지(값 0 )를 의미. 주로 입력 시퀀스에서 패딩 토큰을 다루는 데 사용.
    - decoder_input_ids: 디코더 토큰의 입력 ID를 포함하는 텐서. 이 ID들은 훈련 중 디코더에 대 한 프롬프트로 사용되는 목표 텍스트를 기반으로 토크나이저에 의해 생성. 훈련 중 텐서의 형태는 [batch_size, target_sequence_length]
    - image_featrues: 배치의 각 샘플에 대한 전처리된 이미지 특징을 포함하는 텐서. 텐서의 형태는 [batch_size, num_features, feature_dimension].
    - labels: 목표 텍스트에 대한 실제 레이블을 포함하는 텐서. 텐서의 형태는 [batch_size, target_sequence_length]. 이 레이블들은 훈련 중 손실을 계산하는 데 사용, 추론 시에는 존재하지 않다.


### 7.2.5 데이터: Visual QA
- Visual QA에서 제공하는 데이터셋
    - 이미지에 대한 개방형 질문과 사람이 주석을 단 답변 쌍이 포함. \시각, 언어, 그리고 약간의 상식을 이 해하는 데 필요한 질문 생성하기 위한 데이터셋

```py
# 주어진 주석과 질문 파일에서 VQA 데이터를 읽어오는 함수
def load_vqa_data(annotations_file, questions_file, images_folder, start_at=None, end_ at=None, max_images=None, max_questions=None):
    # 주석과 질문 JSON 파일 불러오기
    with open(annotations_file, "r") as f:
        annotations_data = json.load(f) 
    with open(questions_file, "r") as f:
        questions_data = json.load(f) 
    data = []
    images_used = defaultdict(int)
    # question_id를 주석 데이터에 매핑하는 딕셔너리 만들기
    annotations_dict = {annotation["question_id"]: annotation for annotation in annotations_data["annotations"]}
    # 지정된 범위의 질문에 대해 반복하기
    for question in tqdm(questions_data["questions"][start_at:end_at]):
        ...
        # 이미지 파일이 존재하고 max_questions 제한에 도달하지 않았는지 확인 
        ...
        # 데이터를 딕셔너리에 추가
        data.append(
        {
            "image_id": image_id,
            "question_id": question_id,
            "question": question["question"],
            "answer": decoder_tokenizer.bos_token + ' ' + annotation["multiple_ choice_
            answer"]+decoder_tokenizer.eos_token, "all_answers": all_answers,
            "image": image,
        } )
        ...
        # max_images 제한에 도달하면 break ...
    return data
# 훈련과 검증 VQA data 읽어오기 
train_data = load_vqa_data(
    "v2_mscoco_train2014_annotations.json", "v2_OpenEnded_mscoco_train2014_questions. json", "train2014",
)
val_data = load_vqa_data(
    "v2_mscoco_val2014_annotations.json", "v2_OpenEnded_mscoco_val2014_questions. json", "val2014"
)
from datasets import Dataset
train_dataset = Dataset.from_dict({key: [item[key] for item in train_data] for key in train_data[0].keys()})

# 나중에 검색하기 위해 데이터셋을 디스크에 저장할 수도 있습니다 
train_dataset.save_to_disk("vqa_train_dataset")

# Hugging Face 데이터셋 생성
val_dataset = Dataset.from_dict({key: [item[key] for item in val_data] for key in val_ data[0].keys()})

# 나중에 검색하기 위해 데이터셋을 디스크에 저장할 수도 있습니다 
val_dataset.save_to_disk("vqa_val_dataset")    
```

### 7.2.6 VQA 훈련 과정

- Hugging Face의 Trainer와 TrainingArguments 객체를 맞춤형 모델과 함께 사용
- 훈련의 목적은 단순히 검증 손실의 감소를 기대. 

```py
# 모델 구성을 정의
DECODER_MODEL = 'gpt2'
TEXT_ENCODER_MODEL = 'distilbert-base-uncased'
IMAGE_ENCODER_MODEL = "facebook/dino-vitb16" # A version of ViT from Facebook

# 지정된 구성으로 MultiModalModel 초기화 
model = MultiModalModel(
    image_encoder_model=IMAGE_ENCODER_MODEL, 
    text_encoder_model=TEXT_ENCODER_MODEL, 
    decoder_model=DECODER_MODEL, 
    freeze='nothing'
)
# 훈련 변수들을 구성 
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    optim='adamw_torch',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=device.type == 'cuda', # 이를 통해 GPU 지원 시스템의 메모리를 절약 
    save_strategy='epoch'
)
# 모델, 훈련 변수, 데이터셋으로 Trainer 초기화 
Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    data_collator=data_collator
)
```

## 7.3 사례 연구: 피드백 기반 강화 학습

- 사람으로부터 또는 자동화된 실시간 피드백을 사용하여 생성된 텍스트를 성능 측정으 로, 또는 모델을 최적화하기 위한 손실함수로 사용할수있다면 어떨까요?바로여기서피드백기반 강화 학습(reinforcement learning from feedback (RLF))
- 즉 인간 피드백 기반 강화 학습reinforcement learning from human feedback(RLHF)과 AI 피드백 기반 강화 학습(reinforcement learning from AI feedback(RLAIF)) 등장

1. 언어 모델의 사전 훈련: 
    - 언어 모델을 사전 훈련하는 것은 모델을 기사, 책, 웹사이트 또는 큐레이션된 데이 터셋과 같은 대규모 텍스트 데이터로 훈련시키는 과정. 
    - 이 단계에서 모델은 일반적인 말뭉 치 또는 특정 작업을 위한 텍스트를 생성 학습. 이 과정은 모델이 텍스트 데이터로부터 문 법, 구문, 그리고 일정 수준의 의미를 학습하는 데 도움. 
    - 사전 훈련하는 동안 사용되는 목적 함수 는 일반적으로 교차 엔트로피 손실. 예측된 토큰 확률과 실제 토큰 확률 사이의 차이를 측정. 
    - 사전 훈련을 통해 모델은 나중에 특정 작업에 맞게 파인튜닝될 수 있는 언어의 기본적인 이해를 습득.
2. 보상 모델 정의(잠재적 훈련): 
    - 언어 모델의 사전 훈련 후, 생성된 텍스트의 품질을 평가할 수 있는 보상 모델 정의
    - 다양한 텍스트 샘플에 대한 순위나 점수와 같이 선호도 데 이터셋을 만드는 데 사용할 수 있는 피드백 수집을 포함
    - 보상 모델은 이러한 선호도를 포착하려고 하며 지도 학습 문제로 훈련될 수 있다
    - 여기서의 목표는 생성된 텍스트를 사람의 피드백에 따른 텍스트 품질을 나타내는 보상 신호(스칼라 값)에 매핑하는 함수를 학습하는 것
3. 강화 학습으로 언어 모델을 파인튜닝하기: 
    - 모델은 텍스트를 생성하고, 보상 모델로부터 피드백을 받으며, 보상 신호에 기반하여 파라미터를 업데이트
    - 목표는 생성된 텍스트가 사람의 선호도와 밀접하게 일치하도록 언어 모델을 최적화
    - 인기 있는 강화 학습 알고리즘에는 근위 정책 최적화Proximal Policy Optimization (PPO )와 신뢰 영역 정책 최적화Trust Region and Proximal policy optimization (TRPO )가 있다
    - 강화 학습을 통한 파인튜닝은 모델이 특정 작업에 적응하고 사람의 가치와 선호도를 더 잘 반영하는 텍스트를 생성

### 7.3.2 보상 모델(Reward Model): 감정과 문법 정확도

- LLM의 출력(우리 경우에는 텍스트 시퀀스)을 입력으로 받아 하나의 스칼라scalar (숫자)로 보상을 피드백
- 피드백은 실제 사람으로부터 오기 때문에, 실행 속도가 매우느릴수있다
- 대안으로, 다른 언어 모델이나 잠재적인 모델출력에 순위를매기고 그 순위를 보상으로 전환하는 더 복잡한 시스템도 있다.
- 7장에서는 미리 구축된 LLM 사용
    - cardiffnlp/twitter-roberta-base-sentiment LLM의 감정 분석: 요약이 중립적인 성격을 띠도록 하는 것 목적, 이 모델에서의 보상은 ‘중립’ 클래스의 로짓 값(로짓 값은 음수 가될수있는데, 이 값이 선호됨)
    - textattack/roberta-base-CoLA LLM의 ‘문법 점수’: 보상은 ‘문법적으로 정확함’ 클래스의 로짓 값
    
```py
from transformers import pipeline
# CoLA6 pipeline 초기화
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base- CoLA")
cola_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer)
# sentiment analysis pipeline 초기화
sentiment_pipeline = pipeline('text-classification', 'cardiffnlp/twitter-roberta- basesentiment')
# 텍스트 목록에 대한 CoLA 점수 얻는 함수 
def get_cola_scores(texts):
    scores = []
    results = cola_pipeline(texts, function_to_apply='none', top_k=None) for result in results:
    for label in result:
        if label['label'] == 'LABEL_1': # 문법이 정확합니다
            scores.append(label['score']) 
            return scores
# 텍스트 목록에 대한 감정 점수 얻는 함수 
def get_sentiment_scores(texts):
    scores = []
    results = sentiment_pipeline(texts, function_to_apply='none', top_k=None) 
    for result in results:
        for label in result:
            if label['label'] == 'LABEL_1': # 중립 감정
                scores.append(label['score']) 
                return scores
texts = [
    'The Eiffel Tower in Paris is the tallest structure in the world, with a height of 1,063 metres',
    'This is a bad book', 
    'this is a bad books'
]
# 텍스트 목록에 대한 CoLA와 중립 감정 점수 얻기 
cola_scores = get_cola_scores(texts) neutral_scores = get_sentiment_scores(texts)
# zip을 사용해서 점수를 결합
transposed_lists = zip(cola_scores, neutral_scores)
# 각 인덱스에 대한 가중 평균을 계산
rewards = [1 * values[0] + 0.5 * values[1] for values in transposed_lists]
# 보상을 텐서 목록으로 변환
rewards = [torch.tensor([_]) for _ in rewards]
##보상은 [2.52644997, -0.453404724, -1.610627412]
```

### 7.3.3 트랜스포머 강화 학습(Transformer Reinforcement Learning (TRL))
- 강화 학습으로 훈련하는 데 사용할 수 있는 오픈 소스 라이브러리
- 이 라이브러리는 우리가 가장 좋아하는 패키지인 Hugging Face의 transformers와 통합되었다
- TRL 라이브러리는 GPT-2와 GPT-Neo와 같은 순수 디코더 모델뿐만 아니라 FLAN-T5 와 같은 시퀀스-투-시퀀스 모델을 지원. - 모든 모델은 근위 정책 최적화Proximal Policy Optimization (PPO )를 사용하여 최적화

### 7.3.4 RLF 훈련 과정

- RLF 파인튜닝 과정

1. 모델의 두 가지 버전을 인스턴스화합니다.
    a. ‘참조’ 모델로서, 기존의 FLAN -T5 모델이며 절대 업데이트되지 않습니다. b.‘현재’ 모델로서, 데이터 배치 처리 후에 파라미터가 업데이트.
2. 데이터 소스(여기서는 Hugging Face의 뉴스 기사 말뭉치)에서 데이터를 가져온다.
3. 두 보상 모델로부터 보상을 계산하고, 두 보상의 가중합으로 단일 스칼라(숫자)로 집계. 
4. 보상을 TRL 패키지에 전달하여 두 가지를 계산합니다.
    a. 보상 시스템에 기초하여 모델을 약간 업데이트하는 방법
    b.생성된 텍스트가 참조 모델에서 생성된 텍스트와 얼마나 차이가 나는지, 즉 두 결과 간의 KL -분산 을 계산. 간단히 말하자면, 두 시퀀스(여기서는 두 텍스 트) 간의 차이를 측정하여 기존 모델의 생성 능력에서 너무 멀어지지 않도록 하는 것이 목표.
5. TRL은 ‘현재’ 모델을 데이터 배치로부터 업데이트하고, 보고 시스템에 로그를 기록. 그리고 1단계에서 다시 시작.