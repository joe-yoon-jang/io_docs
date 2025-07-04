---
title: "ch5"
layout: post
parent: deepdive
has_children: false
nav_order: 301
last_modified_at: 2024-02-01T18:20:02-05:00
---
# 17 생성자 함수에 의한 객체 생성

- 다양한 객체 생성 방식 중에서 생성자 함수를 사용하여 객체를 생성하는 방식

## 17.1 Object 생성자 함수

- new 연산자와 함께 Object 생성자 함수를 호출하여 빈 객체를 생성하여 반환
- 빈객체에 프로퍼티 또는 메서드를 추가하여 객체를 완성할 수 있다

```jsx
// 빈 객체 생성
const person = new Object();

// 프로퍼티 추가
person.name = 'Lee';
person.sayHello = function(){
  console.log('hi my name is ' + this.name);
};

console.log(person); // {name:'Lee', sayHello: f}
person.sayHello(); // hi myname is Lee
```

- 생성자 함수(constructor)란 new 연산자와 함께 호출하여 객체를 생성하는 함수다
- 자바 스크립트는 Object  생성자 함수 이외에도 String, Number, Boolean, Function, Array, Date, RegExp, Promise 등의 빌트인 생성자 함수를 제공한다

```jsx
//생성자 함수에 의한 객체 생성
const strObj = new Streing('Lee');
typeof strObj; // object
console.log(strObj); // string {"Lee"}

const numObj = new Number(123);
typeof numObj; // object
console.log(numObj); // Number{123}

const boolObj = new Boolean(true);
typeof boolObj; //object
console.log(boolObj); // Boolean {true}

const func = new Function('x', 'return x * x');
typeof func; // function
console.dir(func); // f anonymous(x)

consot arr = new Array(1,2,3);
typeof arr; // object
conosole.log(arr); // [1,2,3]

const regExp = new RegExp(/ab+c/i);
typeof regExp; // object
console.log(regExp); // /ab+c/i

const date = new Date();
type date; //object
console.log(date); // Mon May 04 2020 08:36:33 GMT +0900 (대한민국 표준시)
```

## 17.2 생성자 함수

### 17.2.1 객체 리터럴에 의한 객체 생성 방식의 문제점

- 단 하나의 객체만 생성하여 동일한 프로퍼티를 갖는 객체를 여러 개 생성해야 하는 경우 매번 같은 프로퍼티를 기술해야 하기 때문에 비효율적이다

```jsx
const circle1 = {
  ratidus: 5,
  getDiameter(){
    return 2 * this.radius;
  }
}

console.log(circle.getDiameter()); // 10
const circle2 = {
  radius: 10,
  getDiameter(){
    return 2 * this.radius;
  }
}
console.log(circle2.getDiameter()); // 20
```

- 객체는 프로퍼티를 통해 객체 고유의 상태(state)를 표현한다
- 메서드를 통해 상태 데이터인 프로퍼티를 참조하고 조작하는 동작(behavior)을 표현한다
- 따라서 프로퍼티는 객체마다 값이 다를 수 있지만 메서드는 내용이 동일한 경우가 일반적이다
    - 즉 많이 생성해야하는 경우 문제

### 17.2.2 생성자 함수에 의한 객체 생성 방식의 장점

- 객체를 생성하기 위한 템플릿 처럼 생성자 함수를 사용하여 프로퍼티 구조가 동일한 객체 여러개를 간편하게 생성할 수 있다

```jsx
//생성자 함수
function Circle(radius){
  this.radius = radius;
  this.getDiameter = function(){
    return 2 * this.radius;
  }
}

//인스턴스 생성
const circle1 = new Circle(5); // 반지름이 5인 Circle 객체를 생성
const circle2 = new Circle(10); // 반지름이 10인  Circle 객체를 생성

console.log(circle1.getDiameter()); // 10
console.log(circle2.getDiameter()); // 20
```

- this는 객체 자신의 프로퍼티나 메서드를 참조하기 위한 자기 참조 변수(self referencing variable)다.  this바인딩은 함수 호출 방식에 따라 동적으로 결정된다. (22장에서 추가 설명)
    
    
    | 함수 호출 방식 | this가 가리키는 값(this 바인딩) |
    | --- | --- |
    | 일반 함수로서 호출 | 전역 객체 |
    | 메서드로서 호출 | 메서드를 호출한 객체 (마침표 앞의 객체) |
    | 생성자 함수로서 호출 | 생성자 함수가(미래에)생성할 인스턴스 |
    
    ```jsx
    function foo(){
    	console.log(this);
    }
    // 일반적인 함수로서 호출
    // 전역 객체는 브라우저 환경에서는 window, Node.js 환경에서는 global을 가리킨다.
    foo(); //window
    const obj = {foo};
    // 메서드로서 호출
    obj.foo(); // obj
    //생성자 함수로서 호출
    const inst = new foo(); // inst
    ```
    
- 생성자 함수는 자바와 같은 클래스 기반 객체지향 언어의 생성자와는 다르게 그형식이 정해져 있는 것이 아니라 일반 함수와 동일한 방법으로 생성자 함수를 정의하고 new 연산자와 함께 호출하면 해당 함수는 생성자 함수로 동작한다.

```jsx
// new 와 함께 호출하지 않으면 생성자 함수로 동작하지 않는다 즉 일반함수로 호출된다
const circle3 = Circle(15);
console.log(circle3); //undefined
// 일반 함수 Circle  내의 this는 전역 객체를 가리킨다
console.log(radius); // 15
```

### 17.2.3 생성자 함수의 인스턴스 생성 과정

- 생성자 역할
    - 프로퍼티 구조가 동일한 인스턴스를 생성
        - 생성자는 반환값이 없어도 암묵적으로 인스턴스를 생성하고 반환한다
    - 생성된 인스턴스를 초기화(인스턴스 프로퍼티 추가 및 초기값 할당)

```jsx
//생성자 함수
function Circle(radius){
  // 인스턴스 초기화
  this.radius = radius;
  this.getDiameter = function(){
    return 2 * this.radius;
  }
}

// 인스턴스 생성
const circle1 = new Circle(5); //  반지름 5인 Circle 객체 생성
```

1. 인스턴스 생성과 this 바인딩
- 암묵적으로 빈 객체가 생성된다(생성자 함수가 생성한 인스턴스)
- 인스턴스는 this에 바인딩된다.
    - 생성자 함수 내부의 this가 생성할 인스턴스를 가리키는 이유
- 이는 코드가 한 줄씩 실행되는 런타임 이전에 실행된다
1. 인스턴스 초기화
- this에 바인딩되어 있는 인스턴스를 초기화 한다
1. 인스턴스 반환
- this가 아닌 다른 객체를 명시적으로 반환하면 this가 반환되지 못하고 return문에 명시한 객체가 반환된다.

```jsx

function Circle(radius){
  // 1. 암묵적으로 인스턴스가 생성되고 this에 바인딩된다
  console.log(this); // Circle {}
  // 2. this에 바인딩되어 있는 인스턴스를 초기화한다
  this.radius = radius;
  this.getDiameter = function(){
    return 2 * this.radius;
  }
  // 3. 완성된 인스턴스가 바인딩된 this가 암묵적으로 반환된다
}
// 인스턴스 생성. Circle 생성자 함수는 암묵적으로 this를 반환한다
const circle = new Circle(1);
console.log(circle); // Circle {radius1, getDiameter: f}

function Circle(radius){
  // 1. 암묵적으로 인스턴스가 생성되고 this에 바인딩된다
  console.log(this); // Circle {}
  // 2. this에 바인딩되어 있는 인스턴스를 초기화한다
  this.radius = radius;
  this.getDiameter = function(){
    return 2 * this.radius;
  }
  // 3. 완성된 인스턴스가 바인딩된 this가 암묵적으로 반환된다
  // 명시적으로 객체를 반환하면 암묵적 this반환은 무시
	return {};
}

const circle = new Circle(1);
console.log(circle); // {}
```

### 17.2.4 내부 메서드 [[Call]]과 [[Construct]]

- 함수는 객체이므로 일반 객체(ordinary object)와 동일하게 동작 가능하다
- 함수 객체는 일반객체가 가지고 있는 내부 슬롯과 내부 메서드를 모두 가지고있다
- 함수는 객체이지만 일반 객체와 다르게 호출할 수 있다
    - [[Environment]], [[Formal Parameters]] 등의 함수 객체만을 위한 내부 슬롯과  [[Call]], [[Construct]] 같은 내부 메서드를 추가로 가지고 있다

```jsx
// 함수는 객체다
function foo(){}
//프로퍼티 소유 가능하다
foo.prop = 10;
// 메서드를 소유할 수 있다
foo.method = function(){
  console.log(this.prop);
}
foo.method(); // 10
// 일반적인 함수로서 호출 : [[Call]]이 호출
foo();
// 생성자 함수로서 호출 : [[Construct]]가 호출된다
new foo();
```

- 내부 메서드 Call을 갖는 함수 객체를 callable이라 한다
    - 호출할 수 있는 객체, 즉 함수를 말한다
    - 함수 객체는 반드시 callable이다. 즉 모든 함수 객체는 내부 메서드 [[Call]]을 갖고 있다
        - 단 반드시 [[Construct]]를 갖는 것은 아니다
- 내부 메서드 Construct를 갖는 함수 객체를 constructor,  갖지 않는 함수 객체를 non-constructor라고 한다
    - 생성자 함수로서 호출할 수 있는 함수, 객체를 생성자 함수로서 호출할 수 없는 함수를 의미한다

### 17.2.5 construct와 non-construct의 구분

- 함수 객체를 생성할 떄 함수 정의 방식에 따라 함수를 constructor와 non-construct로 구분
    - constructor: 함수 선언문, 함수 표현식, 클래스(클래스도 함수다)
    - non-constructor: 메서드(ES6 메서드 축약 표현), 화살표 함수
- 함수를 프로퍼티 값으로 사용하면 일반적으로 메서드로 통칭한다. 하지만 ECMAScript 사양에서 메서드란 ES6의 메서드 축약 표현만을 의미한다
    - 즉 함수가 어디에 할당되어 있는지에 따라 메서드인지를 판단하는 것이 아니라 함수 정의 방식에 따라 constructor와 non-constructor를 구분한다
    - 예제 일반 함수, 즉 함수 선언문과 함수 표현식으로 정의된 함수만이 constructor  이고 ES6 화살표 함수와 메서드 축약 표현으로 정의된 함수는 non이다

```jsx
//일반 함수 정의 : 함수 선언문, 함수 표현식
function foo() {}
const bar = function(){};

// 프로퍼티 x의 값으로 할당된 것은 일반 함수로 정의된 함수다. 이는 메서드로 인정하지 않는다
const baz = {
  x: function(){}
};

// 일반 함수로 정의된 함수만이 constructor다
new foo(); // -> foo{}
new bar(); // -> bar{}
new baz.x(); // -> x{}

// 화살표 함수 정의
const arrow = () => {};
new arrow(); // TypeError: arrow is not a constructor

// 메서드 정의 : ES6의 메서드 축약 표현만 메서드로 인정한다
const obj = {
  x() {}
};

new obj.x(); // typeError : obj.x is not a constructor

//일반 함수로서 호출
//[[call]] 호출 모든 함수 객체는 [call]이 구현되어 있다
foo();
//생성자 함수로서 호출
// [[Construct]]가 호출된다 없으면 에러가 발생
new foo();
```

### 17.2.6 new 연산자

- 일반 함수와 생성자 함수에 특별한 형식적 차이는 없다
- new 연산자와 함께 함수를 호출하면 생성자 함수로 동작하고 Call 이 아닌 Construct가 호출된다

```jsx
function add(x,y){
  return x + y;
}

let inst = new add();
// 생성자 함수로서 정의하지 않은 일반 함수를 new 연산자와 함께 호출
// 함수가 객체를 반환하지 않았으므로 반환문이 무시되고 빈객체가 생성되어 반환된다
console.log(inst); // {}

function createUser(name, role){
  return { name, role};
}
inst = new createUser('Lee', 'admin');
console.log(inst); // {name:'Lee', role:'admin'}
```

### 17.2.7 new.target

- 생성자 함수가 new 없이 호출되는 것을 방하기위해 파스칼 케이스 컨벤션을 사용한다 하더라도 실수가 발생할수 있다
    - 이런 위험을 방지하기 위해 ES6에서 new.target을 지원한다
    - new 연산자와 함께 생성자 함수로서 호출되었는지 확인할 수 있다
        - new 와 함께 사용되었으면 new.target은 함수 자신을 가리키고 없으면 undefined다

```jsx
function Circle(radius){
  // 이 함수가 new 연산자와 함께 호출되지 않았다면 undefined
  if(!new.target){
    return new Circle(radius);
  }
  this.radius = radius;
  this.getDiameter = function(){
    return 2 * this.radius;
  }
}

// 스코프 세이프 생성자 패턴
function Circle(radius){
  // 이 함수가 new 연산자와 함께 호출되지 않았다면 이 시점의 this는 전역 객체 window를 가리킨다
  if(!(this instanceof Circle)){
    return new Circle(radius);
  }
}
```

- 대부분 빌트인 생성자 함수는 new 와 함께 호출되었는지 확인한 후 적절한 값을 반환한다.
- Object나 Function 생성자 함수는 new 연산자 없이 호출해도 new연산자와 동일하게 동작한다
- 그러나 String, Number, Boolean 생성자 함수는 new 없이 호출하면 값을 반환하고 new와 사용하면 객체를 생성하여 반환한다

```jsx
// 동일하게 객체 반환
let obj = new Object();
obj = Object();
obj = new Function('x', 'return x ** x');
obj = Function('x', 'return x ** x');
// 값 반환
const str = String(123);
console.log(str, typeof str); // 123 string
```