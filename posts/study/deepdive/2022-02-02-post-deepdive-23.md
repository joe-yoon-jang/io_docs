---
title: "ch23"
layout: post
parent: deepdive
has_children: false
nav_order: 303
last_modified_at: 2024-02-01T18:20:02-05:00
---

# 22 this

생성일: February 2, 2022 9:14 PM

## 22.1 this 키워드

- 객체는 상태를 나타내는 프로퍼티와 동작을 나타내는 메서드를 논리적인 단위로 묶은 복합적인 자료구조
- 메서드가 자신이 속한 객체의 상태, 즉 프로퍼티를 참조하고 변경할 수 있어야 한다. 이때 메서드가 자신이 속한 객체의 프로퍼티를 참조하려면 먼저 자신이 속한 객체를 가리키는 식별자를 참조할 수 있어야 한다

```jsx
const circle = {
  // 프로퍼티: 객체 고유의 상태 데이터
  radius: 5,
  // 메서드: 상태 데이터를 참조하고 조작하는 동작
  getDiameter(){
    // 이 메서드가 자신이 속한 객체의 프로퍼티나 다른 메서드를 참조 하려면 
    // 자신이 속한 객체인 circle을 참조할 수 있어야 한다
    return 2 * circle.radius;
  }
}

console.log(circle.getDiameter()); // 10
```

- getDiameter 메서드 내에서 메서드 자신이 속한 객체를 가리키는 식별자 circle을 참조하고 있다. 표현식이 평가되는 시점은 메서드가 호출되어 실행되는 시점이다
- 따라서 getDIameter 메서드가 호출되는 시점에는 이미 객체 리터럴의 평가가 완료되어 객체가 생성되어있고 circle 식별자에 객체가 할당된 이후다

```jsx
function Circle(radius){
  // 이 시점에서는 생성자 함수 자신이 생성할 인스턴스를 가리키는 식별자를 알 수 없다
  ????.radius = radius;
}
Circle.prototype.getDiameter = function(){
  // 이 시점에서는 생성자 함수 자신이 생성할 인스턴스를 가리키는 식별자를 알 수 없다
  return 2 * ????.radius;
}

// 생성자 함수로 인스턴스를 생성하려면 먼저 생성자 함수를 정의해야한다
const circle = new Circle(5);
```

- 생성자 함수
    - 내부에서 프로퍼티 또는 메서드를 추가하기 위해 자신이 생성할 인스턴스를 참조할 수 있어야 한다
    - 그러나 참조할 인스턴스는 new 연산자와 함께 생성자 함수를 호출해야 생성된다
    - 즉 생성자 함수로 인스턴스를 참조하기 위해서는 먼저 생성자 함수가 존재해야 한다
- this
    - 자신이 속한 객체 또는 자신이 생성할 인스턴스를 가리키는 자기 참조 변수다
    - 자바스크립트 엔진에 의해 암묵적으로 생성되며 코드 어디서든 참조할 수 있다
    - 함수를 호출하면 arguments 객체와 this가 암묵적으로 함수 내부에 전달된다
    - 함수 내부에서 arguments 객체를 지역 변수처럼 사용할 수 있는 것처럼 this도 가능하다
    - this가 가리키는 값, 즉 this 바인딩은 함수 호출 방식에 의해 동적으로 결정된다

```jsx
// 객체 리터럴
const circle = {
  radius: 5,
  getDiameter(){
    // this는 메서드를 호출한 객체를 가리킨다
    return 2 * this.radius;
  }
}

console.log(circle.getDiameter()); // 10

// 생성자 함수
function Circle(radius){
  // this는 생성자 함수가 생성할 인스턴스를 가리킨다
  this.radius = radius;
}
Circle.prototype.getDiameter = function(){
  // this는 생성자 함수가 생성할 인스턴스를 가리킨다.
  return 2 * this.radius;
}

// 인스턴스 생성
const circle = new Circle(5);
```

- this가 가리키는 객체는 상황에 따라 변한다
    - 자바, C++같은 클래스 기반 언어에서 this는 언제나 클래스가 생성하는 인스턴스이다
    - 자바스크립트의 this는 함수가 호출되는 방식에 따라 this에 바인딩될 값을 동적으로 결정

```jsx
console.log(this); // window

function square(number){
  console.log(this); // window
  return number * number;
}

const person = {
  name: 'Lee',
  getName(){
    // 메서드 내부에서 this는 메서드를 호출한 객체를 가리킨다
    console.log(this);
    return this.name;
  }
}
function Person(name){
  // 생성자 함수 내부에서 this는 생성자 함수가 생성할 인스턴스
  this.name = name;
}
```

- strict mode에서 this
    - 일반 함수 내부의 this에는 undefined가 바인딩

## 22.2 함수 호출 방식과 this 바인딩

- this 바인딩은 함수 호출 방식, 즉 함수가 어떻게 호출되었는지에 따라 동적으로 결정
    1. 일반 함수 호출
        - this는 전역 객체 window
    2. 메서드 호출
        - 내부의 this는 메서드를 호출한 객체 obj를 가리킨다
    3. 생성자 함수 호출
        - 생성한 인스턴스를 가리킨다
    4. Function.prototype.apply/call/bind 메서드에 의한 간접 호출
        - 내부의 this는 인수에 의해 결절된다

```jsx
const foo = function(){
  console.dir(this);
};
// Function.prototype.apply/call/bind 메서드에 의한 간접 호출
const bar = {name: 'bar'};
foo.call(bar); // bar
foo.apply(bar); // bar
foo.bind(bar)(); // bar
```

### 22.2.1 일반 함수 호출

- 기본적으로 this는 전역 객체(global object)가 바인딩

```jsx
function foo(){
  console.log(this); //window
  function bar(){
    console.log(this); // window
  }
  bar();
}
foo();
```

- 전역은 물론 중첩 함수를 일반 함수로 호출하면 함수 내부의 this에는 전역 객체가 바인딩된다
- strict mode 는 undefined
- 메서드 내에서 정의한 중첩 함수도 일반 함수로 호출되면 this에는 전역 객체가 바인딩
- 콜백 함수가 일반 함수로 호출된다면 콜백 함수 내부의 this에도  전역 객체가 바인딩된다
- 어떠한 함수라도 일반 함수로 호출되면 this에 전역 객체가 바인딩된다
- 외부 함수인 메서드와 중첩 함수 또는 콜백 함수의 this가 일치하지 않는다는 것은 헬퍼 함수로 동작하기 어렵게 만든다
    - 아래 예제의 경우 메서드 내부에서 setTimeout 함수에 전달된 콜백 함수의 this를 할당하여 전달하는 방식으로 해결
    - 위 외에도 this를 명시적으로 바인딩할 수 있는 Function.prototype.apply, Function.prototype.call, Function.prototype.bind 메서드를 제공한다

```jsx
function foo(){
  'use strict';
  console.log(this); // undefiend
  function bar(){
    console.log(this); // undefined
  }
  bar();
}
foo();
```

```jsx
var value = 1;
//const value = 1; const 키워드로 선언한 전역변수 value 는 전역 객체의 프로퍼티가 아니다

const obj = {
  value: 100,
  foo(){
    console.log(this); // {value: 100, foo: f}
    console.log(this.value); // 100
    function bar(){
      console.log(this); //window
      console.log(this.value); // 1
    }
    bar();
  }
}
obj.foo();
```

```jsx
var value = 1;

const obj = {
  value: 100,
  foo(){
    const that = this;
    setTimeout(function(){
      // 콜백 함수 내부에서 this 대신 that을 참조한다
      console.log(that.value); // 100
    }, 100);
    // 콜백 함수에 명시적으로 this를 바인딩한다
    setTimeout(function(){
      console.log(this.value); // 100
    }.bind(this), 100);
    // 화살표 함수 내부의 this는 상위 스코프의 this
    setTimeout(() => console.log(this.value), 100); // 100
  }
}
obj.foo();
```

### 22.2.2 메서드 호출

- 메서드 내부의 this 는 메서드를 호출한 객체
    - . 연산자 앞에 객체가 바인딩
    - 주의 메서드를 소유한 객체가 아닌 메서드를 호출한 객체에 바인딩

```jsx
const person = {
  name: 'Lee',
  getName(){
    return this.name;
  }
}
console.log(person.getName()); // Lee
const anotherPerson = {
  name: 'kim'
};
// 다른 객체로 호출
anotherPerson.getName = person.getName;
anotherPerson.getName(); //kim
// 일반 함수로 호출
const getName = person.getName;
getName(); // ''(window.name)
```

```jsx
// 프로토타입 메서드 내부에서 사용된 this도 일반 메서드와 마찬가지다
function Person(name){
  this.name = name;
}
Person.prototype.getName = function(){
  return this.name;
}
const me = new Person('Lee');
me.getName(); // Lee
Person.prototype.name = 'kim';
Person.prototype.getName(); // kim
```

### 22.2.3 생성자 함수 호출

- 생성자 함수 내부의 this에는 생성자 함수가 생성할 인스턴스가 바인딩된다

### 22.2.4 Function.prototype.apply/call/bind 메서드에 의한 간접 호출

- apply, call, bind 메서드는 Function.prototype의 메서드다. 즉 모든 함수가 상속받아 사용 할 수 있다
- apply, call 메서드의 본질적인 기능은 함수를 호출하는 것
    - 첫 번째 인수로 전달한 특정 객체를 호출한 함수의 this에 바인딩
- bind 메서드는 메서드의 this와 메서드 내부의 중첩 함수 또는 콜백 함수의 this가 불일치하는 문제를 해결하기 위해 유용하게 사용된다

```jsx
function getThisBinding(){
  return this;
}
// this로 사용할 객체
const thisArg = { a: 1 };

getThisBinding(); // window
getThisBinding.apply(thisArg); // {a: 1}
getThisBinding.call(thisArg); // {a: 1}

getThisBinding.bind(thisArg)(); // {a: 1}

function getThisBinding(){  
  console.log(arguments);
  return this;
}
function convertArgsTArray(){
  console.log(arguments);
  const arr = Array.prototype.slice.call(arguments);
  console.log(arr);
  return arr;
}
// this로 사용할 객체
const thisArg = { a: 1 };

getThisBinding.apply(thisArg, [1,2,3]); 
// Arguments(3) [1,2,3, callee: f, ~~]
// {a: 1}
getThisBinding.call(thisArg,1,2,3); // {a: 1}
// Arguments(3) [1,2,3, callee:f,~~]
// {a: 1}

convertArgsTArray(1,2,3); // [1,2,3]
```

```jsx
const person = {
  name: 'Lee',
  foo(callback){
    setTimeout(callback.bind(this), 100);
  }
}

person.foo(function() {
  console.log(`Hi ${this.name}`); // Hi Lee
});
```

| 함수 호출 방식 | this 바인딩 |
| --- | --- |
| 일반 함수 호출 | 전역 객체 |
| 메서드 호출 | 메서드를 호출한 객체 |
| 생성자 함수 호출 | 생성자 함수가 생성할 인스턴스 |
| apply/call/bind 메서드에 의한 간접 호출 | apply/call/bind 메서드에 첫번쨰 인수로 전달한 객체 |