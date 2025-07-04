---
title: "Spread 문법"
layout: post
parent: deepdive
has_children: false
nav_order: 305
last_modified_at: 2024-02-01T18:20:02-05:00
---

# 35. 스프레드 문법

- ES6에서 도입된 스프레드 문법(전개 문법) ... 은 하나로 뭉쳐 있는 여러 값들의 집합을 펼쳐서 개별적인 값들의 목록으로만든다
- Array, String, Map, Set, DOM 컬랙션(NodeList, HTMLCollection), Arguments와 같이 for ... of 문으로 순회할 수 있는 이터러블에 한정한다

```jsx
// ...[1,2,3]은 개별요소로 분리한다 (-> 1,2,3)
console.log(...[1,2,3]); // 1 2 3 
// 문자열은 이터러블
console.log(...'Hello'); // H e l l o
// Map 과 Set은 이터러블
console.log(...new Map([['a','1'],['b','2']])); // ['a', '1'] ['b', '2']
console.log(...new Set([1, 2, 3])); // 1 2 3

// 이터러블이 아닌 일반 객체는 스프레드 문법의 대상이 될 수 없다.
conosole.log(...{a:1, b:2});// TypeError
const list = ...[1,2,3]; // SyntaxError
```

- 스프레드 문법 ... 이 피연산자를 연산하여 값을 생성하는 연산자가 아니다
    - 따라서 결과는 변수에 할당하루 수 없

## 35.1 함수 호출문의 인수 목록에서 사용하는 경우

```jsx
const arr = [1,2,3];
// 배열 arr의 요소 중에서 최대값을 구하기 위해 Math.max를 사용
const max  = Math.max(arr); // NaN

//Math.max 가변 인자 함수
Math.max(1); // 1
Math.max(1, 2); // 2
Math.max(1, 2, 3); // 3
Math.max(); // Infinity
```

- 요소들의 집합인 배열을 펼쳐서 개별적인 값들의 목록으로 만든 후, 이를 함수의 인수 목록으로 전달해야 하는 경우가 있다

```jsx
const arr = [1,2,3];

// apply 함수의 2번째 인수(배열)는 apply 함수가 호출하는 함수의 인수 목록이다.
// 따라서 배열이 펼쳐져서 인수로 전달되는 효과가 있다
const max = Math.max.apply(null, arr); // 3
// 스프레드로 펼쳐서 전달
const max = Math.max(...arr); // 3
```

- Rest 파라미터와 반대이다
    - Rest 파라미터는 매개변수 이름앞에 ...

```jsx
//Rest
function foo(...rest){
  console.log(rest); // 1,2,3 -> [1,2,3]
}

//스프레드
foo(...[1,2,3]);
```

## 35.2 배열 리터럴 내부에서 사용하는 경우

### 35.2.1 concat

- ES5에서 2개의 배열을 1개의 배열로 결합하고 싶은 경우 배열 리터럴만으로 해결할 수 없고  concat을 사용해야 한다

```jsx
//ES5
var arr = [1,2].concat([3,4]); // [1,2,3,4]
```

- 스프레드 문법

```jsx
//ES6
const arr = [...[1,2], ...[3,4]];
console.log(arr); // [1,2,3,4]
```

### 35.2.2 splice

- ES5 배열 중간 요소 추가 및 제거 하려면 splice  메서드를 사용한다

```jsx
//ES5
var arr1 = [1,4];
var arr2 = [2,3];
// 3번 인자 arr2를 풀어서 전달해야 의도한 동작을함
arr1.spice(1,0,arr2); // [1, [2,3], 4]

// apply 메서드의 2번째 인수(배열)는 splice 메서드의 인수목록이다
// 따라서 다 풀려서 전달됨
Array.prototype.splice.apply(arr1, [1,0].concat(arr2)); // [1,2,3,4]

//ES6

arr1.splice(1,0, ...arr2); // [1,2,3,4]
```

### 35.2.3 배열복사

- slice를 사용한다
    - 각 원소는 얕은 복사하여 복사본을 생성하다(스프레드 마찬가지)

```jsx
//ES5
var origin = [1, 2];
var copy = origin.slice();

console.log(copy); //[1, 2]
copy === origin // false

// ES6
const origin = [1,2];
const copy = [...origin];
console.log(copy); // [1,2]
copy === origin // false
```

### 35.2.4 이터러블을 배열로 변환

- Function.prototype.apply 또는 [Function.prototype.call](http://Function.prototype.call) 메서드를 사용하여 slice 메서드를 호출해야한다

```jsx
//ES5
function sum(){
  // 이터러블이면서 유사 배열 객체인 arguments를 배열로 변환
  var args = Array.prototype.slice.call(arguments);

  return args.reduce(function (pre, cur){
		return pre + cur;
  }, 0);
}

console.log(sum(1,2,3)); // 6

// 이터러블이 아닌 유사 객체
const arrayLike = {0:1, 1:2,2:3,length:3};
const arr = Array.prototype.slice.call(arrayLike); // [1,2,3]
console.log(Array.isArray(arr)); //true

//ES6
function sum(){
  return [...arguments].reduce((pre, cur) => pre+cur,0);
}

console.log(sum(1,2,3)); // 6

//Rest
const sum = (..args) => args.reduce((pre,cur) => pre+cur, 0);
console.log(sum(1,2,3)); // 6

//이터럴이 아닌 유사 배열 객체는 스프레드 문법의 대상이 될 수 없다
const arrayLike = {0:1, 1:2,2:3,length:3};
const arr = [...arrayLike]; // TypeError

//Array.from 은 사용가능
Array.from(arrayLike); // [1,2,3]
```

## 35.3 객체 리터럴 내부에서 사용하는 경우

- 스프레드 문법의 대상은 이터러블이어야 하지만 스프레드 푸로퍼티 제안은 일반 객체를 대상으로도 스프레드 문법의 사용을 허용한다
