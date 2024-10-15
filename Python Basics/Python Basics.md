
# Python Basics for Data Science

Welcome to the "Python Basics for Data Science"! This module is designed to teach you the fundamental concepts of Python, which is the core programming language for data science. Whether you're a complete beginner or just looking to solidify your understanding of Python basics, this guide will help you build a strong foundation.

## Table of Contents
1. [Introduction to Python](#introduction-to-python)
2. [Setting Up the Environment](#setting-up-the-environment)
3. [Basic Syntax](#basic-syntax)
4. [Variables and Data Types](#variables-and-data-types)
5. [Operators](#operators)
6. [Control Flow (if-else, loops)](#control-flow-if-else-loops)
7. [Functions](#functions)
8. [Data Structures](#data-structures)
   - Lists
   - Tuples
   - Dictionaries
   - Sets
9. [File Handling](#file-handling)
10. [Exception Handling](#exception-handling)
11. [Basic Libraries for Data Science](#basic-libraries-for-data-science)
    - Numpy
    - Pandas
12. [Next Steps](#next-steps)

---

## Introduction to Python
Python is an interpreted, high-level, and general-purpose programming language. It is one of the most popular languages for data science due to its readability, simplicity, and vast ecosystem of libraries for machine learning, data analysis, and visualization.

---

## Setting Up the Environment
Before we dive into Python basics, let’s set up a Python environment.

1. **Install Python**:
   Download and install Python from the official site: [Python Downloads](https://www.python.org/downloads/)
   
2. **Set up a code editor**:
   Install a code editor such as [VS Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/).

3. **Set up Jupyter Notebook**:
   Jupyter Notebooks are widely used in the data science community.
   Install Jupyter with:
   ```bash
   pip install notebook
   ```
   Launch Jupyter Notebook using:
   ```bash
   jupyter notebook
   ```

---

## Basic Syntax
Let's start with the basic syntax of Python. Here are some key components:

- **Printing to console**:
   ```python
   print("Hello, World!")
   ```

- **Comments**:
   ```python
   # This is a single-line comment
   
   """
   This is a multi-line comment.
   """
   ```

- **Indentation**:
   Indentation is used to define blocks of code. Python uses spaces or tabs to do this.
   ```python
   if 5 > 2:
       print("Five is greater than two!")
   ```

---

## Variables and Data Types
Python is dynamically typed, meaning variables don't need explicit declarations.

- **Assigning values to variables**:
   ```python
   x = 5
   y = "Hello"
   z = 3.14
   ```

- **Common Data Types**:
   - `int` (Integer): Whole numbers.
   - `float`: Numbers with decimals.
   - `str` (String): Sequence of characters.
   - `bool` (Boolean): `True` or `False`.
   - `None`: Represents the absence of a value.

---

## Operators
Python supports several types of operators:

1. **Arithmetic Operators**: `+`, `-`, `*`, `/`, `%`, `**` (exponentiation), `//` (floor division)
   ```python
   a = 10
   b = 3
   print(a + b, a - b, a * b, a / b, a % b)
   ```

2. **Comparison Operators**: `==`, `!=`, `>`, `<`, `>=`, `<=`
3. **Logical Operators**: `and`, `or`, `not`
4. **Assignment Operators**: `=`, `+=`, `-=`, etc.

---

## Control Flow (if-else, loops)
### **Conditional Statements**:
```python
x = 10
if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x is 5")
else:
    print("x is less than 5")
```

### **Loops**:
#### **for loop**:
```python
for i in range(5):
    print(i)
```

#### **while loop**:
```python
count = 0
while count < 5:
    print(count)
    count += 1
```

---

## Functions
Functions are reusable blocks of code.

### **Defining a Function**:
```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

### **Lambda Functions**:
Short, anonymous functions:
```python
add = lambda x, y: x + y
print(add(3, 5))
```

---

## Data Structures

### **Lists**:
Ordered and mutable collections.
```python
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
print(fruits)
```

### **Tuples**:
Ordered but immutable collections.
```python
coordinates = (10, 20)
```

### **Dictionaries**:
Unordered collections of key-value pairs.
```python
student = {"name": "John", "age": 20}
print(student["name"])
```

### **Sets**:
Unordered collections of unique elements.
```python
unique_numbers = {1, 2, 3, 4, 5}
```

---

## File Handling
Python allows you to work with files, such as reading from or writing to them.

```python
# Writing to a file
with open("file.txt", "w") as file:
    file.write("Hello, World!")

# Reading from a file
with open("file.txt", "r") as file:
    content = file.read()
    print(content)
```

---

## Exception Handling
Handle errors and exceptions in your code to prevent crashes.

```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("This will always execute.")
```

---

## Basic Libraries for Data Science

### Numpy
Numpy is a library used for numerical computations in Python.

```python
import numpy as np
array = np.array([1, 2, 3, 4])
print(array)
```

### Pandas
Pandas is a powerful data manipulation library.

```python
import pandas as pd
data = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [24, 27, 22]
})
print(data)
```

---





