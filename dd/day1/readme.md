# ==================== 学习内容1：变量、变量类型、作用域 ====================
print("\n===== 学习内容1：变量、变量类型、作用域 =====")

# 基本数据类型
name = "Bob"          # 字符串类型
age = 25              # 整数类型
height = 1.75         # 浮点数类型
is_student = True     # 布尔类型

# 复合数据类型
scores = [92, 88, 95]  # 列表 - 有序可变
grades = (90, 85, 88)  # 元组 - 有序不可变
info = {"name": "Bob", "age": 25, "city": "Beijing"}  # 字典 - 键值对
courses = {"Math", "English", "Science"}  # 集合 - 无序唯一

# 类型转换示例
age_str = str(age)        # 整数转字符串
num = int("123")          # 字符串转整数
pi = float("3.14")        # 字符串转浮点数
bool_value = bool(1)      # 非零值转为True

# 作用域示例
x = 100  # 全局变量

def test_scope():
    global x  # 声明使用全局变量
    x = 200   # 修改全局变量
    y = 50    # 局部变量
    print(f"函数内部: x={x}, y={y}")

test_scope()
print(f"函数外部: x={x}")  # 输出200，全局变量已被修改


# ==================== 学习内容2：运算符及表达式 ====================
print("\n===== 学习内容2：运算符及表达式 =====")

# 算术运算符
a = 15
b = 4
print(f"{a} + {b} = {a + b}")     # 加法
print(f"{a} - {b} = {a - b}")     # 减法
print(f"{a} * {b} = {a * b}")     # 乘法
print(f"{a} / {b} = {a / b}")     # 除法(结果为浮点数)
print(f"{a} // {b} = {a // b}")   # 整除(向下取整)
print(f"{a} % {b} = {a % b}")     # 取余
print(f"{a} ** {b} = {a ** b}")   # 幂运算

# 比较运算符
x = 10
y = 20
print(f"{x} > {y}: {x > y}")      # 大于
print(f"{x} <= {y}: {x <= y}")    # 小于等于
print(f"{x} == {y}: {x == y}")    # 等于
print(f"{x} != {y}: {x != y}")    # 不等于

# 逻辑运算符
p = True
q = False
print(f"{p} and {q}: {p and q}")  # 逻辑与
print(f"{p} or {q}: {p or q}")    # 逻辑或
print(f"not {p}: {not p}")        # 逻辑非

# 位运算符
a = 5  # 二进制: 0101
b = 3  # 二进制: 0011
print(f"{a} & {b} = {a & b}")     # 按位与: 0001 (1)
print(f"{a} | {b} = {a | b}")     # 按位或: 0111 (7)
print(f"{a} ^ {b} = {a ^ b}")     # 按位异或: 0110 (6)
print(f"~{a} = {~a}")             # 按位取反: -6 (补码表示)
print(f"{a} << {b} = {a << b}")   # 左移: 0101000 (40)
print(f"{a} >> {b} = {a >> b}")   # 右移: 0000 (0)


# ==================== 学习内容3：条件、循环、异常 ====================
print("\n===== 学习内容3：条件、循环、异常 =====")

# 条件语句
score = 88
if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")

# 循环语句
# for循环示例
print("1-5的数字:")
for i in range(1, 6):
    print(i, end=' ')
print()

# while循环示例
count = 0
while count < 5:
    print(f"循环次数: {count}")
    count += 1

# 循环控制语句
print("跳过3的循环:")
for i in range(5):
    if i == 3:
        continue  # 跳过当前循环
    print(i, end=' ')
print()

print("找到3后终止循环:")
for i in range(5):
    if i == 3:
        break  # 终止整个循环
    print(i, end=' ')
print()

# 异常处理
try:
    num = int(input("请输入一个数字: "))
    result = 10 / num
except ValueError:
    print("输入不是有效的数字!")
except ZeroDivisionError:
    print("不能除以零!")
else:  # 当没有异常时执行
    print(f"结果: {result}")
finally:  # 无论是否有异常都会执行
    print("程序执行完毕")


# ==================== 学习内容4：函数 ====================
print("\n===== 学习内容4：函数 =====")

# 函数定义与调用
def calculate_area(radius, pi=3.14):
    """计算圆的面积"""
    return pi * radius ** 2

# 位置参数调用
print(f"半径为5的圆面积: {calculate_area(5)}")
# 关键字参数调用
print(f"半径为5，π取3.1416的圆面积: {calculate_area(5, pi=3.1416)}")

# 可变参数函数
def sum_numbers(*args):
    """计算任意数量数字的和"""
    total = 0
    for num in args:
        total += num
    return total

print(f"1+2+3+4+5 = {sum_numbers(1, 2, 3, 4, 5)}")
print(f"2+4+6 = {sum_numbers(2, 4, 6)}")

# 匿名函数(lambda)
square = lambda x: x ** 2
print(f"5的平方: {square(5)}")

# 高阶函数 - map示例
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(f"列表平方: {squared}")

# 高阶函数 - filter示例
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"偶数列表: {even_numbers}")

# 自定义高阶函数
def apply_operation(func, a, b):
    return func(a, b)

add = lambda x, y: x + y
multiply = lambda x, y: x * y

print(f"3+4 = {apply_operation(add, 3, 4)}")
print(f"3*4 = {apply_operation(multiply, 3, 4)}")


# ==================== 学习内容5：包和模块 ====================
print("\n===== 学习内容5：包和模块 =====")

# 模块使用示例
# 假设已创建了以下模块文件:

# mymath.py
# def add(a, b):
#     return a + b
#
# def subtract(a, b):
#     return a - b

# 导入整个模块
import mymath
print(f"3+5 = {mymath.add(3, 5)}")
print(f"8-2 = {mymath.subtract(8, 2)}")

# 导入特定函数
from mymath import add, subtract
print(f"4+6 = {add(4, 6)}")

# 导入模块并使用别名
import mymath as mm
print(f"7-3 = {mm.subtract(7, 3)}")

# 第三方模块示例 (需先安装: pip install requests)
try:
    import requests
    response = requests.get("https://www.example.com")
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容前50个字符: {response.text[:50]}")
except ImportError:
    print("请先安装requests模块: pip install requests")


# ==================== 学习内容6：类和对象 ====================
print("\n===== 学习内容6：类和对象 =====")

# 类的定义与使用
class Animal:
    """动物基类"""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def speak(self):
        return f"{self.name} 发出声音"

# 继承示例
class Dog(Animal):
    """狗类，继承自Animal"""
    def __init__(self, name, age, breed):
        super().__init__(name, age)  # 调用父类构造方法
        self.breed = breed
    
    def speak(self):  # 方法重写
        return f"{self.name} ({self.breed}) 汪汪叫"

class Cat(Animal):
    """猫类，继承自Animal"""
    def speak(self):  # 方法重写
        return f"{self.name} 喵喵叫"

# 创建对象并调用方法
dog = Dog("Buddy", 3, "Golden Retriever")
cat = Cat("Whiskers", 2)

print(dog.speak())  # 输出: Buddy (Golden Retriever) 汪汪叫
print(cat.speak())  # 输出: Whiskers 喵喵叫

# 类属性与实例属性
class Car:
    wheels = 4  # 类属性
    
    def __init__(self, brand, color):
        self.brand = brand  # 实例属性
        self.color = color
    
    def get_info(self):
        return f"{self.color} {self.brand} 汽车，有 {Car.wheels} 个轮子"

car1 = Car("Toyota", "Blue")
car2 = Car("BMW", "Black")

print(car1.get_info())  # 输出: Blue Toyota 汽车，有 4 个轮子
print(car2.get_info())  # 输出: Black BMW 汽车，有 4 个轮子


# ==================== 学习内容7：装饰器 ====================
print("\n===== 学习内容7：装饰器 =====")

# 装饰器基础
def timer_decorator(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    return wrapper

@timer_decorator
def calculate_sum(n):
    """计算1到n的和"""
    total = 0
    for i in range(1, n+1):
        total += i
    return total

print(f"总和: {calculate_sum(100000)}")

# 带参数的装饰器
def repeat(n):
    """重复执行函数n次的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(n):
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # 输出: ['Hello, Alice!', 'Hello, Alice!', 'Hello, Alice!']

# 多个装饰器叠加
def uppercase_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

def add_prefix_decorator(prefix):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return f"{prefix} {result}"
        return wrapper
    return decorator

@add_prefix_decorator("INFO:")
@uppercase_decorator
def get_message():
    return "hello world"

print(get_message())  # 输出: INFO: HELLO WORLD


# ==================== 学习内容8：文件操作 ====================
print("\n===== 学习内容8：文件操作 =====")

# 文件读写基础
# 写入文件
with open("data.txt", "w", encoding="utf-8") as file:
    file.write("第一行数据\n")
    file.write("第二行数据\n")
    file.write("第三行数据\n")

# 读取文件
with open("data.txt", "r", encoding="utf-8") as file:
    # 读取全部内容
    content = file.read()
    print("读取全部内容:")
    print(content)

# 逐行读取
with open("data.txt", "r", encoding="utf-8") as file:
    print("\n逐行读取:")
    for line in file:
        print(line.strip())  # 去除行末换行符

# 追加内容
with open("data.txt", "a", encoding="utf-8") as file:
    file.write("追加的内容\n")

# CSV文件操作
import csv

# 写入CSV
with open("students.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["姓名", "年龄", "成绩"])  # 写入表头
    writer.writerow(["张三", 20, 85])
    writer.writerow(["李四", 21, 90])
    writer.writerow(["王五", 19, 78])

# 读取CSV
with open("students.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    print("\nCSV文件内容:")
    for row in reader:
        print(row)

# JSON文件操作
import json

# 写入JSON
data = {
    "name": "张三",
    "age": 20,
    "courses": ["数学", "英语", "计算机"],
    "scores": {"数学": 85, "英语": 90, "计算机": 92}
}

with open("student.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

# 读取JSON
with open("student.json", "r", encoding="utf-8") as file:
    student_data = json.load(file)
    print("\nJSON数据:")
    print(f"姓名: {student_data['name']}")
    print(f"课程: {', '.join(student_data['courses'])}")