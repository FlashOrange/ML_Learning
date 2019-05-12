### 词向量
### (https://www.bilibili.com/video/av49462224/?p=2)
### 小游戏：FizzBuzz
 ####  3的倍数： Fizz
#### 5的倍数： Buzz
#### 15的倍数： FizzBuzz
#### 目标：使用神经网络来拟合这个小游戏


import numpy as np
import torch

def fizz_buzz_encode(num):
  if num % 15 == 0: return 3
  elif num % 5 == 0: return 2
  elif num % 3 == 0: return 1
  else: return 0
  
def fizz_buzz_decode(i, prediction):
  return [str(i),"fizz","buzz","fizzbuzz"][prediction]

def helper(i):
  print(fizz_buzz_decode(i,fizz_buzz_encode(i)))

#将训练数据转换为二进制，形成10维向量
NUM_DIGITS = 10

def binary_encode(i, num_digits):
  return np.array([i >> d & 1 for d in range(num_digits)][::-1]) #反转


#准备训练数据

trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2** NUM_DIGITS)]) #这里必须为整型


#建立神经网络模型
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)

loss_fn = torch.nn.CrossEntropyLoss() #分类问题
#optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
### 训练模型

BATCH_SIZE = 128


for epoch in range(10000):
  for start in range(0, len(trX), BATCH_SIZE):
    end = start + BATCH_SIZE
    batchX = trX[start:end]
    batchY = trY[start:end]
    
    y_pred = model(batchX) #forward
    loss = loss_fn(y_pred, batchY)
    if start % 16 == 0:
      print("Epoch:",epoch," Loss: ", loss.item())
    
    optimizer.zero_grad()
    loss.backward() #backpass
    optimizer.step()


### 使用测试集测试


testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)]) 
with torch.no_grad():
    testY = model(testX)

#输出最大值索引
argmax = testY.max(1)[1].data.tolist()

predictions = zip(range(1, 101), argmax)

print([fizz_buzz_decode(i, x) for i, x in predictions])




