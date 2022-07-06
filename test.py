from ltd_rust import downsample
import random
length = 1000

x = [i for i in range(length)]
y = [random.randint(0,1000) for i in range(length)]

x_down, y_down = downsample(x,y,100)