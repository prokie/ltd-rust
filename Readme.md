This is a rust implementation of the downsampling alogrithm Large-Triangle-Dynamic. The algorithm was described in the thesis "Downsampling Time Series for Visual Representation
" written by Sveinn Steinarsson. You can find his thesis at https://skemman.is/handle/1946/15343f.

The code is losely taken from https://github.com/janjakubnanista/downsample.

This was a fun projet for me were I was able to implement the algorithm in rust and also publish it as a python package.

## Usage

```
from ltd_rust import downsample
import random
length = 1000

x = [i for i in range(length)]
y = [random.randint(0,1000) for i in range(length)]

x_down, y_down = downsample(x,y,100)
```
