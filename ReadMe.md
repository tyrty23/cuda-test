# Cuda test tasks

```
cd build
./run
```


## Замеры производительности при разных параметрах запуска kernel

Все замеры производяться на сетке 4096х4096 при *max_iter*=1000
### 2D Grid and 2D Blocks

```
Time elapsed on cpu: 165.22
maldelbrot_gpu<<<(128,128), (32,32)>>> elapsed 1.463407 sec
```

```
maldelbrot_gpu<<<(128,256), (32,16)>>> elapsed 1.417980 sec
```
```
maldelbrot_gpu<<<(256,256), (16,16)>>> elapsed 1.395084 sec
```
### 2D Grid and 1D Blocks
```
maldelbrot_gpu<<<(128,4096), (32,1)>>> elapsed 1.432542 sec
```
```
maldelbrot_gpu<<<(16,4096), (256,1)>>> elapsed 1.432624 sec
```
```
maldelbrot_gpu<<<(4,4096), (1024,1)>>> elapsed 1.412368 sec
```
