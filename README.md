# paddle_use_available_gpu
This is a repo to test PaddlePaddle can use only available GPU when other program occupies the GPU

The scripts were tested on 16 GB GPU. It launches a program taking 4 GB GPU, then a PaddlePaddle
program taking 10 GB GPU.

Usage:

Single GPU card

```
CUDA_VISIBLE_DEVICES=0 sh main.sh
```

8 GPU cards

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh main.sh
```

You should see:

output = 1.000000

Create data on GPU successfully

Note: the script will kill process with name "take_cuda_mem" after running. So please check your
computer doesn't have running program with the same name.
