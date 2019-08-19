#!/bin/bash

ps -aux | grep 'take_cuda_mem' | awk '{print $2}' | xargs kill -9 2>/dev/null
