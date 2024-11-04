#!/bin/bash

g++ add.cpp -o add_cpp
nvcc add.cu -o add_cuda

./add_cpp
./add_cuda