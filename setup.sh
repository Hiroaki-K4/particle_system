#!/bin/bash

if [ ! -d "glfw" ]; then
    git clone https://github.com/glfw/glfw.git
    cd glfw
    git checkout b35641f4a3c62aa86a0b3c983d163bc0fe36026d
    cmake -S . -B build
    cd build
    make
fi
