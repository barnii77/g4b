cmake_minimum_required(VERSION 3.27)
project(g4b LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_library(g4b SHARED
        g4b/tokenizer.cpp
)
