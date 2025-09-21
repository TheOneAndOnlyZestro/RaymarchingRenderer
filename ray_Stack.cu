//
// Created by Andrew on 9/21/2025.
//
#include "ray_Stack.cuh"

__host__ __device__
ray::Stack::Stack(unsigned int max_num) {
    max_size = max_num;
    data = new float[max_num];
    size = 0;
}

__host__ __device__
void ray::Stack::push(float x) {
    data[size++] = x;
}

__host__ __device__
float ray::Stack::pop() {
    return data[--size];
}

__host__ __device__
float ray::Stack::peek() {
    return data[size];
}

__host__ __device__
bool ray::Stack::isEmpty() const {
    return size == 0;
}
__host__ __device__
bool ray::Stack::isFull() const {
    return size == max_size;
}

__host__ __device__
ray::Stack::~Stack() {
    delete[] data;
}


