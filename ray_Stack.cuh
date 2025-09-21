//
// Created by Andrew on 9/21/2025.
//

#ifndef RAYMARCHINGCUDA_RAY_STACK_H
#define RAYMARCHINGCUDA_RAY_STACK_H

namespace ray {
    template<typename T, unsigned int max_size>
    class Stack {
    private:
        T data[max_size];
        size_t size;
    public:
        __host__ __device__
        inline Stack(): size(0) {}

        __host__ __device__
        inline void push(T x) {
            data[size++] = x;
        }
        __host__ __device__
        inline T pop() {
            return data[--size];
        }
        __host__ __device__
        inline T peek() {
            return data[size];
        }

        __host__ __device__
        inline bool isEmpty() const {
            return size == 0;
        }
        __host__ __device__
        inline bool isFull() const {
            return size == max_size;
        }
        __host__ __device__
        inline ~Stack(){}

    };
}



#endif //RAYMARCHINGCUDA_RAY_STACK_H