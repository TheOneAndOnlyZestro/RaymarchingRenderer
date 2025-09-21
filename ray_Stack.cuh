//
// Created by Andrew on 9/21/2025.
//

#ifndef RAYMARCHINGCUDA_RAY_STACK_H
#define RAYMARCHINGCUDA_RAY_STACK_H

namespace ray {
    class Stack {
    private:
        float* data;
        size_t size;
        size_t max_size;
    public:
        __host__ __device__
        Stack(unsigned int max_num = 50);

        __host__ __device__
        void push(float x);
        __host__ __device__
        float pop();
        __host__ __device__
        float peek();

        __host__ __device__
        bool isEmpty() const;
        __host__ __device__
        bool isFull() const;
        __host__ __device__
        ~Stack();



    };
}



#endif //RAYMARCHINGCUDA_RAY_STACK_H