        cmake_minimum_required(VERSION 4.0)
        project(RaymarchingCUDA LANGUAGES CUDA CXX)

        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CXX_STANDARD 20)

        set(THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/Vendor")
        set(GLFW_ROOT "${THIRDPARTY_DIR}/GLFW")
        set(GLEW_ROOT "${THIRDPARTY_DIR}/GLEW")
        set(IMGUI_ROOT "${THIRDPARTY_DIR}/ImGui")

        find_package(CUDAToolkit REQUIRED)
        find_package(OpenGL REQUIRED)

        add_library(glfw3 STATIC IMPORTED GLOBAL)
        set_target_properties(glfw3 PROPERTIES
                IMPORTED_LOCATION "${GLFW_ROOT}/lib/glfw3.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${GLFW_ROOT}/include"
        )

        add_library(GLEW::GLEW STATIC IMPORTED GLOBAL)
        set_target_properties(GLEW::GLEW PROPERTIES
                IMPORTED_LOCATION "${GLEW_ROOT}/lib/Release/x64/glew32s.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${GLEW_ROOT}/include"
                INTERFACE_COMPILE_DEFINITIONS "GLEW_STATIC"
        )

        add_executable(RaymarchingCUDA
                main.cpp
                fragment_kernel.cu
                stb_image.h
                stb_image_write.h
                ray_maths.cuh
                ray_FFT.cuh
                stbi_impl.cpp
                ${IMGUI_ROOT}/src/imgui.cpp
                ${IMGUI_ROOT}/src/imgui_draw.cpp
                ${IMGUI_ROOT}/src/imgui_tables.cpp
                ${IMGUI_ROOT}/src/imgui_widgets.cpp
                ${IMGUI_ROOT}/src/imgui_impl_glfw.cpp
                ${IMGUI_ROOT}/src/imgui_impl_opengl3.cpp
                Primitive.cu
                Primitive.cuh
                ray_flatten_to_CUDA.cu
                ray_flatten_to_CUDA.cuh
        )


        target_include_directories(RaymarchingCUDA PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS}
                ${IMGUI_ROOT}/include
        )

        target_link_libraries(RaymarchingCUDA PRIVATE
                CUDA::cudart
                CUDA::cuda_driver
                OpenGL::GL
                glfw3
                GLEW::GLEW
        )
        target_include_directories(RaymarchingCUDA PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies)
        set_target_properties(RaymarchingCUDA PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON)
