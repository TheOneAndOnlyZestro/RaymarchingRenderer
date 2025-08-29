# Raymarching CUDA Mandelbulb

A tiny real-time raymarching renderer that draws a rotating **Mandelbulb** using **CUDA** for rendering and **OpenGL** (with **GLFW** + **GLEW**) for presentation. CUDA writes directly into an OpenGL texture via interop; a fullscreen quad displays the result.

> Requires an NVIDIA GPU. This project targets Windows and uses vendored GLFW/GLEW binaries.

---

## Features
- Real-time Mandelbulb raymarching (distance estimator)
- CUDA → OpenGL interop (no CPU readbacks)
- Fullscreen textured quad (GL 3.3+)
- Minimal, readable code and shaders

---

## Tech Stack
- **Languages:** C++20, CUDA C++
- **Graphics:** OpenGL 3.3+, GLFW (window/input), GLEW (extensions)
- **Build:** CMake
- **Interop:** `cudaGraphicsGLRegisterImage` + CUDA surface writes
