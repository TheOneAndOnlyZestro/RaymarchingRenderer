#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "fragment_kernel_api.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

const float default_verts[] = {
    -1.f, -1.f, 0.f,    0.f,0.f,
    1.f, -1.f, 0.f,     1.f,0.f,
    1.f, 1.f,0.f,       1.f,1.f,

    1.f, 1.f,0.f,       1.f,1.f,
    -1.f,1.f,0.f,       0.f,1.f,
    -1.f,-1.f,0.f,      0.f,0.f
};

const char* vertexShaderSource = R"glsl(
#version 330 core

layout(location = 0) in vec3 aPos;      // vertex position
layout(location = 1) in vec2 aTexCoord; // texture coordinate

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)glsl";

// Fragment shader
const char* fragmentShaderSource = R"glsl(
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uTexture; // texture bound to GL_TEXTURE0

void main()
{
    FragColor = texture(uTexture, TexCoord);
}
)glsl";

const unsigned int width = 800;
const unsigned int height = 800;
int main() {


    glfwInit();
    GLFWwindow* window = glfwCreateWindow(width, height, "Fractals", nullptr, nullptr);

    //check for window
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW." << std::endl;
        return -1;
    }

    //Setup IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    //initialize imgui for glfw and opengl
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    cudaGLSetGLDevice(0);

    //Setup both shaders
    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, nullptr);
    glCompileShader(vs);

    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fs);

    unsigned int sP = glCreateProgram();
    glAttachShader(sP, vs);
    glAttachShader(sP, fs);
    glLinkProgram(sP);
    glUseProgram(sP);

    glDeleteShader(vs);
    glDeleteShader(fs);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenerateMipmap(GL_TEXTURE_2D);

    //Register the texture form cuda
    cudaGraphicsResource* cudaRes = nullptr;
    cudaGraphicsGLRegisterImage(&cudaRes, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    //Create Vertex Buffer
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(default_verts), default_verts, GL_STATIC_DRAW);

    //Vertex Atrributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    ray::vec3 loc;
    ray::vec3 rot;
    float n = 8.f;
    while (!glfwWindowShouldClose(window)) {
        //Imgui new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //GUI Vars
        //IMGUI PLAYGROUND
        {
            ImGui::Begin("Control Panel");
            ImGui::DragFloat3("Position",loc.v,0.2f,-5.0f,5.0f);
            ImGui::DragFloat3("Rotation",rot.v,0.2f,-360.0f,360.0f);
            ImGui::DragFloat("Exp",&n,.02f, 0,100.f);
            ImGui::End();
        }
        //Game Loop
        glClearColor(0.0f,0.2f,0.5f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //PLAYGROUND
        cudaArray_t cudaArray = nullptr;
        cudaGraphicsMapResources(1,&cudaRes, 0);
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaRes, 0,0);

        cudaResourceDesc resourceDesc = {};
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = cudaArray;

        cudaSurfaceObject_t surf = 0;
        cudaCreateSurfaceObject(&surf, &resourceDesc);


        launchFragment(surf, glfwGetTime() * 0.1f,width, height,loc,rot,n);

        cudaDestroySurfaceObject(surf);
        cudaGraphicsUnmapResources(1,&cudaRes, 0);

        glBindTexture(GL_TEXTURE_2D, texture);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        //Render everything
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        // Poll for and process events
        glfwPollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

