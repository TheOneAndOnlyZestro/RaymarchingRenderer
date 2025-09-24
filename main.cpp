#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "fragment_kernel_api.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "ray_flatten_to_CUDA.cuh"
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

const unsigned int width = 900;
const unsigned int height = 900;

static unsigned int ID_Counter = 0;
void addObject(std::shared_ptr<Primitive>* scene, std::vector<float>* output,
    std::vector<PrimitiveType>* outputDesc, float** output_device, PrimitiveType** output_disc_device,
    const std::vector<std::shared_ptr<Primitive>>& scene_objects, const PrimitiveType& type = PrimitiveType::UNION) {

    assert(type == PrimitiveType::UNION || type == PrimitiveType::INTERSECT);
    if (*scene == nullptr) {
        if (type == PrimitiveType::UNION) {
        *scene = std::make_shared<Union>(ID_Counter++,scene_objects[0], scene_objects[1]);
        }else {
            *scene = std::make_shared<Intersect>(ID_Counter++,scene_objects[0], scene_objects[1]);
        }
    }else {
        std::shared_ptr<Primitive> tempScene = *scene;
        if (type == PrimitiveType::UNION) {
            *scene = std::make_shared<Union>(ID_Counter++,tempScene, scene_objects[scene_objects.size()-1]);
        }else {
            *scene = std::make_shared<Intersect>(ID_Counter++,tempScene, scene_objects[scene_objects.size()-1]);
        }
    }
    output->clear();
    outputDesc->clear();
    ray::flatten(*scene, output, outputDesc);
    Free(*output_device,*output_disc_device);

    Allocate(output_device, output_disc_device, output->size(), outputDesc->size());
    toDevice(output->data(), outputDesc->data(), *output_device, *output_disc_device,
        output->size(), outputDesc->size());
}
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

    std::vector<std::shared_ptr<Primitive>> scene_objects;
    std::shared_ptr<Primitive> scene = nullptr;
    std::shared_ptr<Primitive> light =
        std::make_shared<Sphere>(ID_Counter++, ray::vec3(0.f,0.f,0.0f), ray::vec3(), ray::vec3(0.5f,0.5f,0.5f), 0.1f);

    scene_objects.push_back(light);

    std::vector<float> output;
    std::vector<PrimitiveType> outputDesc;
    ray::flatten(light, &output,&outputDesc);

    float* output_device;
    PrimitiveType* output_disc_device;
    Allocate(&output_device, &output_disc_device, output.size(), outputDesc.size());

    toDevice(output.data(), outputDesc.data(), output_device, output_disc_device, output.size(), outputDesc.size());

    bool intersect = false;

    ray::vec3* lightSourceDevice;
    AllocateLightSource(&lightSourceDevice);
    UpdateDeviceLightSource(light->getLocRef(), lightSourceDevice);


    while (!glfwWindowShouldClose(window)) {
        //Imgui new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //GUI Vars
        //IMGUI PLAYGROUND
        {
            ImGui::Begin("Control Panel");
            ImGui::Text("LightSource");
            if (ImGui::DragFloat3("Position", light->getLocRef()->v, 0.01f, -10.0f,10.0f)) {
                UpdateDeviceLightSource(light->getLocRef(), lightSourceDevice);
                output.clear();
                outputDesc.clear();
                ray::flatten(scene == nullptr ? light : scene, &output, &outputDesc);
                toDevice(output.data(), outputDesc.data(), output_device, output_disc_device, output.size(), outputDesc.size());
            }
            for (unsigned int i =0; i < scene_objects.size(); i++) {
                if ( scene_objects[i]->getID() == 0.f ) {
                    continue;
                }
                switch (scene_objects[i]->getType()) {
                    case PrimitiveType::CUBE:
                        ImGui::Text("Cube");
                        if (
                        ImGui::DragFloat3(("Position##" +std::to_string(i)).c_str(), scene_objects[i]->getLocRef()->v,0.05f,-5.0f,5.0f) ||
                        ImGui::DragFloat3(("Rotation##"+std::to_string(i)).c_str(), scene_objects[i]->getRotRef()->v,0.5f,-360.0f,360.0f) ||
                        ImGui::DragFloat3(("Scale##"+std::to_string(i)).c_str(), scene_objects[i]->getScaleRef()->v,0.05f,0.0f,5.0f) ) {
                            output.clear();
                            outputDesc.clear();
                            ray::flatten( scene, &output, &outputDesc);
                            toDevice(output.data(), outputDesc.data(), output_device, output_disc_device, output.size(), outputDesc.size());
                        }
                        break;
                    case PrimitiveType::SPHERE:

                        ImGui::Text("Sphere");
                        if (
                        ImGui::DragFloat3(("Position##"+std::to_string(i)).c_str(), scene_objects[i]->getLocRef()->v,0.05f,-5.0f,5.0f)||
                        ImGui::DragFloat3(("Rotation##"+std::to_string(i)).c_str(), scene_objects[i]->getRotRef()->v,0.5f,-360.0f,360.0f)||
                        ImGui::DragFloat3(("Scale##"+std::to_string(i)).c_str(), scene_objects[i]->getScaleRef()->v,0.05f,0.0f,5.0f)||
                        ImGui::DragFloat(("Radius##"+std::to_string(i)).c_str(), std::dynamic_pointer_cast<Sphere>(scene_objects[i])->getRadiusRef(),0.05f,0.0f,10.0f)) {
                            output.clear();
                            outputDesc.clear();
                            ray::flatten(scene == nullptr ? light : scene, &output, &outputDesc);
                            toDevice(output.data(), outputDesc.data(), output_device, output_disc_device, output.size(), outputDesc.size());
                        }
                        break;
                    case PrimitiveType::MANDELBROT:
                        ImGui::Text("Mandelbulb");
                        if (
                        ImGui::DragFloat3(("Position##"+std::to_string(i)).c_str(), scene_objects[i]->getLocRef()->v,0.05f,-5.0f,5.0f)||
                        ImGui::DragFloat3(("Rotation##"+std::to_string(i)).c_str(),scene_objects[i]->getRotRef()->v,0.5f,-360.0f,360.0f)||
                        ImGui::DragFloat3(("Scale##"+std::to_string(i)).c_str(), scene_objects[i]->getScaleRef()->v,0.05f,-5.0f,5.0f)||
                        ImGui::DragFloat(("Exponent##"+std::to_string(i)).c_str(),std::dynamic_pointer_cast<Mandelbulb>(scene_objects[i])->getExponentRef(),0.05f,0.f,50.0f)) {
                            output.clear();
                            outputDesc.clear();
                            ray::flatten(scene, &output, &outputDesc);
                            toDevice(output.data(), outputDesc.data(), output_device, output_disc_device, output.size(), outputDesc.size());
                        }
                        break;

                    case PrimitiveType::LINE:
                        ImGui::Text("Line");
                        if (
                        ImGui::DragFloat3(("Position##"+std::to_string(i)).c_str(), scene_objects[i]->getLocRef()->v,0.05f,-5.0f,5.0f)||
                        ImGui::DragFloat3(("Rotation##"+std::to_string(i)).c_str(), scene_objects[i]->getRotRef()->v,0.5f,-360.0f,360.0f)||
                        ImGui::DragFloat3(("Scale##"+std::to_string(i)).c_str(), scene_objects[i]->getScaleRef()->v,0.05f,-5.0f,5.0f)||
                        ImGui::DragFloat3(("A##"+std::to_string(i)).c_str(), std::dynamic_pointer_cast<Line>(scene_objects[i])->getARef()->v,0.05f,-5.f,5.0f)||
                        ImGui::DragFloat3(("B##"+std::to_string(i)).c_str(), std::dynamic_pointer_cast<Line>(scene_objects[i])->getBRef()->v,0.05f,-5.f,5.0f)||
                        ImGui::DragFloat(("Radius##"+std::to_string(i)).c_str(), std::dynamic_pointer_cast<Line>(scene_objects[i])->getRadiusRef(),0.05f,0.f,10.0f)
                        ) {
                            output.clear();
                            outputDesc.clear();
                            ray::flatten(scene, &output, &outputDesc);
                            toDevice(output.data(), outputDesc.data(), output_device, output_disc_device, output.size(), outputDesc.size());
                        }
                        break;
                    default:
                        break;
                }
            }
            ImGui::Checkbox("Intersect", &intersect);
            if (ImGui::Button("Add Cube")) {
                scene_objects.push_back(std::make_shared<Cube>(ID_Counter++,ray::vec3(0.f,0.f,-3.3f), ray::vec3(), ray::vec3(0.5f,0.5f,0.5f)));
                addObject( &scene, &output, &outputDesc, &output_device, &output_disc_device, scene_objects, intersect ? PrimitiveType::INTERSECT: PrimitiveType::UNION);
                for (int i =0; i < output.size(); i++) {
                    printf("output[%d] = %f ", i, output[i]);
                }
            }
            if (ImGui::Button("Add Sphere")) {
                scene_objects.push_back(std::make_shared<Sphere>(ID_Counter++,ray::vec3(0.f,0.f,-3.3f), ray::vec3(), ray::vec3(0.5f,0.5f,0.5f), 0.5f));
                addObject( &scene, &output, &outputDesc, &output_device, &output_disc_device, scene_objects, intersect ? PrimitiveType::INTERSECT: PrimitiveType::UNION);
            }
            if (ImGui::Button("Add Mandelbulb")) {
                scene_objects.push_back(std::make_shared<Mandelbulb>(ID_Counter++,ray::vec3(0.f,0.f,-3.3f), ray::vec3(), ray::vec3(0.5f,0.5f,0.5f), 8, 8.f));
                addObject( &scene, &output, &outputDesc, &output_device, &output_disc_device, scene_objects, intersect ? PrimitiveType::INTERSECT: PrimitiveType::UNION);
            }
            if (ImGui::Button("Add Line")) {
                scene_objects.push_back(std::make_shared<Line>(ID_Counter++,ray::vec3(0.f,0.f,-3.3f), ray::vec3(), ray::vec3(0.5f,0.5f,0.5f),
                    ray::vec3(-0.5,0.f, -3.3f), ray::vec3(0.5,0.f, -3.3f), 0.2f) );
                addObject( &scene, &output, &outputDesc, &output_device, &output_disc_device, scene_objects, intersect ? PrimitiveType::INTERSECT: PrimitiveType::UNION);
            }
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

        launchFragment(surf, width,height,glfwGetTime(),output_device,output.size(), output_disc_device, outputDesc.size(), lightSourceDevice);

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

    Free(output_device, output_disc_device);
    FreeDeviceLightSource(lightSourceDevice);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

