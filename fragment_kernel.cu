#include "fragment_kernel_api.h"

//Do it on the kernel

std::string getNextRenderFileName(const std::string& directoryName) {
    std::filesystem::path output_dir = directoryName;

    if (std::filesystem::exists(output_dir)) {

        int maxIndex = -1;
        for (std::filesystem::directory_iterator iter(output_dir), end; iter != end; iter++) {
            std::string currentPath = iter->path().filename().string();
            int index = std::stoi(currentPath.substr(
            currentPath.find_first_of("(") + 1, currentPath.find_last_of(")")
            ));
            maxIndex = index > maxIndex ? index : maxIndex;
        }

        std::stringstream s;
        s << "Render(" << maxIndex+1 << ").jpg";
        output_dir = output_dir / s.str();
    }

    return output_dir.string();
}

#define MAX_STEPS 200

__device__ void parseExpression(const ray::vec3& p, const float* output, const PrimitiveType* output_disc, float* data, ray::vec3* normal, const unsigned int output_size,const unsigned int outputDiscSize) {
    //We want to parse the postfix expression at hand using a stack
    ray::Stack<float,50> distance_stack;
    ray::Stack<ray::vec3,50> normal_stack;

    float currentData[4];
    float operands[2];
    size_t s;
    unsigned int dataOffset =0;
    for (unsigned int i = 0; i < outputDiscSize; i++) {
        switch (output_disc[i]) {
            case PrimitiveType::CUBE:
                Cube::CubeSDFF(p,output+dataOffset, &s, currentData);
                break;

            case PrimitiveType::SPHERE:
                Sphere::SphereSDFF(p,output+dataOffset, &s, currentData);
                break;

            case PrimitiveType::MANDELBROT:
                Mandelbulb::MandelbulbSDFF(p,output+dataOffset, &s, currentData);
                break;

            case PrimitiveType::UNION:
                Union::UnionSDFF(distance_stack.pop(),distance_stack.pop(), normal_stack.pop(),normal_stack.pop(), &s, currentData);
                break;

            case PrimitiveType::INTERSECT:
                Intersect::IntersectSDFF(distance_stack.pop(),distance_stack.pop(), normal_stack.pop(),normal_stack.pop(), &s, currentData);
                break;

            default:
                break;
        }
        distance_stack.push(currentData[0]);
        normal_stack.push(ray::vec3(currentData[1], currentData[2], currentData[3]));
        dataOffset += getPrimSize(output_disc[i]);

    }

    *data = distance_stack.pop();
    *normal = normal_stack.pop();
}


__global__
void FragmentKernel(cudaSurfaceObject_t surf,unsigned int width, unsigned int height, float time, const float* output, const unsigned int output_size, const PrimitiveType* output_disc, const unsigned int outputDiscSize) {

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    float aspectRatio = (float)width / (float)height;

    float u = ((((float)x/width) * 2.0f ) -1.0f ) * aspectRatio;
    float v = ((((float)(height - y)/height) * 2.0f) - 1.0f);

    ray::vec3 ro(0.f,0.f,-5.0f);
    ray::vec3 lightSource(0.f,4.f, -5.5f);
    ray::vec3 rd(u,v,1.0f);
    rd = ray::normalize(rd);

    float t = 0.0f;

    unsigned int i;
    float data = 0;
    ray::vec3 normal(0.f,0.f,0.f);
    ray::vec3 p;
    size_t dataSize;
    for (i = 0; i < MAX_STEPS; i++) {
        //Calculate position after marching according to line
        p = ro + (rd * t);

        //Check SDF
        parseExpression(p,output, output_disc, &data, &normal, output_size ,outputDiscSize);
        //printf("%f \n", data[0]);
        //March by this unit
        if (abs(data) < 0.001f || t > 100.0f) { break;}
        t+=data;
    }
    float diffuse = 0.f;
    float intensity = 0.9f;
    const float epsilon = 0.0001f;
    float ambient = 0.0f;
    float edge = 0.0f;
    float rim = 0.f;
    float mask = 0.0f;
    if (data < 0.01f) {
        mask =1.f;
        //we hit a surface, calculate its normal
        //calculate dot product between normal and a light source
        ray::vec3 incident = ray::normalize(lightSource - p);
        diffuse = ray::dot(incident, normal) * intensity;
        ambient = 0.7f;


        edge = (i/(float)MAX_STEPS);
        edge = edge * edge * 0.7f;

        ray::vec3 viewing = ray::normalize(ro - p);
        rim = pow(1.0f - max(ray::dot(normal, viewing), 0.0f), 2.0f);

    }
    float val =  ray::clamp(t * 0.2f + (i /(float)MAX_STEPS), 0.f, 1.f);

    ray::vec3 fcolor;

    ray::vec3 base_color(1.0f,1.0f,1.0f);
    fcolor =  base_color * diffuse * mask;


    //fcolor = color * diffuse;
    ray::vec3 debug(val,val,val);
    //fcolor = debug;
    if (x < width && y < height) {
        // image[(y * width + x) * 3 + 0] = (unsigned int)(fcolor.r * 255.0f);
        // image[(y * width + x) * 3 + 1] = (unsigned int)(fcolor.g * 255.0f);
        // image[(y * width + x) *3 + 2] = (unsigned int)(fcolor.b * 255.0f);

        unsigned char r = (unsigned char)(fcolor.r * 255.0f);
        unsigned char g = (unsigned char)(fcolor.g * 255.0f);
        unsigned char b = (unsigned char)(fcolor.b * 255.0f);
        unsigned char a = 255;

        surf2Dwrite(make_uchar4(r,g,b,a), surf, x* sizeof(uchar4), (height-1)-y);

    }

}

// void write_solid_image() {
//     unsigned char* image = new unsigned char[width * height * 3];
//     unsigned char* image_device;
//
//     cudaMalloc(&image_device, sizeof(unsigned char) * width * height * 3);
//
//     dim3 ThreadsPerBlock(16, 16);
//     dim3 GridDim((width + 15) / 16, (height + 15) / 16 );
//
//     FragmentKernel<<<GridDim, ThreadsPerBlock>>>(image_device);
//
//     cudaMemcpy(image,image_device,sizeof(unsigned char) * width * height * 3, cudaMemcpyDeviceToHost);
//
//     cudaDeviceSynchronize();
//
//
//     stbi_write_jpg(getNextRenderFileName("../Renders").c_str(),width , height, 3, image, 300);
// }
__global__
void Debugkernel(cudaSurfaceObject_t surf,unsigned int width, unsigned int height) {
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        // image[(y * width + x) * 3 + 0] = (unsigned int)(fcolor.r * 255.0f);
        // image[(y * width + x) * 3 + 1] = (unsigned int)(fcolor.g * 255.0f);
        // image[(y * width + x) *3 + 2] = (unsigned int)(fcolor.b * 255.0f);

        unsigned char r = (unsigned char)(0.f);
        unsigned char g = (unsigned char)(0.f);
        unsigned char b = (unsigned char)(0.f);
        unsigned char a = 255;

        surf2Dwrite(make_uchar4(r,g,b,a), surf, x* sizeof(uchar4), (height-1)-y);

    }
}
void launchFragment(cudaSurfaceObject_t surf,unsigned int width, unsigned int height, float time, const float* output,const unsigned int output_size, const PrimitiveType* output_disc, const unsigned int outputDiscSize) {
    dim3 ThreadsPerBlock(16, 16);
    dim3 GridDim((width + 15) / 16, (height + 15) / 16 );

    float* output_device;
    PrimitiveType* output_disc_device;
    cudaMalloc(&output_device, sizeof(float) * output_size);
    cudaMalloc(&output_disc_device, sizeof(PrimitiveType) * outputDiscSize);

    cudaMemcpy(output_device, output, sizeof(float) * output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(output_disc_device, output_disc, sizeof(PrimitiveType) * outputDiscSize, cudaMemcpyHostToDevice);
    FragmentKernel<<<GridDim, ThreadsPerBlock>>>(surf,width, height, time, output_device,output_size, output_disc_device, outputDiscSize);
    cudaDeviceSynchronize();

    cudaFree(output_device);
    cudaFree(output_disc_device);
}
