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

__device__ void parseExpression(const ray::vec3& p, const float* output, const PrimitiveType* output_disc, float* data, float* d_values, const unsigned int output_size,const unsigned int outputDiscSize, float* minD, unsigned int* minIndex) {

    //We want to parse the postfix expression at hand using a stack
    ray::Stack<float,20> distance_stack;
    float currentData[1];
    size_t s;
    unsigned int dataOffset =0;
    for (unsigned int i = 0; i < outputDiscSize; i++) {
        switch (output_disc[i]) {
            case PrimitiveType::CUBE:
                Cube::CubeSDFF(p,output+dataOffset+1, &s, currentData);
                d_values[(unsigned int)*(output+dataOffset)] = currentData[0];

                if (minD != nullptr && minIndex != nullptr) {
                    if (currentData[0] < *minD) {
                        *minD = currentData[0];
                        *minIndex = (unsigned int)*(output+dataOffset);
                    }
                }
                break;

            case PrimitiveType::SPHERE:
                Sphere::SphereSDFF(p,output+dataOffset+1, &s, currentData);
                d_values[(unsigned int)*(output+dataOffset)] = currentData[0];

                if (minD != nullptr && minIndex != nullptr) {
                    if (currentData[0] < *minD) {
                        *minD = currentData[0];
                        *minIndex = (unsigned int)*(output+dataOffset);
                    }
                }
                break;

            case PrimitiveType::MANDELBROT:
                Mandelbulb::MandelbulbSDFF(p,output+dataOffset+1, &s, currentData);
                d_values[(unsigned int)*(output+dataOffset)] = currentData[0];

                if (minD != nullptr && minIndex != nullptr) {
                    if (currentData[0] < *minD) {
                        *minD = currentData[0];
                        *minIndex = (unsigned int)*(output+dataOffset);
                    }
                }
                break;
            case PrimitiveType::LINE:
                Line::LineSDFF(p,output+dataOffset+1, &s, currentData);
                d_values[(unsigned int)*(output+dataOffset)] = currentData[0];

                if (minD != nullptr && minIndex != nullptr) {
                    if (currentData[0] < *minD) {
                        *minD = currentData[0];
                        *minIndex = (unsigned int)*(output+dataOffset);
                    }
                }
                break;
            case PrimitiveType::UNION:
                Union::UnionSDFF(distance_stack.pop(),distance_stack.pop(), &s, currentData);
                d_values[(unsigned int)*(output+dataOffset)] = currentData[0];
                break;

            case PrimitiveType::INTERSECT:
                Intersect::IntersectSDFF(distance_stack.pop(),distance_stack.pop(), &s, currentData);
                d_values[(unsigned int)*(output+dataOffset)] = currentData[0];
                break;

            default:
                break;
        }
        distance_stack.push(currentData[0]);
        dataOffset += getPrimSize(output_disc[i]);

    }

    *data = distance_stack.pop();
}

__device__ void parseExpressionNorm(const ray::vec3& p, const float* output, const PrimitiveType* output_disc, ray::vec3* data, const float* d_values, const unsigned int output_size,const unsigned int outputDiscSize) {
    ray::Stack<ray::vec3,20> norm_stack;
    ray::Stack<float,20> d_stack;
    ray::vec3 currentData;
    size_t s;
    unsigned int dataOffset =0;
    float currentD = 0;
    for (unsigned int i = 0; i < outputDiscSize; i++) {
        switch (output_disc[i]) {
            case PrimitiveType::CUBE:
                Cube::CubeSDFFNorm(p,output+dataOffset+1,&currentData);
                d_stack.push(d_values[(unsigned int)*(output+dataOffset)]);
                break;

            case PrimitiveType::SPHERE:
                Sphere::SphereSDFFNorm(p,output+dataOffset+1, &currentData);
                d_stack.push(d_values[(unsigned int)*(output+dataOffset)]);
                break;

            case PrimitiveType::MANDELBROT:
                Mandelbulb::MandelbulbSDFFNorm(p,output+dataOffset+1,&currentData);
                d_stack.push(d_values[(unsigned int)*(output+dataOffset)]);
                break;
            case PrimitiveType::LINE:
                Line::LineSDFFNorm(p,output+dataOffset+1,&currentData);
                d_stack.push(d_values[(unsigned int)*(output+dataOffset)]);
                break;
            case PrimitiveType::UNION:
                Union::UnionSDFFNorm(d_stack.pop(),d_stack.pop(), norm_stack.pop(), norm_stack.pop() ,&currentData, &currentD);
                d_stack.push(currentD);
                break;

            case PrimitiveType::INTERSECT:
                Intersect::IntersectSDFFNorm(d_stack.pop(),d_stack.pop(), norm_stack.pop(), norm_stack.pop() ,&currentData, &currentD);
                d_stack.push(currentD);
                break;

            default:
                break;
        }
        norm_stack.push(currentData);
        dataOffset += getPrimSize(output_disc[i]);

    }

    *data = norm_stack.pop();
}

__device__ void parseExpressionNormAll(const ray::vec3& p, const float* output, const PrimitiveType* output_disc, ray::vec3* data, float* d_values, const unsigned int output_size,const unsigned int outputDiscSize) {
    float dxp,dxn,dyp,dyn,dzp,dzn;
    ray::vec3 pxp = p + ray::vec3(EPSILON,0.0f,0.0f);
    ray::vec3 pxn = p - ray::vec3(EPSILON,0.0f,0.0f);

    ray::vec3 pyp = p + ray::vec3(0.0f,EPSILON,0.0f);
    ray::vec3 pyn = p - ray::vec3(0.0f,EPSILON,0.0f);

    ray::vec3 pzp = p +  ray::vec3(0.0f,0.0f,EPSILON);
    ray::vec3 pzn = p -  ray::vec3(0.0f,0.0f,EPSILON);

    parseExpression(pxp, output, output_disc, &dxp, d_values, output_size, outputDiscSize, nullptr, nullptr);
    parseExpression(pxn, output, output_disc, &dxn, d_values, output_size, outputDiscSize, nullptr, nullptr);

    parseExpression(pyp, output, output_disc, &dyp, d_values, output_size, outputDiscSize, nullptr, nullptr);
    parseExpression(pyn, output, output_disc, &dyn, d_values, output_size, outputDiscSize, nullptr, nullptr);

    parseExpression(pzp, output, output_disc, &dzp, d_values, output_size, outputDiscSize, nullptr, nullptr);
    parseExpression(pzn, output, output_disc, &dzn, d_values, output_size, outputDiscSize, nullptr, nullptr);

    *data = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );
}
__global__
void FragmentKernel(cudaSurfaceObject_t surf,unsigned int width, unsigned int height, float time, const float* output, const unsigned int output_size, const PrimitiveType* output_disc, const unsigned int outputDiscSize, const ray::vec3* lightSource) {
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    float aspectRatio = (float)width / (float)height;

    float u = ((((float)x/width) * 2.0f ) -1.0f ) * aspectRatio;
    float v = ((((float)(height - y)/height) * 2.0f) - 1.0f);

    ray::vec3 ro(0.f,0.f,-5.0f);
    ray::vec3 rd(u,v,1.0f);
    rd = ray::normalize(rd);

    float t = 0.0f;

    unsigned int i;
    float data = 0;
    ray::vec3 normal(0.f,0.f,0.f);
    ray::vec3 p;
    float dValues[20];
    float minD = 100.f;
    unsigned int minIndex = 0;
    unsigned int dValuesSize = 0;
    size_t dataSize;
    for (i = 0; i < MAX_STEPS; i++) {
        //Calculate position after marching according to line
        p = ro + (rd * t);
        //Check SDF
        parseExpression(p,output, output_disc, &data, dValues,output_size ,outputDiscSize, &minD, &minIndex);
        //printf("%f \n", data[0]);
        //March by this unit
        if (abs(data) < 0.001f || t > 1000.0f) { break;}
        t+=data;
    }
    float specularPower = 32.0f;
    float intensity = 0.7f;
    float ambient = 0.3f;
    float reflection_intensity = 0.2f;

    float diffuse = 0.f;
    float rim = 0.f;
    float mask = 0.0f;
    float specularity = 0.0f;
    ray::vec3 fcolor;
    ray::vec3 baseColor(0.0f,0.0f,0.0f);
    ray::vec3 ambientColor(1.0f,1.0f,1.0f);

    if (minIndex == 0 && minD < 0.01f) {
        fcolor = ray::vec3(1.f,1.f,1.f);
    }else {
        if (data < 0.01f) {
            switch (minIndex) {
                case 0:
                    baseColor = ray::vec3(1.0f, 1.0f, 1.0f);
                    break;
                case 1: // cyan
                    baseColor = ray::vec3(0.0f, 1.0f, 1.0f);
                    break;
                case 2: // green
                    baseColor = ray::vec3(0.0f, 1.0f, 0.0f);
                    break;
                case 3: // magenta
                    baseColor = ray::vec3(1.0f, 0.0f, 1.0f);
                    break;
                case 4: // orange
                    baseColor = ray::vec3(1.0f, 0.5f, 0.0f);
                    break;
                case 5: // blue
                    baseColor = ray::vec3(0.0f, 0.0f, 1.0f);
                    break;
                case 6: // pink
                    baseColor = ray::vec3(1.0f, 0.4f, 0.7f);
                    break;
                case 7: // purple
                    baseColor = ray::vec3(0.6f, 0.2f, 0.8f);
                    break;
                default:
                    baseColor = ray::vec3(1.0f, 1.0f, 1.0f);
                    break;
            }
            mask =1.f;
            parseExpressionNormAll(p, output, output_disc, &normal, dValues, output_size ,outputDiscSize);
            //we hit a surface, calculate its normal
            //calculate dot product between normal and a light source
            ray::vec3 incident = ray::normalize(*lightSource - p);
            diffuse = ray::dot(incident, normal ) * intensity;

            ray::vec3 viewing = ray::normalize(p - ro);
            ray::vec3 reflected = ray::reflect(incident, normal);
            specularity =  powf( max(ray::dot(viewing, reflected),0.f), specularPower);
            rim = pow(1.0f - max(ray::dot(normal, viewing), 0.0f), 32.0f);
            fcolor =  (mask * ray::vec3(1.f,1.f,1.f) * baseColor * (diffuse + specularity + ambient + (rim * -0.01f)) );

            ray::vec3 new_ro = p;
            float new_t = 0.01f;
            reflected = reflected * -1.f;
            //let's try adding actual reflections
            for (unsigned int i =0; i < 300; i++) {
                p = new_ro + (reflected * new_t);
                parseExpression(p,output, output_disc, &data, dValues,output_size ,outputDiscSize, &minD, &minIndex);
                if (new_t > 100.0f) { break;}
                new_t+=data;
            }
             float reflected_diffuse = 0.f;
             float reflected_specularity = 0.f;

                 mask =1.f;
                 parseExpressionNormAll(p, output, output_disc, &normal, dValues, output_size ,outputDiscSize);
                 //we hit a surface, calculate its normal
                 //calculate dot product between normal and a light source
                 incident = ray::normalize(*lightSource - p);
                 reflected_diffuse = ray::dot(incident, normal) * intensity;

                 viewing = ray::normalize(p - new_ro);
                 reflected = ray::reflect(incident, normal);
                 reflected_specularity =  powf( max(ray::dot(viewing, reflected),0.f), specularPower) * intensity;

                 switch (minIndex) {
                     case 0: // bright yello fo rlight
                         baseColor = ray::vec3(1.0f, 1.0f, 1.0f);
                         break;
                     case 1: // cyan
                         baseColor = ray::vec3(0.0f, 1.0f, 1.0f);
                         break;
                     case 2: // green
                         baseColor = ray::vec3(0.0f, 1.0f, 0.0f);
                         break;
                     case 3: // magenta
                         baseColor = ray::vec3(1.0f, 0.0f, 1.0f);
                         break;
                     case 4: // orange
                         baseColor = ray::vec3(1.0f, 0.5f, 0.0f);
                         break;
                     case 5: // blue
                         baseColor = ray::vec3(0.0f, 0.0f, 1.0f);
                         break;
                     case 6: // pink
                         baseColor = ray::vec3(1.0f, 0.4f, 0.7f);
                         break;
                     case 7: // purple
                         baseColor = ray::vec3(0.6f, 0.2f, 0.8f);
                         break;
                     default:
                         baseColor = ray::vec3(1.0f, 1.0f, 1.0f);
                         break;
                 }
            ray::vec3 reflected_img = (mask * ray::vec3(1.f,1.f,1.f) * baseColor * (ambient + reflected_specularity + reflected_diffuse) );
            //fcolor = ray::clamp(fcolor, 0.0f, 1.0f);
            fcolor = ray::clamp( ((1-reflection_intensity)* fcolor) + (reflection_intensity * reflected_img)  , 0.0f,1.f);

            //edge = (i/(float)MAX_STEPS);
            //edge = edge * edge * 0.7f;
            ;

        }else {
            fcolor = ambient * ambientColor;
        }
    }
    float val =  ray::clamp(t * 0.2f + (i /(float)MAX_STEPS), 0.f, 1.f);
    //fcolor = color * diffuse;
    ray::vec3 debug(val,val,val);
    //fcolor = debug;
    //printf("%f", data);
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
void launchFragment(cudaSurfaceObject_t surf,unsigned int width, unsigned int height, float time, const float* output,const unsigned int output_size, const PrimitiveType* output_disc, const unsigned int outputDiscSize, const ray::vec3* lightSource) {
    dim3 ThreadsPerBlock(16, 16);
    dim3 GridDim((width + 15) / 16, (height + 15) / 16 );
    FragmentKernel<<<GridDim, ThreadsPerBlock>>>(surf,width, height, time, output,output_size, output_disc, outputDiscSize, lightSource);
    cudaDeviceSynchronize();

}

void Allocate(float** output_device, PrimitiveType** output_disc_device, unsigned int output_size, unsigned int outputDiscSize) {
    cudaMalloc(output_device, sizeof(float) * output_size);
    cudaMalloc(output_disc_device, sizeof(PrimitiveType) * outputDiscSize);
}
void Free(float* output_device, PrimitiveType* output_disc_device) {
    cudaFree(output_disc_device);
    cudaFree(output_device);
}
void toDevice(float* output_host, PrimitiveType* output_disc_host,float* output_device, PrimitiveType* output_disc_device, unsigned int output_size, unsigned int outputDiscSize) {
    cudaMemcpy(output_device, output_host, sizeof(float) * output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(output_disc_device, output_disc_host, sizeof(PrimitiveType) * outputDiscSize, cudaMemcpyHostToDevice);
}

void AllocateLightSource(ray::vec3** LightSourceDevice) {
    cudaMalloc(LightSourceDevice, sizeof(float) * 3);
}

void UpdateDeviceLightSource(const ray::vec3* lightSource, ray::vec3* LightSourceDevice) {
    cudaMemcpy(LightSourceDevice, lightSource, sizeof(float) * 3, cudaMemcpyHostToDevice);
}

void FreeDeviceLightSource(ray::vec3* LightSourceDevice) {
    cudaFree(LightSourceDevice);
}