#include "fragment_kernel_api.h"

//Do it on the kernel

#define MAX_STEPS 200
__device__ float sphereSDF(const ray::vec3& p, const ray::vec3& l, const float radius) {
    return ray::length(p-l) - radius;
}

__device__ float cubeSDF(const ray::vec3& p,
const ray::vec3& l,
const ray::vec3& r) {
    ray::vec3 q(abs(p - l) - r);
    return ray::length(ray::max(q, 0.f)) + min( ray::compMax(q) , 0.0);
}

__device__ float coneSDF(const ray::vec3& p, const ray::vec3& l,const float radius, const float height) {

    float k = (radius/height);
    return  max(sqrt((p.x -l.x)*(p.x-l.x) + (p.z - l.z) *(p.z-l.z)) - ( (radius - ((p.y- l.y) * k ) / sqrtf(1+ k*k) ) ), -((p.y - l.y) ) );
}
__device__ float coneSDF2( const ray::vec3& p, const ray::vec3& l ,const ray::vec3& c, float h )
{
    float q = sqrt((p.x -l.x)*(p.x-l.x));
    return max(ray::dot(c,ray::vec3(q,p.y,0)),-h-p.y);
}
__device__ float scene2(const ray::vec3& p) {


    ray::vec3 np = ray::mod(p ,1.f) - .5;

    //np = np + ray::vec3(sinf(p.z * 0.2f) * 0.1f,0.f,0.f);
    //np.y += sinf(p.z) * 0.15f;
    float wave = sinf(abs(p.z) * 100.f) * 0.02f + sinf(abs(p.z) * 10.f) * 0.04f;
    float cube = (cubeSDF(np + ray::vec3(wave * 1.1f,wave * 0.97f,wave), ray::vec3(0.f, 0.f, 0.f), ray::vec3(0.15, 0.15, 0.15)) -0.025f) * 0.25f;
    float sphere =  sphereSDF( np+ ray::vec3(0.0,wave,0.0), ray::vec3(0.f, 0.f, 0.f), 0.25f) * 0.25f;


    return min(cube,sphere) ;


}

__device__ float scene1(const ray::vec3& p) {
    return min(
            max(
            -cubeSDF(p, ray::vec3(-1.0,0.5,-0.5), ray::vec3(0.25,0.5,0.5)),
            cubeSDF(p, ray::vec3(-1.0,0.0,0.0), ray::vec3(0.5,0.5,0.5))
            ),
            max(
            -cubeSDF(p, ray::vec3(1.0,-0.5,-0.5), ray::vec3(0.25,0.5,0.5)),
            cubeSDF(p, ray::vec3(1.0,0.0,0.0),ray::vec3( 0.5,0.5,0.5))
                )

        ) - 0.05f;
}


__device__ ray::vec3 mandelbulbSDF(const ray::vec3& p, const int N, const float pw) {

    ray::vec3 zold(0.f,0.f,0.f);
        ray::vec3 znew(0.f,0.f,0.f);

    float dr = 1.0f;
    for (unsigned int i = 0; i < N; i++) {

        if (ray::length(zold) > 8.f) {
            break;
        }
        znew = (zold ^ pw) + p;
        dr = (pw * powf(length(zold), pw-1.f) * dr) + 1.f;
        zold = znew;
    }

    float d = 0.5f * (ray::length(znew) * logf(ray::length(znew)) )/(dr+EPSILON);
    return ray::vec3(d, length(zold) - floorf(length(zold)),0.f);
}
__device__ ray::vec3 scene3(const ray::vec3& p, const ray::vec3& loc, const ray::vec3& rot, const float n) {
    return mandelbulbSDF( ray::rotate(ray::rotate(ray::rotate( ((p - ray::vec3(0.f,0.f,-3.3f)) -loc),0,rot.x * (PI/180.f)),1,rot.y * (PI/180.f)),2,rot.z * (PI/180.f))   ,
        8,n);

}

__global__
void FragmentKernel(cudaSurfaceObject_t surf, float time, unsigned int width, unsigned int height,const ray::vec3 loc,const ray::vec3 rot, const float n) {
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
    float d;
    ray::vec3 p;
    float vy = 0.0f;
    for (i = 0; i < MAX_STEPS; i++) {
        //Calculate position after marching according to line
        p = ro + (rd * t);

        //Check SDF
        ray::vec3 o =  scene3(p,loc,rot,n);
        d = o.x;
        vy = o.y;
        //March by this unit
        if (abs(d) < 0.001f || t > 100.0f) { break;}
        t+=d;
    }
    float diffuse = 0.f;
    float intensity = 0.9f;
    const float epsilon = 0.0001f;
    float ambient = 0.0f;
    float edge = 0.0f;
    float customE =0.0f;
    float rim;
    float mask = 0.0f;
    if (d < 0.01f) {
        mask =1.f;
        //we hit a surface, calculate its normal
        ray::vec3 normal(
            scene3(p + ray::vec3(epsilon,0.0f,0.0f),loc,rot,n).x - scene3(p - ray::vec3(epsilon,0.0f,0.0f),loc,rot,n).x,
           scene3(p + ray::vec3(0.0f,epsilon,0.0f),loc,rot,n).x - scene3(p - ray::vec3(0.0f,epsilon,0.0f),loc,rot,n).x,
           scene3(p + ray::vec3(0.0f,0.0f,epsilon),loc,rot,n).x - scene3(p - ray::vec3(0.0f,0.0f,epsilon),loc,rot,n).x);
        normal = ray::normalize(normal);

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
    ray::vec3 color(fabsf(sinf(vy * 2.f * PI * 0.1f)),fabsf(sinf(vy * 2 * PI * 2.f)),fabsf(sinf(vy * 2 * PI*0.5f)));
    ray::vec3 base_color(0.876f,0.9f,1.0f);
    ray::vec3 fcolor =  ((color*0.75) * diffuse) + (base_color * ray::clamp(rim,0.f,1.f));
    fcolor.x =ray::clamp(fcolor.x,0.f,1.f);
    fcolor.y =ray::clamp(fcolor.y,0.f,1.f);
    fcolor.z =ray::clamp(fcolor.z,0.f,1.f);
    fcolor =fcolor * mask;
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

void launchFragment(cudaSurfaceObject_t surf,float time, unsigned int width, unsigned int height,const ray::vec3& loc, const ray::vec3& rot, const float n) {
    dim3 ThreadsPerBlock(16, 16);
    dim3 GridDim((width + 15) / 16, (height + 15) / 16 );

    FragmentKernel<<<GridDim, ThreadsPerBlock>>>(surf,time, width, height,loc,rot,n);
    cudaDeviceSynchronize();
}
