#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

#define WIDTH 800
#define HEIGHT 400
#define NUM_SPHERES 3

struct Vec3{
    float x,y,z;

    __host__ __device__ Vec3(){}
    __host__ __device__ Vec3(float a,float b,float c):x(a),y(b),z(c){}

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x+v.x,y+v.y,z+v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x-v.x,y-v.y,z-v.z);
    }

    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x*t,y*t,z*t);
    }
};

struct Ray{
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray(Vec3 o, Vec3 d):origin(o),direction(d){}
};

struct Sphere{
    Vec3 center;
    float radius;
    Vec3 color;
};

/////////////////////////////////////////////////////
// Ray-Sphere Intersection
/////////////////////////////////////////////////////

__host__ __device__ bool hit_sphere(Sphere sphere, Ray r, float &t)
{
    Vec3 oc = r.origin - sphere.center;

    float a = r.direction.x*r.direction.x +
              r.direction.y*r.direction.y +
              r.direction.z*r.direction.z;

    float b = 2.0f*(oc.x*r.direction.x +
                    oc.y*r.direction.y +
                    oc.z*r.direction.z);

    float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z -
              sphere.radius*sphere.radius;

    float discriminant = b*b - 4*a*c;

    if(discriminant > 0)
    {
        t = (-b - sqrtf(discriminant))/(2.0f*a);
        return true;
    }

    return false;
}

/////////////////////////////////////////////////////
// Color computation
/////////////////////////////////////////////////////

__host__ __device__ Vec3 ray_color(Ray r, Sphere *spheres)
{
    float closest = 1e20;
    Vec3 color;

    for(int i=0;i<NUM_SPHERES;i++)
    {
        float t;

        if(hit_sphere(spheres[i],r,t))
        {
            if(t < closest)
            {
                closest = t;
                color = spheres[i].color;
            }
        }
    }

    if(closest < 1e20)
        return color;

    float unit = (r.direction.y + 1.0f)*0.5f;

    return Vec3(1,1,1)*(1-unit) + Vec3(0.5,0.7,1.0)*unit;
}

/////////////////////////////////////////////////////
// GPU Kernel
/////////////////////////////////////////////////////

__global__ void render_gpu(Vec3 *fb, Sphere *spheres)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x>=WIDTH || y>=HEIGHT)
        return;

    int pixel = y*WIDTH + x;

    float u = float(x)/WIDTH;
    float v = float(y)/HEIGHT;

    Ray r(Vec3(0,0,0), Vec3(u*2-1, v*2-1, -1));

    fb[pixel] = ray_color(r,spheres);
}

/////////////////////////////////////////////////////
// CPU Version
/////////////////////////////////////////////////////

void render_cpu(Vec3 *fb, Sphere *spheres)
{
    for(int y=0;y<HEIGHT;y++)
    {
        for(int x=0;x<WIDTH;x++)
        {
            int pixel = y*WIDTH + x;

            float u = float(x)/WIDTH;
            float v = float(y)/HEIGHT;

            Ray r(Vec3(0,0,0), Vec3(u*2-1, v*2-1, -1));

            fb[pixel] = ray_color(r,spheres);
        }
    }
}

/////////////////////////////////////////////////////

int main()
{
    int pixels = WIDTH*HEIGHT;

    Vec3 *fb_cpu = new Vec3[pixels];
    Vec3 *fb_gpu;

    cudaMallocManaged(&fb_gpu,pixels*sizeof(Vec3));

    Sphere *spheres;
    cudaMallocManaged(&spheres,NUM_SPHERES*sizeof(Sphere));

    spheres[0] = {Vec3(0,0,-1),0.5,Vec3(1,0,0)};
    spheres[1] = {Vec3(-1,0,-2),0.5,Vec3(0,1,0)};
    spheres[2] = {Vec3(1,0,-2),0.5,Vec3(0,0,1)};

    //////////////////////////////////////////////////
    // CPU Timing
    //////////////////////////////////////////////////

    auto cpu_start = std::chrono::high_resolution_clock::now();

    render_cpu(fb_cpu,spheres);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double>(cpu_end-cpu_start).count();

    //////////////////////////////////////////////////
    // GPU Timing
    //////////////////////////////////////////////////

    dim3 threads(16,16);
    dim3 blocks(WIDTH/16+1,HEIGHT/16+1);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    render_gpu<<<blocks,threads>>>(fb_gpu,spheres);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms,start,stop);

    double gpu_time = gpu_time_ms/1000.0;

    //////////////////////////////////////////////////
    // Speedup
    //////////////////////////////////////////////////

    double speedup = cpu_time/gpu_time;

    std::cout<<"CPU Time: "<<cpu_time<<" seconds\n";
    std::cout<<"GPU Time: "<<gpu_time<<" seconds\n";
    std::cout<<"Speedup: "<<speedup<<"x\n";

    ////////////////////////////////////////////////
    // Write Image
    ////////////////////////////////////////////////

    std::ofstream image("output.ppm");

    image << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";

    for(int j=HEIGHT-1;j>=0;j--)
    {
        for(int i=0;i<WIDTH;i++)
        {
            int pixel = j*WIDTH + i;

            int ir = int(255.99 * fb_gpu[pixel].x);
            int ig = int(255.99 * fb_gpu[pixel].y);
            int ib = int(255.99 * fb_gpu[pixel].z);

            image << ir << " "
                  << ig << " "
                  << ib << "\n";
        }
    }

    image.close();

    cudaFree(fb_gpu);
    cudaFree(spheres);
    delete[] fb_cpu;
}
