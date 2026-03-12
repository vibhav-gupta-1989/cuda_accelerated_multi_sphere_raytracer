#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>

#define WIDTH 800
#define HEIGHT 400
#define NUM_SPHERES 1000
#define MAX_NODES (NUM_SPHERES + 1)

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

    __host__ __device__ Ray(Vec3 o,Vec3 d):origin(o),direction(d){}
};

struct Sphere{
    Vec3 center;
    float radius;
    Vec3 color;
};

struct BVHNode{
    Vec3 min;
    Vec3 max;
    int sphere;
};

////////////////////////////////////////////////////////////
// Sphere intersection
////////////////////////////////////////////////////////////

__host__ __device__
bool hit_sphere(Sphere s,Ray r,float &t)
{
    Vec3 oc=r.origin-s.center;

    float a=r.direction.x*r.direction.x +
            r.direction.y*r.direction.y +
            r.direction.z*r.direction.z;

    float b=2*(oc.x*r.direction.x +
               oc.y*r.direction.y +
               oc.z*r.direction.z);

    float c=oc.x*oc.x + oc.y*oc.y + oc.z*oc.z -
            s.radius*s.radius;

    float disc=b*b-4*a*c;

    if(disc>0)
    {
        t=(-b-sqrtf(disc))/(2*a);
        return true;
    }

    return false;
}

////////////////////////////////////////////////////////////
// AABB intersection (CPU + GPU)
////////////////////////////////////////////////////////////

__host__ __device__
bool hit_box(Vec3 min,Vec3 max,Ray r)
{
    float tmin=(min.x-r.origin.x)/r.direction.x;
    float tmax=(max.x-r.origin.x)/r.direction.x;

    if(tmin>tmax){float tmp=tmin;tmin=tmax;tmax=tmp;}

    float tymin=(min.y-r.origin.y)/r.direction.y;
    float tymax=(max.y-r.origin.y)/r.direction.y;

    if(tymin>tymax){float tmp=tymin;tymin=tymax;tymax=tmp;}

    if((tmin>tymax)||(tymin>tmax))
        return false;

    return true;
}

////////////////////////////////////////////////////////////
// CPU BVH traversal
////////////////////////////////////////////////////////////

Vec3 ray_color_cpu(Ray r,Sphere* spheres,BVHNode* nodes)
{
    float closest=1e20;
    Vec3 color;
    bool hit=false;

    for(int i=1;i<=NUM_SPHERES;i++)
    {
        if(!hit_box(nodes[i].min,nodes[i].max,r))
            continue;

        float t;

        if(hit_sphere(spheres[nodes[i].sphere],r,t))
        {
            if(t<closest)
            {
                closest=t;
                color=spheres[nodes[i].sphere].color;
                hit=true;
            }
        }
    }

    if(hit) return color;

    float unit=(r.direction.y+1.0f)*0.5f;

    return Vec3(1,1,1)*(1-unit)+Vec3(0.5,0.7,1)*unit;
}

////////////////////////////////////////////////////////////
// GPU BVH traversal
////////////////////////////////////////////////////////////

__device__
Vec3 ray_color_gpu(Ray r,Sphere* spheres,BVHNode* nodes)
{
    float closest=1e20;
    Vec3 color;
    bool hit=false;

    for(int i=1;i<=NUM_SPHERES;i++)
    {
        if(!hit_box(nodes[i].min,nodes[i].max,r))
            continue;

        float t;

        if(hit_sphere(spheres[nodes[i].sphere],r,t))
        {
            if(t<closest)
            {
                closest=t;
                color=spheres[nodes[i].sphere].color;
                hit=true;
            }
        }
    }

    if(hit) return color;

    float unit=(r.direction.y+1.0f)*0.5f;

    return Vec3(1,1,1)*(1-unit)+Vec3(0.5,0.7,1)*unit;
}

////////////////////////////////////////////////////////////
// CPU render
////////////////////////////////////////////////////////////

void render_cpu(Vec3* fb,Sphere* spheres,BVHNode* nodes)
{
    for(int y=0;y<HEIGHT;y++)
    {
        for(int x=0;x<WIDTH;x++)
        {
            int pixel=y*WIDTH+x;

            float u=float(x)/WIDTH;
            float v=float(y)/HEIGHT;

            Ray r(Vec3(0,0,0),Vec3(u*2-1,v*2-1,-1));

            fb[pixel]=ray_color_cpu(r,spheres,nodes);
        }
    }
}

////////////////////////////////////////////////////////////
// GPU render
////////////////////////////////////////////////////////////

__global__
void render_gpu(Vec3* fb,Sphere* spheres,BVHNode* nodes)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;

    if(x>=WIDTH||y>=HEIGHT) return;

    int pixel=y*WIDTH+x;

    float u=float(x)/WIDTH;
    float v=float(y)/HEIGHT;

    Ray r(Vec3(0,0,0),Vec3(u*2-1,v*2-1,-1));

    fb[pixel]=ray_color_gpu(r,spheres,nodes);
}

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////

int main()
{
    int pixels=WIDTH*HEIGHT;

    Vec3* fb_cpu=new Vec3[pixels];
    Vec3* fb_gpu;

    cudaMallocManaged(&fb_gpu,pixels*sizeof(Vec3));

    Sphere* spheres;
    cudaMallocManaged(&spheres,NUM_SPHERES*sizeof(Sphere));

    BVHNode* nodes;
    cudaMallocManaged(&nodes,MAX_NODES*sizeof(BVHNode));

    ////////////////////////////////////////////////////////
    // Generate spheres
    ////////////////////////////////////////////////////////

    for(int i=0;i<NUM_SPHERES;i++)
    {
        float x=((rand()%200)-100)/50.0f;
        float y=((rand()%200)-100)/50.0f;
        float z=-((rand()%200)/30.0f+1);

        spheres[i].center=Vec3(x,y,z);
        spheres[i].radius=0.1;

        spheres[i].color=Vec3(
            (rand()%100)/100.0f,
            (rand()%100)/100.0f,
            (rand()%100)/100.0f);
        
        //std::cout << "Sphere " << x << "," << y << "," << z << std::endl;
    }

    ////////////////////////////////////////////////////////
    // Build BVH leaves
    ////////////////////////////////////////////////////////

    for(int i=0;i<NUM_SPHERES;i++)
    {
        Vec3 c=spheres[i].center;
        float r=spheres[i].radius;

        nodes[i+1].min=Vec3(c.x-r,c.y-r,c.z-r);
        nodes[i+1].max=Vec3(c.x+r,c.y+r,c.z+r);
        nodes[i+1].sphere=i;
    }

    ////////////////////////////////////////////////////////
    // CPU timing
    ////////////////////////////////////////////////////////

    auto cpu_start=std::chrono::high_resolution_clock::now();

    render_cpu(fb_cpu,spheres,nodes);

    auto cpu_end=std::chrono::high_resolution_clock::now();

    double cpu_time=
    std::chrono::duration<double>(cpu_end-cpu_start).count();

    ////////////////////////////////////////////////////////
    // GPU timing
    ////////////////////////////////////////////////////////

    dim3 threads(16,16);
    dim3 blocks((WIDTH+15)/16,(HEIGHT+15)/16);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    render_gpu<<<blocks,threads>>>(fb_gpu,spheres,nodes);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms,start,stop);

    double gpu_time=gpu_ms/1000.0;

    ////////////////////////////////////////////////////////

    std::cout<<"CPU Time: "<<cpu_time<<" seconds\n";
    std::cout<<"GPU Time: "<<gpu_time<<" seconds\n";
    std::cout<<"Speedup: "<<cpu_time/gpu_time<<"x\n";

    ////////////////////////////////////////////////////////
    // Write image
    ////////////////////////////////////////////////////////

    std::ofstream image("output.ppm");

    image<<"P3\n"<<WIDTH<<" "<<HEIGHT<<"\n255\n";

    for(int j=HEIGHT-1;j>=0;j--)
    {
        for(int i=0;i<WIDTH;i++)
        {
            int pixel=j*WIDTH+i;

            int ir=int(255.99*fb_gpu[pixel].x);
            int ig=int(255.99*fb_gpu[pixel].y);
            int ib=int(255.99*fb_gpu[pixel].z);

            image<<ir<<" "<<ig<<" "<<ib<<"\n";
        }
    }

    image.close();

    cudaFree(fb_gpu);
    cudaFree(spheres);
    cudaFree(nodes);

    delete[] fb_cpu;
}