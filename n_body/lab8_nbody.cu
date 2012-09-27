#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

int   const N       = 32;
int   const THREADS = 12;
int   const BSZ     = 3;
float const EPS2    = 0.0001;

double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

__global__ void direct_sh(float4 *sourceGlob, float *targetGlob)
{
    __shared__ float4 p_sh[THREADS];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int I = bx * THREADS + tx;
    float dx,dy,dz,r;
    float4 p1 = sourceGlob[I];
    float p = - p1.w/ sqrtf(EPS2);
    float4 p2;
    for (unsigned int m = 0; m < BSZ-1; m++ ){
        p_sh[tx] = sourceGlob[m * THREADS + tx];
        __syncthreads();
#pragma unroll 10
        for (unsigned int i = 0; i < THREADS; i++){
            p2 = p_sh[i];
            dx = p1.x - p2.x;
            dy = p1.y - p2.y;
            dz = p1.z - p2.z;
            r = sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
            p += p1.w / r; 
        }
        __syncthreads();
    }
    int m = BSZ-1;
    p_sh[tx] = sourceGlob[m*THREADS + tx];
    __syncthreads();
    int lastDim = N%THREADS;
    for ( unsigned int i = 0; i < lastDim; i++){
         p2 = p_sh[i];
         dx = p1.x - p2.x;
         dy = p1.y - p2.y;
         dz = p1.z - p2.z;
         r = sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
         p += p1.w / r; 
    }
    __syncthreads();

    targetGlob[I] = p;
}

__global__ void direct(float4 *sourceGlob, float *targetGlob) 
{	/*** wirte your kernel here! *****/
    int tx = threadIdx.x;
    float dx,dy,dz,r;
    float4 p1,p2;
    p1 = sourceGlob[tx];
    float p = - p1.w / sqrtf(EPS2);
    for ( int j = 0; j < N; j++ ){
        p2 = sourceGlob[j];
        dx = p1.x - p2.x;
        dy = p1.y - p2.y;
        dz = p1.z - p2.z;
        r = sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
        p += p1.w / r;
    }
    targetGlob[tx] = p; 
}
int main() {
  float4 *sourceHost,*sourceDevc;
  float  *targetHost,*targetDevc;
// Allocate memory on host and device
  sourceHost = (float4*)     malloc( N*sizeof(float4) );
  targetHost = (float *)     malloc( N*sizeof(float ) );
  cudaMalloc(  (void**) &sourceDevc, N*sizeof(float4) );
  cudaMalloc(  (void**) &targetDevc, N*sizeof(float ) );
// Initialize
  for( int i=0; i<N; i++ ) {
    sourceHost[i].x = rand()/(1.+RAND_MAX);
    sourceHost[i].y = rand()/(1.+RAND_MAX);
    sourceHost[i].z = rand()/(1.+RAND_MAX);
    sourceHost[i].w = 1.0/N;
  }
// Direct summation on device
  cudaMemcpy(sourceDevc,sourceHost,N*sizeof(float4),cudaMemcpyHostToDevice);
  
  double start = get_time();
  direct_sh<<< int(N-0.5/THREADS)+1, THREADS >>>(sourceDevc,targetDevc);
  double stop = get_time();

  cudaMemcpy(targetHost,targetDevc,N*sizeof(float ),cudaMemcpyDeviceToHost);

  double time = stop - start;

  std::cout<<"Kernel execution time: "<<time<<std::endl;

// Direct summation on host
  float dx,dy,dz,r;
  for( int i=0; i<N; i++ ) {
    float p = - sourceHost[i].w / sqrtf(EPS2);
    for( int j=0; j<N; j++ ) {
      dx = sourceHost[i].x - sourceHost[j].x;
      dy = sourceHost[i].y - sourceHost[j].y;
      dz = sourceHost[i].z - sourceHost[j].z;
      r = sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
      p += sourceHost[j].w / r;
    }
    printf("%d %f %f\n",i,p,targetHost[i]);
  }
}
