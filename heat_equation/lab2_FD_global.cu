/*** Heat equation with FD in global memory ***/
#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>

texture <float,2> tex_u;
texture <float,2> tex_u_prev;

void checkErrors(char *label)
{
// we need to synchronise first to catch errors due to
// asynchroneous operations that would otherwise
// potentially go unnoticed
cudaError_t err;
err = cudaThreadSynchronize();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
err = cudaGetLastError();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
}

double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

// GPU kernels
__global__ void copy_array (float *u, float *u_prev, int N, int BSZ)
{	
    int index;
    int i = threadIdx.x + BSZ * blockIdx.x;
    int j = threadIdx.y + BSZ * blockIdx.y;
    float u_t = tex2D(tex_u, i, j);


    index = i*N + j;
    u_prev[index] = u_t;
}

__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ)
{	
    int I;
    int i = threadIdx.x + BSZ * blockIdx.x;
    int j = threadIdx.y + BSZ * blockIdx.y;
    if ( i == 0 || j == 0 || i >= N-1 || j >= N-1 ) return;
    
    float c_t, r_t, l_t, u_t, d_t;
    
    c_t = tex2D(tex_u_prev, i, j);
    r_t = tex2D(tex_u_prev, i+1, j);
    l_t = tex2D(tex_u_prev, i-1, j);
    u_t = tex2D(tex_u_prev, i, j-1);
    d_t = tex2D(tex_u_prev, i, j+1);

    I = i*N + j;
    u[I] = c_t + alpha*dt/(h*h) * (r_t + l_t + u_t + d_t - 4*c_t);
}

int main()
{
	// Allocate in CPU
	int N = 128;
	int BLOCKSIZE = 16;

	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	//float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;

	int steps = (int) ceil(time/dt);
	int I;

	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N*N];
	float *u_prev  	= new float[N*N];


	// Generate mesh and intial condition
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i==0) || (j==0)) 
				{u[I] = 200.0f;}
		}
	}

	// Allocate in GPU
    float *u_d;
    float *u_prev_d;
    size_t size = N * N * sizeof(float);
    cudaMalloc((void**) &u_d, size);
    cudaMalloc((void**) &u_prev_d, size);
	
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, tex_u, u_d, desc, N, N, sizeof(float)*N);
    cudaBindTexture2D(NULL, tex_u_prev, u_prev_d, desc, N, N, sizeof(float)*N);
    // Copy to GPU
    cudaMemcpy(u_d, u, size, cudaMemcpyHostToDevice); 
	
    // Loop 
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE); // number of blocks?
	dim3 dimGrid(int(N-0.5)/BLOCKSIZE + 1, int(N-0.5)/BLOCKSIZE + 1); // threads per block?
	
    double start = get_time();
    for (int t=0; t<steps; t++)
	{	copy_array <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, BLOCKSIZE);
		update <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE);
	}
    double stop  = get_time();

    double elapsed = stop - start;
    std::cout<<"time = "<<elapsed<<std::endl;

    cudaMemcpy(u, u_d, size, cudaMemcpyDeviceToHost);

	std::ofstream temperature("temperature_global_t.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			temperature<<x[I]<<"\t"<<y[I]<<"\t"<<u[I]<<std::endl;
		}
		temperature<<"\n";
	}

	temperature.close();

	// Free device
    cudaUnbindTexture(tex_u);
    cudaUnbindTexture(tex_u_prev);
    
	cudaFree(u_d);
	cudaFree(u_prev_d);
}
