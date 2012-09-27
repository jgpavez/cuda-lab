#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

int main(void)
{
    int N = 100000;
    thrust::host_vector<int> h_data(N);
    thrust::device_vector<int> d_data(N);

    //Method 1 (one cudaMemcpy per element, slow)
   // for (int i = 0; i < N; i++)
   //     d_data[i] = i;

    //Method 2 (one cudaMemcpy per entire array, faster)
    for (int i = 0; i < N; i++)
        h_data[i] = i;
    thrust::copy(h_data.begin(), h_data.end(), d_data.begin());

    return 0;
}
