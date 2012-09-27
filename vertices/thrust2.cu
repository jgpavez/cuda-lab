#include <thrust/device_vector.h>
#include <thrust/transform.h>

struct triple
{
    // functor puede ser utilizado por el host o el device
    __host__ __device__
    int operator()(int x)
    {
        return 3 * x;
    }
};

int main(void)
{
    thrust::device_vector<int> input(4);
    input[0] = 10; input[1] = 20; input[2] = 30; input[3] = 40;
    thrust::device_vector<int> output(4);
    thrust::transform(input.begin(), input.end(), output.begin(), triple());

    for (int i = 0; i < output.size(); i++)
        std::cout<<output[i]<<std::endl;
    return 0;
}
