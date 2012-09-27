#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>

struct is_odd
{
    __host__ __device__
    bool operator()(int x)
    {
        return (x%2) == 1;
    }
};
int main(void)
{
    thrust::device_vector<int> data(8);
    data[0] = 6;
    data[1] = 3;
    data[2] = 7;
    data[3] = 5;
    data[4] = 9;
    data[5] = 0;
    data[6] = 8;
    data[7] = 1;

    int N = thrust::count_if(data.begin(), data.end(), is_odd());
    std::cout<<"counted"<<N<<"odd values"<<std::endl;

    thrust::device_vector<int> odds(N);
    thrust::copy_if(data.begin(), data.end(), odds.begin(), is_odd());

    for ( int i = 0; i < odds.size(); i++)
        std::cout<<odds[i]<<std::endl;
     return 0; 
}
