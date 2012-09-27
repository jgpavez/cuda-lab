#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <iostream>

struct compare2
{
    __host__ __device__
    bool operator()(float2 a,float2 b){
        return a.x == b.x ? ((a.y == b.y ? false : 
                                (a.y < b.y ? true : false))) :
                            (a.x < b.x ? true : false);
    }
};
struct equal2
{
    __host__ __device__
    bool operator()(float2 a, float2 b){
        return (a.x == b.x && a.y == b.y);
    }
};
int main(void)
{
    // allocate memory for input mesh representation
    thrust::device_vector<float2> input(9);
    
    input[0] = make_float2(0,0);  // First Triangle
    input[1] = make_float2(1,0);
    input[2] = make_float2(0,1);
    input[3] = make_float2(1,0);  // Second Triangle
    input[4] = make_float2(1,1);
    input[5] = make_float2(0,1);
    input[6] = make_float2(1,0);  // Third Triangle
    input[7] = make_float2(2,0);
    input[8] = make_float2(1,1);

    // allocate space for output mesh representation
    thrust::device_vector<float2>       vertices = input;
    thrust::device_vector<unsigned int> indices(input.size());
    compare2 comp;
    equal2 eq;
    
    thrust::sort(vertices.begin(), vertices.end(),comp);
    vertices.erase(thrust::unique(vertices.begin(),vertices.end(), eq),
                    vertices.end());
    thrust::lower_bounds(input.begin(), input.end(),
                        vertices.begin(), vertices.end(),
                        indices.begin(),
                        comp);
                        
                        
  // print output mesh representation
    std::cout << "Output Representation" << std::endl;
    for(size_t i = 0; i < vertices.size(); i++)
    {
        float2 v = vertices[i];
        std::cout << " vertices[" << i << "] = (" << v.x << "," << v.y << ")" << std::endl;
    }
//    for(size_t i = 0; i < indices.size(); i++)
//    {
//        std::cout << " indices[" << i << "] = " << indices[i] << std::endl;
//    }

    return 0;
}

