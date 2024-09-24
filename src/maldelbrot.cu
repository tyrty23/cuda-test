#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
using namespace std::complex_literals;


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

template<typename T>
T* arr2Dto1D(T ** arr,int N,int M){
    T* arr2=new T[N*M];
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            arr2[i*M+j]=arr[i][j];
    
    return arr2;
}

template<typename T>
T** arr1Dto2D(T * arr,int N,int M){
    T** arr2=new bool*[N];
    for (int i = 0; i < N; i++){arr2[i] = new bool [M];}

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            arr2[i][j]=arr[i*M+j];
    return arr2;
}

template<typename T> 
void maldelbrot_cpu(complex<T> ** zn, bool ** map,int N,int M,int max_iter){
    complex<T> z;
    complex<T> c;
    for (int y = 0; y <N; y++){
        for (int x = 0; x < M; x++){
            c=zn[y][x];
            bool inside=true;
            z=zn[y][x];
            for (int i=0;i<max_iter;i++){
                if(abs(z)>2){inside=false;break;}
                z=pow(z,2)+c;
            }
            map[y][x]=inside;
        }
    }
}

template<typename T> 
__global__ void maldelbrot_gpu(complex<T> * zn, bool * map,int N,int M,int max_iter){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int ind = iy*N + ix;   
    if (ix < N && iy < M){
        complex<T> i(0.0, 1.0);
        complex<T> z=zn[ind];
        complex<T> c=zn[ind];

        bool inside=true;
        for (int i=0;i<max_iter;i++){
            if(abs(z)>2){inside=false;break;}
            z=pow(z,2)+c;
        }  
        *(map+ind)=inside;

    }   

}


template<typename T> 
void init(complex<T> ** &arr,int N,int M){
    complex<T> i(0.0, 1.0);
    T h_n=2.0/(N);
    T h_m=3.0/(M);
    for (int x = 0; x <N; x++){
        for (int y = 0; y < M; y++){
            arr[y][x] =T(-2.0+x*h_m)+T(1.0-y*h_n)*i ;
        }
    }

}

template<typename T> 
void show(T ** arr,int N,int M){
    for (size_t i = 0; i <N; i++){
        std::cout<<std::endl;
        for (size_t j = 0; j < M; j++){
            std::cout<<arr[i][j]<<"\t";
        }
    }
    cout<<endl;
}

void write(bool ** arr,string filename,int N,int M){
    ofstream out;
    string name="../TextFiles/"+filename+".txt";
	out.open(name);
	if (out.is_open()) {
        out << setprecision(20) << N << " " << setprecision(20) << M << endl;
        for (int k = 0; k <N; k++){
            for (int j = 0; j < M; j++){
                if(arr[k][j]){
                    out << setprecision(20) << k << " " << setprecision(20) << j << endl;
                }
            }
        }
	}
	else {cout<<"Error while writing"<<endl;}
	out.close();
}

int main(){
    int N=1024*4;     
    int M=1024*4;
    int max_iter=1000;
    int nxy=N*M;
    int nBytes=nxy*sizeof(complex<float>);
    double iStart, iElaps;

    complex<float> **arr = new complex<float>* [N];
    for (int i = 0; i < N; i++){arr[i] = new complex<float> [M];}

    bool ** map_cpu=new bool*[N];
    for (int i = 0; i < N; i++){map_cpu[i] = new bool [M];}

    bool * map_gpu2=new bool[M*N];
    init(arr,N,M);
    
    // timing cpu
    iStart = cpuSecond();
    maldelbrot_cpu(arr,map_cpu,N,M,max_iter);
    iElaps = cpuSecond() - iStart;
    write(map_cpu,"cpu",N,M);
    cout<<"Time elapsed on cpu: "<<iElaps<<endl;
    
    // gpu 
    complex<float> * arr_1D=arr2Dto1D(arr,N,M);
    complex<float> * arr_gpu;
    bool*map_gpu;
    cudaMalloc(&arr_gpu,nBytes);
    cudaMalloc(&map_gpu,N*M*sizeof(bool));

    cudaMemcpy(arr_gpu,arr_1D,nBytes,cudaMemcpyHostToDevice);
    dim3 block(1024);
    dim3 grid((N + block.x - 1) / block.x,N);

    // a few iterations to start kernel
    maldelbrot_gpu<<<grid,block>>>(arr_gpu,map_gpu,N,M,5);


    // timing gpu
    iStart = cpuSecond();
    maldelbrot_gpu<<<grid,block>>>(arr_gpu,map_gpu,N,M,max_iter);
    cudaDeviceSynchronize();    
    iElaps = cpuSecond() - iStart;

    cudaMemcpy(map_gpu2, map_gpu, N*M*sizeof(bool), cudaMemcpyDeviceToHost);
    printf("maldelbrot_gpu<<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    map_cpu=arr1Dto2D(map_gpu2,N,M);
    write(map_cpu,"gpu",N,M);

    // free
    
    for (size_t i = 0; i < M; i++){delete[] arr[i];delete[] map_cpu[i];}
    delete[] arr;delete[] map_cpu;delete[] map_gpu2;
    cudaFree(arr_gpu);
    cudaFree(map_gpu);

    return 0;
}