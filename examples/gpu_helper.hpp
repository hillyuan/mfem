#ifndef __GPU_HELPER_HPP__
#define __GPU_HELPER_HPP__

#include "mfem.hpp"
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace mfem;

template <typename LOOP_BODY>
__global__ void forall_kernel_gpu(int start, int length, LOOP_BODY body)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < length) {
    body(idx);
  }

}

template<typename LOOP_BODY>
void my_forall(int begin, int end, LOOP_BODY&& body)
{
  size_t blockSize = 32;
  size_t gridSize = (end - begin + blockSize - 1) / blockSize;
  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end - begin, body);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {    
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

namespace kernels
{
//Sparse Matrix multiplication...
void SpMatVec(Vector &y_vec, SparseMatrix &A_Sp, const Vector &b_vec)
{
  double *y = y_vec.GetData();
  double *b = b_vec.GetData();

  int *rowPtr = A_Sp.GetI(); //Row offsets
  int *colPtr = A_Sp.GetJ(); //Column index
  double *data = A_Sp.GetData(); //Get data

  //Vectors
  GET_ADRS(y);
  GET_ADRS(b);

  //Sparse Matrix
  GET_ADRS_T(rowPtr, int);
  GET_ADRS_T(colPtr, int);
  GET_ADRS(data);

  my_forall(0, A_Sp.Size(), [=] __device__ (int i) {

      double dot(0);
      for(int k = d_rowPtr[i]; k < d_rowPtr[i+1]; ++k) {
        dot += d_data[k]*d_b[d_colPtr[k]];
      }

      d_y[i] = dot;
    });
}

//z = x + y
void VecAdd(Vector &z_vec, const Vector &x_vec, const Vector &y_vec)
{
  double *x = x_vec.GetData();
  double *y = y_vec.GetData();
  double *z = z_vec.GetData();

  GET_ADRS(x);
  GET_ADRS(y);
  GET_ADRS(z);

  my_forall(0, y_vec.Size(), [=] __device__ (int i) {      
      d_z[i] = d_x[i] + d_y[i];
    });
}

//z = alpha*x + y
void VecScaleAdd(Vector &z_vec, const Vector &x_vec, const double alpha, const Vector &y_vec)
{
  double *x = x_vec.GetData();
  double *y = y_vec.GetData();
  double *z = z_vec.GetData();

  GET_ADRS(x);
  GET_ADRS(y);
  GET_ADRS(z);

  my_forall(0, y_vec.Size(), [=] __device__ (int i) {      
      d_z[i] = d_x[i] + alpha*d_y[i];
    });
}

double dotProduct(Vector &x_vec, Vector &y_vec)
{
  
  Vector res_vec(1); //store result here

  int len = x_vec.Size();
  double *x = x_vec.GetData();
  double *y = y_vec.GetData();
  double *res = res_vec.GetData();

  GET_ADRS(x);
  GET_ADRS(y);
  GET_ADRS(res);

  ///Worst dot product ever... 
  my_forall(0, 1, [=] __device__ (int i) {

      double dot(0);
      for(int k=0; k<len; ++k) {
        dot += d_x[k]*d_y[k];
      }
      
      d_res[0] = dot;
    });

  res_vec.Pull();

  return res[0];
}

double l2Norm(Vector &r_vec) 
{

  Vector norm_vec(1);  
  int len = r_vec.Size();
  
  double *r = r_vec.GetData();
  double *norm = norm_vec.GetData();
  
  GET_ADRS(r); 
  GET_ADRS(norm);

  return 5;
}


void myCG(Vector &x_vec, SparseMatrix &A_Sp, const Vector &b_vec) 
{
  
  printf("Running my CG \n");
  Vector r_vec(b_vec); //old residual 
  Vector p_vec(r_vec); //search direction
  
  Vector rnew_vec(r_vec); //new residual
  Vector xnew_vec(x_vec); //new solution

  Vector y_vec(x_vec); //temp - A*p_vec

  x_vec *= 0.0;
  x_vec.Push();

  double res = 10;
  while (res*res > 1e-9) {

  //Compute step length... 
  res = dotProduct(r_vec, r_vec);

  printf("residual = %.10f \n", res);
  
  //y_vec = A*p_vec
  SpMatVec(y_vec, A_Sp, p_vec);
  double bottom = dotProduct(p_vec,y_vec);

  double alpha = res / bottom;
  
  //update approximate solution
  VecScaleAdd(xnew_vec, x_vec, alpha, p_vec);
  
  //Update residual 
  VecScaleAdd(rnew_vec, r_vec, (-alpha), y_vec); 
  
  //Compute a gradient correction factor
  double beta = dotProduct(rnew_vec,rnew_vec)/dotProduct(r_vec, r_vec);
  
  //Set new search direction
  VecScaleAdd(p_vec, rnew_vec, beta, p_vec);

  //update residual 
  VecScaleAdd(r_vec, rnew_vec, 0.0, rnew_vec);

  }

  

  


}



}//kernels namespace


#endif
