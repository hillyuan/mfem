#ifndef __GPU_HELPER_HPP__
#define __GPU_HELPER_HPP__

#include "mfem.hpp"
#include "../linalg/kernels/vector.hpp"
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

  return mfem::kVectorDot(x_vec.Size(), x, y);
}

//norm is given squared here.
double l2Norm(Vector &a_vec, Vector &b_vec)
{

  Vector norm_vec(1);
  int len = a_vec.Size();

  double *a = a_vec.GetData();
  double *b = b_vec.GetData();
  double *norm = norm_vec.GetData();

  GET_ADRS(a);
  GET_ADRS(b);
  GET_ADRS(norm);

  //Worst l2 norm ever..
  my_forall(0, 1, [=] __device__ (int i) {

      double dot(0);
      for(int k=0; k<len; ++k) {
        dot += (d_a[k]-d_b[k])*(d_a[k]-d_b[k]);
      }

      d_norm[0] = dot;
    });

  norm_vec.Pull();

  return norm[0];
}


//x_vec = inv(A_Sp)*b_vec
void myCG(Vector &x_vec, SparseMatrix &A_Sp, const Vector &b_vec)
{

  x_vec = 0.0;
  static Vector rk_vec(b_vec);
  static Vector pk_vec(rk_vec);
  static Vector y_vec(b_vec.Size());
  static Vector xnew_vec(x_vec.Size());
  static Vector rnew_vec(x_vec.Size());

  double res = 10.0;
  while(res*res > 1e-9) {

    //compute step size
    double top = dotProduct(rk_vec, rk_vec);
    SpMatVec(y_vec, A_Sp, pk_vec);
    double bottom = dotProduct(pk_vec, y_vec);
    double alpha = top/bottom;

    //update approx solution
    VecScaleAdd(xnew_vec, x_vec, alpha, pk_vec);

    //update residual
    VecScaleAdd(rnew_vec, rk_vec, (-alpha), y_vec);

    //compute ||res||^2_l2
    res = dotProduct(rnew_vec, rnew_vec);

    //compute gradient correction factor
    double beta = dotProduct(rnew_vec, rnew_vec)/dotProduct(rk_vec, rk_vec);

    //new search direction
    VecScaleAdd(pk_vec, rnew_vec, beta, pk_vec);

    //update r-old and x-old
    //VecScaleAdd(rk_vec, rnew_vec, 0.0, rnew_vec);
    //VecScaleAdd(x_vec, xnew_vec, 0.0, rnew_vec);
    rk_vec = rnew_vec;
    x_vec  = xnew_vec;
  }



}



}//kernels namespace


#endif
