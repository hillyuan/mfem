//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
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
void forall(int begin, int end, LOOP_BODY&& body)
{
  size_t blockSize = 32;
  size_t gridSize = (end - begin + blockSize - 1) / blockSize;
  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end - begin, body);
  cudaDeviceSynchronize();
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = true;
   bool cuda = true;
   bool occa = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-p", "--pa", "-no-p", "--no-pa",
                  "Enable Partial Assembly.");
   args.AddOption(&cuda, "-cu", "--cuda", "-no-cu", "--no-cuda", "Enable CUDA.");
   args.AddOption(&occa, "-oc", "--occa", "-no-oc", "--no-occa", "Enable OCCA.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 7. Set MFEM config parameters from the command line options
   config::usePA(pa);
   if (cuda) { config::useCuda(); }
   if (occa) { config::useOcca(); }
   config::enableGpu(0/*,occa,cuda*/);
   config::SwitchToGpu(); //Tuns on the GPU

   //RAJA - Sandbox
   int NN = 10;
   Vector r_vec_x(NN);
   Vector r_vec_y(NN);
   Vector r_vec_z(NN);

   for(int i=0; i<NN; ++i) {
     r_vec_x(i) = 1;
     r_vec_y(i) = 2;
     r_vec_z(i) = -55;
   }

   double alpha = 2.0;

   r_vec_x.Push();
   r_vec_y.Push();
   r_vec_z.Push();

   //get host pointer
   double * vec_x = r_vec_x.GetData();
   double * vec_y = r_vec_y.GetData();
   double * vec_z = r_vec_z.GetData();

   //Get device pointer
   GET_CONST_ADRS(vec_x);
   GET_CONST_ADRS(vec_y);
   GET_ADRS(vec_z);
   
   printf("address for z vector %p %p \n",vec_z, d_vec_z);

   const bool gpu = mfem::config::usingGpu();

   if(gpu) {
     printf("ON GPU \n");
   }else {
     printf("NOT ON GPU \n");
   }

#if 0
   //MFEM style for all
   MFEM_FORALL(i, NN,
               d_vec_z[i] = d_vec_x[i]*alpha + d_vec_y[i];
               );
#else
   //RAJA-like methods
   forall(0, NN, [=] __device__ (int i) {

       d_vec_z[i] = d_vec_x[i]*alpha + d_vec_y[i];
       double val = d_vec_z[i];

       printf("d_vec_z[%d] = %f \n",i, val);

     });
#endif
   cudaDeviceSynchronize();

   r_vec_x.Pull();
   r_vec_y.Pull();
   r_vec_z.Pull();


   for(int i=0; i<NN; ++i) {
     printf("entry %d : 2 * %f +  %f  = %f \n"
            ,i
            ,r_vec_x(i)
            ,r_vec_y(i)
            ,r_vec_z(i)
            );
   }

   std::cout<<"Successful run "<<std::endl;

   return 0;
}
