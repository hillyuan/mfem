// Copyright (c) 2016, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Author: Stefano Zampini <stefano.zampini@gmail.com>

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC

#include "linalg.hpp"
#include "../fem/fem.hpp"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

// Error handling
// Prints PETSc's stacktrace and then calls MFEM_ABORT
// We cannot use PETSc's CHKERRQ since it returns a PetscErrorCode
// TODO this can be improved
// TODO debug/release
#define PCHKERRQ(obj,err)                                                  \
  if ((err)) {                                                             \
    PetscError(PetscObjectComm((PetscObject)obj),__LINE__,_MFEM_FUNC_NAME, \
               __FILE__,err,PETSC_ERROR_REPEAT,NULL);                      \
    MFEM_ABORT("Error in PETSc. See stacktrace above.");                   \
  }

#define CCHKERRQ(comm,err)                               \
  if ((err)) {                                           \
    PetscError(comm,__LINE__,_MFEM_FUNC_NAME,            \
               __FILE__,err,PETSC_ERROR_REPEAT,NULL);    \
    MFEM_ABORT("Error in PETSc. See stacktrace above."); \
  }

// prototype for auxiliary functions
static PetscErrorCode mat_shell_apply(Mat,Vec,Vec);
static PetscErrorCode mat_shell_apply_transpose(Mat,Vec,Vec);
static PetscErrorCode mat_shell_destroy(Mat);
static PetscErrorCode pc_shell_apply(PC,Vec,Vec);
static PetscErrorCode pc_shell_apply_transpose(PC,Vec,Vec);
static PetscErrorCode pc_shell_setup(PC);
static PetscErrorCode pc_shell_destroy(PC);
static PetscErrorCode array_container_destroy(void*);
static PetscErrorCode Convert_Array_IS(MPI_Comm,mfem::Array<int>*,PetscInt,IS*);
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix*,Mat*);
static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix*,Mat*);

// structs used by PETSc code
typedef struct
{
   mfem::Operator *op;
} mat_shell_ctx;

typedef struct
{
   mfem::Solver *op;
} solver_shell_ctx;

// use global scope ierr to check PETSc errors inside mfem calls
PetscErrorCode ierr;

using namespace std;

namespace mfem
{

// PetscParVector methods

void PetscParVector::_SetDataAndSize_()
{
   const PetscScalar *array;
   PetscInt           n;

   ierr = VecGetArrayRead(x,&array); PCHKERRQ(x,ierr);
   ierr = VecGetLocalSize(x,&n); PCHKERRQ(x,ierr);
   SetDataAndSize((PetscScalar*)array,n);
   ierr = VecRestoreArrayRead(x,&array); PCHKERRQ(x,ierr);
}

PetscInt PetscParVector::GlobalSize()
{
   PetscInt N;
   ierr = VecGetSize(x,&N); PCHKERRQ(x,ierr);
   return N;
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *col) : Vector()
{
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);
   if (col)
   {
      PetscMPIInt myid;
      MPI_Comm_rank(comm, &myid);
      ierr = VecSetSizes(x,col[myid+1]-col[myid],PETSC_DECIDE); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecSetSizes(x,PETSC_DECIDE,glob_size); PCHKERRQ(x,ierr);
   }
   ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

PetscParVector::~PetscParVector()
{
   MPI_Comm comm = PetscObjectComm((PetscObject)x);
   ierr = VecDestroy(&x); CCHKERRQ(comm,ierr);
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscScalar *_data, PetscInt *col) : Vector()
{
   if (col)
   {
      PetscMPIInt myid;
      MPI_Comm_rank(comm, &myid);
      ierr = VecCreateMPIWithArray(comm,1,col[myid+1]-col[myid],PETSC_DECIDE,_data,
                                   &x); CCHKERRQ(comm,ierr)
   }
   else
   {
      ierr = VecCreateMPIWithArray(comm,1,PETSC_DECIDE,glob_size,_data,&x);
      CCHKERRQ(comm,ierr)
   }
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(const PetscParVector &y) : Vector()
{
   ierr = VecDuplicate(y.x,&x); PCHKERRQ(x,ierr);
}

PetscParVector::PetscParVector(const PetscParMatrix &A,
                               int transpose) : Vector()
{
   MPI_Comm comm = PETSC_COMM_WORLD;
   if (!transpose)
   {
      ierr = MatCreateVecs(const_cast<PetscParMatrix&>(A),&x,NULL);
      CCHKERRQ(comm,ierr);
   }
   else
   {
      ierr = MatCreateVecs(const_cast<PetscParMatrix&>(A),NULL,&x);
      CCHKERRQ(comm,ierr);
   }
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(Vec y, bool ref) : Vector()
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)y); PCHKERRQ(y,ierr);
   }
   x = y;
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(ParFiniteElementSpace *pfes) : Vector()
{

   HYPRE_Int* offsets = pfes->GetTrueDofOffsets();
   MPI_Comm  comm = pfes->GetComm();
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);

   PetscMPIInt myid = 0;
   if (!HYPRE_AssumedPartitionCheck())
   {
      MPI_Comm_rank(comm,&myid);
   }
   ierr = VecSetSizes(x,offsets[myid+1]-offsets[myid],PETSC_DECIDE);
   PCHKERRQ(x,ierr);
   ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

Vector * PetscParVector::GlobalVector() const
{
   VecScatter   scctx;
   Vec          vout;
   PetscScalar *array;
   PetscInt     size;

   ierr = VecScatterCreateToAll(x,&scctx,&vout); PCHKERRQ(x,ierr);
   ierr = VecScatterBegin(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterEnd(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterDestroy(&scctx); PCHKERRQ(x,ierr);
   ierr = VecGetArray(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecGetLocalSize(vout,&size); PCHKERRQ(x,ierr);
   Array<PetscScalar> data(size);
   data.Assign(array);
   ierr = VecRestoreArray(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecDestroy(&vout); PCHKERRQ(x,ierr);
   Vector *v = new Vector(data, internal::to_int(size));
   v->MakeDataOwner();
   data.LoseData();
   return v;
}

PetscParVector& PetscParVector::operator=(PetscScalar d)
{
   ierr = VecSet(x,d); PCHKERRQ(x,ierr);
   return *this;
}

PetscParVector& PetscParVector::operator=(const PetscParVector &y)
{
   ierr = VecCopy(y.x,x); PCHKERRQ(x,ierr);
   return *this;
}

void PetscParVector::SetData(PetscScalar *_data)
{
   ierr = VecPlaceArray(x,_data); PCHKERRQ(x,ierr);
}

void PetscParVector::ResetData()
{
   ierr = VecResetArray(x); PCHKERRQ(x,ierr);
}

void PetscParVector::Randomize(PetscInt seed)
{
   PetscRandom rctx;

   ierr = PetscRandomCreate(PetscObjectComm((PetscObject)x),&rctx);
   PCHKERRQ(x,ierr);
   ierr = PetscRandomSetSeed(rctx,(unsigned long)seed); PCHKERRQ(x,ierr);
   ierr = PetscRandomSeed(rctx); PCHKERRQ(x,ierr);
   ierr = VecSetRandom(x,rctx); PCHKERRQ(x,ierr);
   ierr = PetscRandomDestroy(&rctx); PCHKERRQ(x,ierr);
}

void PetscParVector::Print(const char *fname) const
{
   if (fname)
   {
      PetscViewer view;

      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)x),fname,&view);
      PCHKERRQ(x,ierr);
      ierr = VecView(x,view); PCHKERRQ(x,ierr);
      ierr = PetscViewerDestroy(&view); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecView(x,NULL); PCHKERRQ(x,ierr);
   }
}

// PetscParMatrix methods

PetscInt PetscParMatrix::GetNumRows()
{
   PetscInt N;
   ierr = MatGetLocalSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::GetNumCols()
{
   PetscInt N;
   ierr = MatGetLocalSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::M()
{
   PetscInt N;
   ierr = MatGetSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::N()
{
   PetscInt N;
   ierr = MatGetSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::NNZ()
{
   MatInfo info;
   ierr = MatGetInfo(A,MAT_GLOBAL_SUM,&info); PCHKERRQ(A,ierr);
   return (PetscInt)info.nz_used;
}

void PetscParMatrix::Init()
{
   A = NULL;
   X = Y = NULL;
   height = width = 0;
}

PetscParMatrix::PetscParMatrix()
{
   Init();
}

PetscParMatrix::PetscParMatrix(const HypreParMatrix *hmat, bool wrap,
                               bool assembled)
{
   Init();
   height = hmat->Height();
   width  = hmat->Width();
   if (wrap)
   {
      MFEM_VERIFY(assembled,
                  "PetscParMatrix::PetscParMatrix(const HypreParMatrix*,bool,bool)" <<
                  "Cannot wrap in PETSc's unassembled format")
      MakeWrapper(hmat->GetComm(),hmat,&A);
   }
   else
   {
      if (assembled)
      {
         ierr = MatConvert_hypreParCSR_AIJ(const_cast<HypreParMatrix&>(*hmat),&A);
         CCHKERRQ(hmat->GetComm(),ierr);
      }
      else
      {
         ierr = MatConvert_hypreParCSR_IS(const_cast<HypreParMatrix&>(*hmat),&A);
         CCHKERRQ(hmat->GetComm(),ierr);
      }
   }
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, const Operator *op, bool wrap)
{
   Init();
   PetscParMatrix *pA = const_cast<PetscParMatrix *>
                        (dynamic_cast<const PetscParMatrix *>(op));
   height = op->Height();
   width  = op->Width();
   if (wrap && !pA) { MakeWrapper(comm,op,&A); }
   else { ConvertOperator(comm,*op,&A); }
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *row_starts, SparseMatrix *diag, bool assembled)
{
   Init();
   BlockDiagonalConstructor(comm,glob_size,glob_size,row_starts,row_starts,diag,
                            assembled,&A);
   // update base class
   height = GetNumRows();
   width  = GetNumCols();
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt global_num_rows,
                               PetscInt global_num_cols, PetscInt *row_starts,
                               PetscInt *col_starts, SparseMatrix *diag, bool assembled)
{
   Init();
   BlockDiagonalConstructor(comm,global_num_rows,global_num_cols,row_starts,
                            col_starts,diag,assembled,&A);
   // update base class
   height = GetNumRows();
   width  = GetNumCols();
}

void PetscParMatrix::BlockDiagonalConstructor(MPI_Comm comm,
                                              PetscInt global_num_rows,
                                              PetscInt global_num_cols, PetscInt *row_starts,
                                              PetscInt *col_starts, SparseMatrix *diag, bool assembled, Mat* Ad)
{
   Mat      A;
   PetscInt lrsize,lcsize,rstart,cstart;
   PetscMPIInt myid = 0,commsize;

   ierr = MPI_Comm_size(comm,&commsize); CCHKERRQ(comm,ierr);
   if (!HYPRE_AssumedPartitionCheck())
   {
      ierr = MPI_Comm_rank(comm,&myid); CCHKERRQ(comm,ierr);
   }
   lrsize = row_starts[myid+1]-row_starts[myid];
   rstart = row_starts[myid];
   lcsize = col_starts[myid+1]-col_starts[myid];
   cstart = col_starts[myid];

   if (!assembled)
   {
      IS is;
      ierr = ISCreateStride(comm,diag->Height(),rstart,1,&is); CCHKERRQ(comm,ierr);
      ISLocalToGlobalMapping rl2g,cl2g;
      ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); PCHKERRQ(is,ierr);
      ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
      if (row_starts != col_starts)
      {
         ierr = ISCreateStride(comm,diag->Width(),cstart,1,&is); CCHKERRQ(comm,ierr);
         ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); PCHKERRQ(is,ierr);
         ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = PetscObjectReference((PetscObject)rl2g); PCHKERRQ(rl2g,ierr);
         cl2g = rl2g;
      }

      // Create the PETSc object (MATIS format)
      ierr = MatCreate(comm,&A); CCHKERRQ(comm,ierr);
      ierr = MatSetSizes(A,lrsize,lcsize,PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(A,ierr);
      ierr = MatSetType(A,MATIS); PCHKERRQ(A,ierr);
      ierr = MatSetLocalToGlobalMapping(A,rl2g,cl2g); PCHKERRQ(A,ierr);
      ierr = ISLocalToGlobalMappingDestroy(&rl2g); PCHKERRQ(A,ierr)
      ierr = ISLocalToGlobalMappingDestroy(&cl2g); PCHKERRQ(A,ierr)

      // Copy SparseMatrix into PETSc SeqAIJ format
      Mat lA;
      ierr = MatISGetLocalMat(A,&lA); PCHKERRQ(A,ierr);
      if (sizeof(PetscInt) == sizeof(int))
      {
         ierr = MatSeqAIJSetPreallocationCSR(lA,diag->GetI(),diag->GetJ(),
                                             diag->GetData()); PCHKERRQ(lA,ierr);
      }
      else
      {
         MFEM_ABORT("64bit indices not yet supported");
      }
   }
   else
   {
      PetscScalar *da;
      PetscInt    *dii,*djj,*oii, m = diag->Height()+1,
                                  nnz = diag->NumNonZeroElems();

      diag->SortColumnIndices();
      // if we can take ownership of the SparseMatrix arrays, we can avoid this step
      ierr = PetscMalloc1(m,&dii); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscMalloc1(nnz,&djj); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscMalloc1(nnz,&da); CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (sizeof(PetscInt) == sizeof(int))
      {
         ierr = PetscMemcpy(dii,diag->GetI(),m*sizeof(PetscInt));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = PetscMemcpy(djj,diag->GetJ(),nnz*sizeof(PetscInt));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = PetscMemcpy(da,diag->GetData(),nnz*sizeof(PetscScalar));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
      }
      else
      {
         MFEM_ABORT("64bit indices not yet supported");
      }
      ierr = PetscCalloc1(m,&oii);
      CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (commsize > 1)
      {
         ierr = MatCreateMPIAIJWithSplitArrays(comm,lrsize,lcsize,PETSC_DECIDE,
                                               PETSC_DECIDE,
                                               dii,djj,da,oii,NULL,NULL,&A);
         CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = MatCreateSeqAIJWithArrays(comm,lrsize,lcsize,dii,djj,da,&A);
         CCHKERRQ(comm,ierr);
      }

      void *ptrs[4] = {dii,djj,da,oii};
      const char *names[4] = {"_mfem_csr_dii",
                              "_mfem_csr_djj",
                              "_mfem_csr_da",
                              "_mfem_csr_oii",
                             };
      for (PetscInt i=0; i<4; i++)
      {
         PetscContainer c;

         ierr = PetscContainerCreate(comm,&c); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetPointer(c,ptrs[i]); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetUserDestroy(c,array_container_destroy);
         CCHKERRQ(comm,ierr);
         ierr = PetscObjectCompose((PetscObject)A,names[i],(PetscObject)c);
         CCHKERRQ(comm,ierr);
         ierr = PetscContainerDestroy(&c); CCHKERRQ(comm,ierr);
      }
   }

   // Tell PETSc the matrix is ready to be used
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);

   *Ad = A;
}

// TODO ADD THIS CONSTRUCTOR
//PetscParMatrix::PetscParMatrix(MPI_Comm comm, int nrows, PetscInt glob_nrows,
//                  PetscInt glob_ncols, int *I, PetscInt *J,
//                  double *data, PetscInt *rows, PetscInt *cols)
//{
//}

// TODO This should take a reference but how?
void PetscParMatrix::MakeWrapper(MPI_Comm comm, const Operator* op, Mat *A)
{
   mat_shell_ctx *ctx = new mat_shell_ctx;
   ierr = MatCreate(comm,A); CCHKERRQ(comm,ierr);
   ierr = MatSetSizes(*A,op->Height(),op->Width(),
                      PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(A,ierr);
   ierr = MatSetType(*A,MATSHELL); PCHKERRQ(A,ierr);
   ierr = MatShellSetContext(*A,(void *)ctx); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT,
                               (void (*)())mat_shell_apply); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,
                               (void (*)())mat_shell_apply_transpose); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_DESTROY,
                               (void (*)())mat_shell_destroy); PCHKERRQ(A,ierr);
   ierr = MatSetUp(*A); PCHKERRQ(*A,ierr);
   ctx->op = const_cast<Operator *>(op);
}

void PetscParMatrix::ConvertOperator(MPI_Comm comm, const Operator &op, Mat* A)
{
   PetscParMatrix *pA = const_cast<PetscParMatrix *>
                        (dynamic_cast<const PetscParMatrix *>(&op));
   HypreParMatrix *pH = const_cast<HypreParMatrix *>
                        (dynamic_cast<const HypreParMatrix *>(&op));
   BlockOperator  *pB = const_cast<BlockOperator *>
                        (dynamic_cast<const BlockOperator *>(&op));
   if (pA)
   {
      ierr = PetscObjectReference((PetscObject)(pA->A)); PCHKERRQ(pA->A,ierr);
      *A = pA->A;
   }
   else if (pH)
   {
      ierr = MatConvert_hypreParCSR_AIJ(const_cast<HypreParMatrix&>(*pH),A);
      CCHKERRQ(pH->GetComm(),ierr);
   }
   else if (pB)
   {
      Mat      *mats;
      PetscInt i,j,nr,nc;

      nr = pB->NumRowBlocks();
      nc = pB->NumColBlocks();
      ierr = PetscCalloc1(nr*nc,&mats); CCHKERRQ(PETSC_COMM_SELF,ierr);
      for (i=0; i<nr; i++)
      {
         for (j=0; j<nc; j++)
         {
            if (!pB->IsZeroBlock(i,j))
            {
               ConvertOperator(comm,pB->GetBlock(i,j),&mats[i*nc+j]);
            }
         }
      }
      ierr = MatCreateNest(comm,nr,NULL,nc,NULL,mats,A); CCHKERRQ(comm,ierr);
      for (i=0; i<nr*nc; i++) { ierr = MatDestroy(&mats[i]); CCHKERRQ(comm,ierr); }
      ierr = PetscFree(mats); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
   else
   {
      MFEM_ABORT("PetscParMatrix::ConvertOperator : don't know how to convert");
   }
}


void PetscParMatrix::Destroy()
{
   MPI_Comm comm = MPI_COMM_NULL;
   if (A != NULL)
   {
      ierr = PetscObjectGetComm((PetscObject)A,&comm); PCHKERRQ(A,ierr);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
   }
   if (X) { delete X; }
   if (Y) { delete Y; }
}

PetscParMatrix::PetscParMatrix(Mat a, bool ref)
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)a); PCHKERRQ(a,ierr);
   }
   Init();
   A = a;
   height = GetNumRows();
   width = GetNumCols();
}

// Computes y = alpha * A  * x + beta * y
//       or y = alpha * A^T* x + beta * y
static void MatMultKernel(Mat A,PetscScalar a,Vec X,PetscScalar b,Vec Y,
                          bool transpose)
{
   PetscErrorCode (*f)(Mat,Vec,Vec);
   PetscErrorCode (*fadd)(Mat,Vec,Vec,Vec);
   if (transpose)
   {
      f = MatMultTranspose;
      fadd = MatMultTransposeAdd;
   }
   else
   {
      f = MatMult;
      fadd = MatMultAdd;
   }
   if (a != 0.)
   {
      if (b == 1.)
      {
         ierr = VecScale(X,a); PCHKERRQ(A,ierr);
         ierr = (*fadd)(A,X,Y,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(X,1./a); PCHKERRQ(A,ierr);
      }
      else if (b != 0.)
      {
         ierr = VecScale(X,a); PCHKERRQ(A,ierr);
         ierr = VecScale(Y,b); PCHKERRQ(A,ierr);
         ierr = (*fadd)(A,X,Y,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(X,1./a); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = (*f)(A,X,Y); PCHKERRQ(A,ierr);
         if (a != 1.)
         {
            ierr = VecScale(Y,a); PCHKERRQ(A,ierr);
         }
      }
   }
   else
   {
      if (b == 1.)
      {
         // do nothing
      }
      else if (b != 0.)
      {
         ierr = VecScale(Y,b); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = VecSet(Y,0.); PCHKERRQ(A,ierr);
      }
   }
}

void PetscParMatrix::MakeRef(const PetscParMatrix &master)
{
   ierr = PetscObjectReference((PetscObject)master.A); PCHKERRQ(master.A,ierr);
   Destroy();
   Init();
   A = master.A;
   height = master.height;
   width = master.width;
}

PetscParVector * PetscParMatrix::GetX() const
{
   if (!X)
   {
      MFEM_VERIFY(A,"Mat not present");
      X = new PetscParVector(*this,false); PCHKERRQ(A,ierr);
   }
   return X;
}

PetscParVector * PetscParMatrix::GetY() const
{
   if (!Y)
   {
      MFEM_VERIFY(A,"Mat not present");
      Y = new PetscParVector(*this,true); PCHKERRQ(A,ierr);
   }
   return Y;
}

PetscParMatrix * PetscParMatrix::Transpose(bool action)
{
   Mat B;
   if (action)
   {
      ierr = MatCreateTranspose(A,&B); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B); PCHKERRQ(A,ierr);
   }
   return new PetscParMatrix(B,false);
}

void PetscParMatrix::operator*=(double s)
{
   ierr = MatScale(A,s); PCHKERRQ(A,ierr);
}

void PetscParMatrix::Mult(double a, const Vector &x, double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   XX->SetData(x.GetData());
   YY->SetData(y.GetData());
   MatMultKernel(A,a,XX->x,b,YY->x,false);
   XX->ResetData();
   YY->ResetData();
}

void PetscParMatrix::MultTranspose(double a, const Vector &x, double b,
                                   Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(y.Size() == Width(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Width());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   YY->SetData(x.GetData());
   XX->SetData(y.GetData());
   MatMultKernel(A,a,YY->x,b,XX->x,true);
   XX->ResetData();
   YY->ResetData();
}

PetscParMatrix * RAP(PetscParMatrix *Rt, PetscParMatrix *A, PetscParMatrix *P)
{
   Mat       pA = *A,pP = *P,pRt = *Rt;
   Mat       B;
   PetscBool Aismatis,Aisaij,Pismatis,Pisaij,Rtismatis,Rtisaij;

   MFEM_VERIFY(A->Width() == P->Height(),
               "Petsc RAP: Number of local cols of A " << A->Width() <<
               " differs from number of local rows of P " << P->Height());
   MFEM_VERIFY(A->Height() == Rt->Height(),
               "Petsc RAP: Number of local rows of A " << A->Height() <<
               " differs from number of local rows of Rt " << Rt->Height());
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATIS,&Aismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATAIJ,&Aisaij);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pP,MATIS,&Pismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pP,MATAIJ,&Pisaij);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pRt,MATIS,&Rtismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pRt,MATAIJ,&Rtisaij);
   PCHKERRQ(pA,ierr);
   if (Aismatis &&
       Pismatis &&
       Rtismatis) // handle special case (this code will eventually go into PETSc)
   {
      Mat                    lA,lP,lB,lRt;
      ISLocalToGlobalMapping cl2gP,cl2gRt;
      PetscInt               rlsize,clsize;

      ierr = MatGetLocalToGlobalMapping(pP,NULL,&cl2gP); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalToGlobalMapping(pRt,NULL,&cl2gRt); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalSize(pP,NULL,&clsize); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalSize(pRt,NULL,&rlsize); PCHKERRQ(pA,ierr);
      ierr = MatCreate(A->GetComm(),&B); PCHKERRQ(pA,ierr);
      ierr = MatSetSizes(B,rlsize,clsize,PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(B,ierr);
      ierr = MatSetType(B,MATIS); PCHKERRQ(B,ierr);
      ierr = MatSetLocalToGlobalMapping(B,cl2gRt,cl2gP); PCHKERRQ(B,ierr);
      ierr = MatISGetLocalMat(pA,&lA); PCHKERRQ(pA,ierr);
      ierr = MatISGetLocalMat(pP,&lP); PCHKERRQ(pA,ierr);
      ierr = MatISGetLocalMat(pRt,&lRt); PCHKERRQ(pA,ierr);
      if (lRt == lP)
      {
         ierr = MatPtAP(lA,lP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lB); PCHKERRQ(lA,ierr);
      }
      else
      {
         Mat lR;
         ierr = MatTranspose(lRt,MAT_INITIAL_MATRIX,&lR); PCHKERRQ(lRt,ierr);
         ierr = MatMatMatMult(lR,lA,lP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lB);
         PCHKERRQ(lRt,ierr);
         ierr = MatDestroy(&lR); PCHKERRQ(lRt,ierr);
      }
      ierr = MatISSetLocalMat(B,lB); PCHKERRQ(lB,ierr);
      ierr = MatDestroy(&lB); PCHKERRQ(lA,ierr);
      ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
      ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
   }
   else // it raises an error if the PtAP is not supported in PETSc
   {
      if (pP == pRt)
      {
         ierr = MatPtAP(pA,pP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B); PCHKERRQ(pA,ierr);
      }
      else
      {
         Mat pR;
         ierr = MatTranspose(pRt,MAT_INITIAL_MATRIX,&pR); PCHKERRQ(Rt,ierr);
         ierr = MatMatMatMult(pR,pA,pP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);
         PCHKERRQ(pRt,ierr);
         ierr = MatDestroy(&pR); PCHKERRQ(pRt,ierr);
      }
   }
   return new PetscParMatrix(B);
}

PetscParMatrix * RAP(PetscParMatrix *A, PetscParMatrix *P)
{
   PetscParMatrix *out = RAP(P,A,P);
   return out;
}

PetscParMatrix* PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols)
{
   Mat             Ae;
   const int       *data;
   PetscInt        M,N,i,n,*idxs,rst;

   ierr = MatGetSize(A,&M,&N); PCHKERRQ(A,ierr);
   MFEM_VERIFY(M == N,"Rectangular case unsupported");
   ierr = MatGetOwnershipRange(A,&rst,NULL); PCHKERRQ(A,ierr);
   ierr = MatDuplicate(A,MAT_COPY_VALUES,&Ae); PCHKERRQ(A,ierr);
   ierr = MatSetOption(A,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE); PCHKERRQ(A,ierr);
   // rows need to be in global numbering
   n = rows_cols.Size();
   data = rows_cols.GetData();
   ierr = PetscMalloc1(n,&idxs); PCHKERRQ(A,ierr);
   for (i=0; i<n; i++) { idxs[i] = data[i] + rst; }
   ierr = MatZeroRowsColumns(A,n,idxs,1.,NULL,NULL); PCHKERRQ(A,ierr);
   ierr = PetscFree(idxs); PCHKERRQ(A,ierr);
   ierr = MatAXPY(Ae,-1.,A,SAME_NONZERO_PATTERN); PCHKERRQ(A,ierr);
   return new PetscParMatrix(Ae);
}

void PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const HypreParVector &X,
                                       HypreParVector &B)
{
   MFEM_ABORT("To be implemented");
}

void EliminateBC(PetscParMatrix &A, PetscParMatrix &Ae,
                 const Array<int> &ess_dof_list,
                 const Vector &X, Vector &B)
{
   const PetscScalar *array;
   Mat pA = const_cast<PetscParMatrix&>(A);

   // B -= Ae*X
   Ae.Mult(-1.0, X, 1.0, B);

   Vec diag = const_cast<PetscParVector&>((*A.GetX()));
   ierr = MatGetDiagonal(pA,diag); PCHKERRQ(pA,ierr);
   ierr = VecGetArrayRead(diag,&array); PCHKERRQ(diag,ierr);
   for (int i = 0; i < ess_dof_list.Size(); i++)
   {
      int r = ess_dof_list[i];
      B(r) = array[r] * X(r);
   }
   ierr = VecRestoreArrayRead(diag,&array); PCHKERRQ(diag,ierr);
}

// PetscSolver methods

void PetscSolver::Init()
{
   obj = NULL;
   B = X = NULL;
   clcustom = false;
   _prset = true;
   cid = -1;
}

void PetscSolver::SetPrefix(std::string prefix)
{
   _prefix = prefix;
   _prset  = false;
}

PetscSolver::PetscSolver()
{
   Init();
}

PetscSolver::~PetscSolver()
{
   if (B) { delete B; }
   if (X) { delete X; }
}

void PetscSolver::Customize() const
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      if (!_prset && _prefix.c_str())
      {
         ierr = KSPSetOptionsPrefix(ksp,_prefix.c_str());
         PCHKERRQ(ksp,ierr);
         _prset = true;
      }
      if (!clcustom)
      {
         ierr = KSPSetFromOptions(ksp);
         PCHKERRQ(ksp,ierr);
         clcustom = true;
      }
   }
   else if (cid == PC_CLASSID)
   {
      PC pc = (PC)obj;
      if (!_prset && _prefix.c_str())
      {
         ierr = PCSetOptionsPrefix(pc,_prefix.c_str());
         PCHKERRQ(pc,ierr);
         _prset = true;
      }
      if (!clcustom)
      {
         ierr = PCSetFromOptions(pc);
         PCHKERRQ(pc,ierr);
         clcustom = true;
      }
   }
   else
   {
      MFEM_ABORT("Customize() to be implemented!");
   }
}

void PetscSolver::SetTol(double tol)
{
   SetRelTol(tol);
}

void PetscSolver::SetRelTol(double tol)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
   }
   else
   {
      MFEM_ABORT("SetRelTol() to be implemented!");
   }
   PCHKERRQ(obj,ierr);
}

void PetscSolver::SetAbsTol(double tol)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,tol,PETSC_DEFAULT,PETSC_DEFAULT);
   }
   else
   {
      MFEM_ABORT("SetAbsTol() to be implemented!");
   }
   PCHKERRQ(obj,ierr);
}

void PetscSolver::SetMaxIter(int max_iter)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,max_iter);
   }
   else
   {
      MFEM_ABORT("SetMaxIter() to be implemented!");
   }
   PCHKERRQ(obj,ierr);
}

void PetscSolver::SetPrintLevel(int plev)
{
   PetscViewerAndFormat *vf;
   PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm(obj));

   ierr = PetscViewerAndFormatCreate(viewer,PETSC_VIEWER_DEFAULT,&vf); PCHKERRQ(obj,ierr);
   if (cid == KSP_CLASSID)
   {
      // there are many other options, see KSPSetFromOptions at src/ksp/ksp/interface/itcl.c
      KSP ksp = (KSP)obj;
      ierr = KSPMonitorCancel(ksp); PCHKERRQ(ksp,ierr);
      if (plev == 1)
      {
         ierr = KSPMonitorSet(ksp,(PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorDefault,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy); PCHKERRQ(ksp,ierr);
      }
      else if (plev > 1)
      {
         ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE); PCHKERRQ(ksp,ierr);
         ierr = KSPMonitorSet(ksp,(PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorSingularValue,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy); PCHKERRQ(ksp,ierr);
         if (plev > 2)
         {
            ierr = PetscViewerAndFormatCreate(viewer,PETSC_VIEWER_DEFAULT,&vf); PCHKERRQ(viewer,ierr);
            ierr = KSPMonitorSet(ksp,(PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorTrueResidualNorm,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy); PCHKERRQ(ksp,ierr);
         }
      }
   } else {
      MFEM_ABORT("SetPrintLevel() to be implemented!");
   }
}

void PetscSolver::Mult(const PetscParVector &b, PetscParVector &x) const
{
   Customize();

   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetInitialGuessNonzero(ksp,PetscBool(iterative_mode));
      PCHKERRQ(ksp,ierr);
      ierr = KSPSolve(ksp,b.x,x.x); PCHKERRQ(obj,ierr);
   }
   else if (cid == PC_CLASSID)
   {
      PC pc = (PC)obj;
      ierr = PCApply(pc,b.x,x.x); PCHKERRQ(obj,ierr);
   }
   else
   {
      MFEM_ABORT("To be implemented!");
   }
}

void PetscSolver::Mult(const Vector &b, Vector &x) const
{
   // constructs PetscParVectors if not present
   if (!B || !X)
   {
      Mat pA;
      if (cid == KSP_CLASSID)
      {
         KSP ksp = (KSP)obj;
         ierr = KSPGetOperators(ksp,&pA,NULL); PCHKERRQ(obj,ierr);
      }
      else if (cid == PC_CLASSID)
      {
         PC pc = (PC)obj;
         ierr = PCGetOperators(pc,NULL,&pA); PCHKERRQ(obj,ierr);
      }
      else
      {
         MFEM_ABORT("To be implemented!");
      }
      if (!B)
      {
         PetscParMatrix A = PetscParMatrix(pA,true);
         B = new PetscParVector(A,true);
      }
      if (!X)
      {
         PetscParMatrix A = PetscParMatrix(pA,true);
         X = new PetscParVector(A,false);
      }
   }

   // Apply Mult
   B->SetData(b.GetData());
   X->SetData(x.GetData());
   Mult(*B, *X);
   B->ResetData();
   X->ResetData();
}

int PetscSolver::GetConverged()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      KSPConvergedReason reason;
      ierr = KSPGetConvergedReason(ksp,&reason);
      PCHKERRQ(ksp,ierr);
      return (int)reason;
   }
   else
   {
      MFEM_WARNING("GetConverged to be implemented!");
      return -1;
   }
}

int PetscSolver::GetNumIterations()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      PetscInt its;
      ierr = KSPGetIterationNumber(ksp,&its);
      PCHKERRQ(ksp,ierr);
      return its;
   }
   else
   {
      MFEM_WARNING("GetNumIterations to be implemented!");
      return -1;
   }
}

double PetscSolver::GetFinalNorm()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      PetscReal norm;
      ierr = KSPGetResidualNorm(ksp,&norm);
      PCHKERRQ(ksp,ierr);
      return norm;
   }
   else
   {
      MFEM_WARNING("GetConverged to be implemented!");
      return PETSC_MAX_REAL;
   }
}

// PetscLinearSolver methods

void PetscLinearSolver::Init()
{
   wrap = false;
}

PetscLinearSolver::PetscLinearSolver(MPI_Comm comm,
                                     std::string prefix) : PetscSolver()
{
   Init();
   SetPrefix(prefix);

   KSP ksp;
   ierr = KSPCreate(comm,&ksp); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
}

PetscLinearSolver::PetscLinearSolver(PetscParMatrix &_A,
                                     std::string prefix) : PetscSolver()
{
   Init();
   SetPrefix(prefix);

   KSP ksp;
   ierr = KSPCreate(_A.GetComm(),&ksp); CCHKERRQ(_A.GetComm(),ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   SetOperator(_A);
}

PetscLinearSolver::PetscLinearSolver(HypreParMatrix &_A,bool wrapin,
                                     std::string prefix) : PetscSolver()
{
   Init();
   SetPrefix(prefix);
   wrap = wrapin;

   KSP ksp;
   ierr = KSPCreate(_A.GetComm(),&ksp); CCHKERRQ(_A.GetComm(),ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   SetOperator(_A);
}

void PetscLinearSolver::SetOperator(const Operator &op)
{
   HypreParMatrix *hA = const_cast<HypreParMatrix *>
                        (dynamic_cast<const HypreParMatrix *>(&op));
   PetscParMatrix *pA = const_cast<PetscParMatrix *>
                        (dynamic_cast<const PetscParMatrix *>(&op));
   Operator       *oA = const_cast<Operator *>
                        (dynamic_cast<const Operator *>(&op));
   // update base classes: Operator, Solver, PetscLinearSolver
   bool delete_pA = false;
   if (!pA)
   {
      if (hA)
      {
         // Create MATSHELL object or convert
         // into PETSc AIJ format depending on wrap
         pA = new PetscParMatrix(hA,wrap);
         delete_pA = true;

      }
      else if (oA) // fallback to general operator
      {
         // Create MATSHELL or MATNEST object
         pA = new PetscParMatrix(PetscObjectComm(obj),oA);
         delete_pA = true;
      }
   }
   if (!pA)
   {
      MFEM_ABORT("PetscLinearSolver::SetOperator : Unsupported operation!");
   }

   KSP ksp = (KSP)obj;
   Mat A = pA->A;
   PetscInt nheight,nwidth;
   ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
   if (height || width)
   {
      if (nheight != height || nwidth != width)
      {
         // reinit without destroying the KSP
         // communicator remains the same
         ierr = KSPReset(ksp); PCHKERRQ(ksp,ierr);
         if (X) { delete X; }
         if (B) { delete B; }
         B = X = NULL;
         wrap = false;
      }
   }
   ierr = KSPSetOperators(ksp,A,A); PCHKERRQ(ksp,ierr);
   if (delete_pA) { delete pA; }
   height = nheight;
   width  = nwidth;
}

void PetscLinearSolver::SetPreconditioner(Solver &precond)
{
   KSP ksp = (KSP)obj;
   PetscPreconditioner *ppc = const_cast<PetscPreconditioner *>
                              (dynamic_cast<const PetscPreconditioner *>(&precond));
   if (ppc)
   {
      ierr = KSPSetPC(ksp,*ppc); PCHKERRQ(ksp,ierr);
   }
   else // wrap the Solver action
   {
      PC pc;
      ierr = KSPGetPC(ksp,&pc); PCHKERRQ(ksp,ierr);
      ierr = PCSetType(pc,PCSHELL); PCHKERRQ(pc,ierr);
      solver_shell_ctx *ctx = new solver_shell_ctx;
      ctx->op = &precond;
      ierr = PCShellSetContext(pc,(void *)ctx); PCHKERRQ(pc,ierr);
      ierr = PCShellSetApply(pc,pc_shell_apply); PCHKERRQ(pc,ierr);
      ierr = PCShellSetApplyTranspose(pc,pc_shell_apply_transpose); PCHKERRQ(pc,ierr);
      ierr = PCShellSetSetUp(pc,pc_shell_setup); PCHKERRQ(pc,ierr);
      ierr = PCShellSetDestroy(pc,pc_shell_destroy); PCHKERRQ(pc,ierr);
   }
}

PetscLinearSolver::~PetscLinearSolver()
{
   MPI_Comm comm;
   KSP ksp = (KSP)obj;
   ierr = PetscObjectGetComm((PetscObject)ksp,&comm); PCHKERRQ(ksp,ierr);
   ierr = KSPDestroy(&ksp); CCHKERRQ(comm,ierr);
}

// PetscPCGSolver methods

PetscPCGSolver::PetscPCGSolver(MPI_Comm comm,
                               std::string prefix) : PetscLinearSolver(comm,prefix)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

PetscPCGSolver::PetscPCGSolver(PetscParMatrix& _A,
                               std::string prefix) : PetscLinearSolver(_A,prefix)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

PetscPCGSolver::PetscPCGSolver(HypreParMatrix& _A,
                               bool wrap, std::string prefix) : PetscLinearSolver(_A,wrap,prefix)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

// PetscPreconditioner methods

void PetscPreconditioner::Init()
{
   // do nothing
}

PetscPreconditioner::PetscPreconditioner(MPI_Comm comm,
                                         std::string prefix) : PetscSolver()
{
   Init();
   SetPrefix(prefix);

   PC pc;
   ierr = PCCreate(comm,&pc); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
}

PetscPreconditioner::PetscPreconditioner(PetscParMatrix &_A,
                                         std::string prefix) : PetscSolver()
{
   Init();
   SetPrefix(prefix);

   PC pc;
   ierr = PCCreate(_A.GetComm(),&pc); CCHKERRQ(_A.GetComm(),ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   SetOperator(_A);
}

PetscPreconditioner::PetscPreconditioner(MPI_Comm comm, Operator &op,
                                         std::string prefix)
{
   Init();
   SetPrefix(prefix);

   PC pc;
   ierr = PCCreate(comm,&pc); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   SetOperator(op);
}

void PetscPreconditioner::SetOperator(const Operator &op)
{
   bool delete_pA = false;
   PetscParMatrix *pA = const_cast<PetscParMatrix *>
                        (dynamic_cast<const PetscParMatrix *>(&op));
   BlockOperator  *pB = const_cast<BlockOperator *>
                        (dynamic_cast<const BlockOperator *>(&op));
   if (!pA && pB)
   {
      pA = new PetscParMatrix(PetscObjectComm(obj),pB,false);
      delete_pA = true;
   }
   if (!pA) { MFEM_ABORT("PetscPreconditioner::SetOperator Operator should be a PetscParMatrix or a BlockOperator"); }

   PC pc = (PC)obj;
   ierr = PCSetOperators(pc,pA->A,pA->A); PCHKERRQ(obj,ierr);
   if (delete_pA) { delete pA; };
}

PetscPreconditioner::~PetscPreconditioner()
{
   MPI_Comm comm;
   PC pc = (PC)obj;
   ierr = PetscObjectGetComm((PetscObject)pc,&comm); PCHKERRQ(pc,ierr);
   ierr = PCDestroy(&pc); CCHKERRQ(comm,ierr);
}

// PetscBDDCSolver methods

PetscBDDCSolver::PetscBDDCSolver(PetscParMatrix &A, PetscBDDCSolverParams opts,
                                 std::string prefix) : PetscPreconditioner(A,prefix)
{
   PC pc = (PC)obj;
   MPI_Comm comm = A.GetComm();
   // index sets for boundary dofs specification (Essential = dir, Natural = neu)
   IS dir = NULL, neu = NULL;
   PetscInt rst;

   // Check
   Mat pA;
   PetscBool ismatis;
   ierr = PCGetOperators(pc,NULL,&pA); PCHKERRQ(pc,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATIS,&ismatis);
   PCHKERRQ(pA,ierr);
   MFEM_VERIFY(ismatis,"PetscBDDCSolver needs the matrix in unassembled format");

   ierr = PCSetType(pc,PCBDDC); PCHKERRQ(obj,ierr);

   // pass information about index sets (essential dofs, fields, etc.)
#ifdef MFEM_DEBUG
   {
      // make sure ess/nat_tdof_list have been collectively set
      PetscBool lpr = PETSC_FALSE,pr;
      if (opts.ess_tdof_list) { lpr = PETSC_TRUE; }
      ierr = MPI_Allreduce(&lpr,&pr,1,MPIU_BOOL,MPI_LOR,comm);
      PCHKERRQ(pA,ierr);
      MFEM_VERIFY(lpr == pr,"ess_tdof_list should be collectively set");
      lpr = PETSC_FALSE;
      if (opts.nat_tdof_list) { lpr = PETSC_TRUE; }
      ierr = MPI_Allreduce(&lpr,&pr,1,MPIU_BOOL,MPI_LOR,comm);
      PCHKERRQ(pA,ierr);
      MFEM_VERIFY(lpr == pr,"nat_tdof_list should be collectively set");
   }
#endif
   // need the sets in global ordering
   ierr = MatGetOwnershipRange(pA,&rst,NULL); PCHKERRQ(pA,ierr);
   if (opts.ess_tdof_list)
   {
      ierr = Convert_Array_IS(comm,opts.ess_tdof_list,rst,&dir); CCHKERRQ(comm,ierr);
      ierr = PCBDDCSetDirichletBoundaries(pc,dir); PCHKERRQ(pc,ierr);
   }
   if (opts.nat_tdof_list)
   {
      ierr = Convert_Array_IS(comm,opts.nat_tdof_list,rst,&neu); CCHKERRQ(comm,ierr);
      ierr = PCBDDCSetNeumannBoundaries(pc,neu); PCHKERRQ(pc,ierr);
   }

   ParFiniteElementSpace *fespace = opts.fespace;
   int bs = 1;
   if (fespace)
   {
      const     FiniteElementCollection *fec = fespace->FEColl();
      bool      edgespace, rtspace;
      bool      needint = false;
      bool      tracespace, rt_tracespace, edge_tracespace;
      int       dim , p;
      PetscBool B_is_Trans = PETSC_FALSE;

      ParMesh *pmesh = (ParMesh *) fespace->GetMesh();
      dim = pmesh->Dimension();
      bs = fec->DofForGeometry(Geometry::POINT);
      bs = bs ? bs : 1;
      rtspace = dynamic_cast<const RT_FECollection*>(fec);
      edgespace = dynamic_cast<const ND_FECollection*>(fec);
      edge_tracespace = dynamic_cast<const ND_Trace_FECollection*>(fec);
      rt_tracespace = dynamic_cast<const RT_Trace_FECollection*>(fec);
      tracespace = edge_tracespace || rt_tracespace;

      p = 1;
      if (fespace->GetNE() > 0)
      {
         if (!tracespace)
         {
            p = fespace->GetOrder(0);
         }
         else
         {
            p = fespace->GetFaceOrder(0);
            if (dim == 2) { p++; }
         }
      }

      if (edgespace) // H(curl)
      {
         if (dim == 2)
         {
            needint = true;
            if (tracespace)
            {
               MFEM_WARNING("Tracespace case doesn't work for H(curl) and p=2, not using auxiliary quadrature");
               needint = false;
            }
         }
         else
         {
            FiniteElementCollection *vfec;
            if (tracespace)
            {
               vfec = new H1_Trace_FECollection(p,dim);
            }
            else
            {
               vfec = new H1_FECollection(p,dim);
            }
            ParFiniteElementSpace *vfespace = new ParFiniteElementSpace(pmesh,vfec);
            ParDiscreteLinearOperator *grad;
            grad = new ParDiscreteLinearOperator(vfespace,fespace);
            if (tracespace)
            {
               grad->AddTraceFaceInterpolator(new GradientInterpolator);
            }
            else
            {
               grad->AddDomainInterpolator(new GradientInterpolator);
            }
            grad->Assemble();
            grad->Finalize();
            HypreParMatrix *hG = grad->ParallelAssemble();
            PetscParMatrix *G = new PetscParMatrix(hG,false,true);
            delete hG;
            delete grad;

            PetscBool conforming = PETSC_TRUE;
            if (pmesh->Nonconforming()) { conforming = PETSC_FALSE; }
            ierr = PCBDDCSetDiscreteGradient(pc,*G,p,0,PETSC_TRUE,conforming);
            PCHKERRQ(pc,ierr);
            delete vfec;
            delete vfespace;
            delete G;
         }
      }
      else if (rtspace) // H(div)
      {
         needint = true;
         if (tracespace)
         {
            MFEM_WARNING("Tracespace case doesn't work for H(div), not using auxiliary quadrature");
            needint = false;
         }
      }
      else if (bs == dim) // Elasticity?
      {
         needint = true;
      }

      PetscParMatrix *B = NULL;
      if (needint)
      {
         // Generate bilinear form in unassembled format which is used to
         // compute the net-flux across subdomain boundaries
         // for H(div) and Elasticity, and the line integral \int u x n of
         // 2D H(curl) fields
         FiniteElementCollection *auxcoll;
         if (tracespace) { auxcoll = new RT_Trace_FECollection(p,dim); }
         else { auxcoll = new L2_FECollection(p,dim); };
         ParFiniteElementSpace *pspace = new ParFiniteElementSpace(pmesh,auxcoll);
         ParMixedBilinearForm *b = new ParMixedBilinearForm(fespace,pspace);

         b->SetUseNonoverlappingFormat();
         if (edgespace)
         {
            if (tracespace)
            {
               b->AddTraceFaceIntegrator(new VectorFECurlIntegrator);
            }
            else
            {
               b->AddDomainIntegrator(new VectorFECurlIntegrator);
            }
         }
         else
         {
            if (tracespace)
            {
               b->AddTraceFaceIntegrator(new VectorFEDivergenceIntegrator);
            }
            else
            {
               b->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            }
         }
         b->Assemble();
         b->Finalize();
         B = b->PetscParallelAssemble();

         // Support for this is in master.
         if (dir) // if essential dofs are present, we need to zero the columns
         {
            Mat pB = *B;
            ierr = MatTranspose(pB,MAT_REUSE_MATRIX,&pB); PCHKERRQ(pA,ierr);
            ierr = MatZeroRowsIS(pB,dir,0.,NULL,NULL); PCHKERRQ(pA,ierr);
            B_is_Trans = PETSC_TRUE;
         }
         delete b;
         delete pspace;
         delete auxcoll;
      }

      // this API call is still not in master.
      // You need to checkout next to use it
      if (B)
      {
         ierr = PCBDDCSetDivergenceMat(pc,*B,B_is_Trans,NULL); PCHKERRQ(pc,ierr);
      }
      if (bs)
      {
         ierr = MatSetBlockSize(pA,bs); PCHKERRQ(pc,ierr);
      }
      delete B;
   }
   ierr = ISDestroy(&dir); PCHKERRQ(pc,ierr);
   ierr = ISDestroy(&neu); PCHKERRQ(pc,ierr);
}

PetscFieldSplitSolver::PetscFieldSplitSolver(MPI_Comm comm, BlockOperator &op,
                                             std::string prefix) : PetscPreconditioner(comm,op,prefix)
{
   PC pc = (PC)obj;
   Mat pA;
   IS  *isrow;
   PetscInt nr,i;
   ierr = PCSetType(pc,PCFIELDSPLIT); PCHKERRQ(pc,ierr);
   ierr = PCGetOperators(pc,&pA,NULL); PCHKERRQ(pc,ierr);
   ierr = MatNestGetSize(pA,&nr,NULL); PCHKERRQ(pc,ierr);
   ierr = PetscCalloc1(nr,&isrow); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = MatNestGetISs(pA,isrow,NULL); PCHKERRQ(pc,ierr);
   for (i=0; i<nr; i++)
   {
      ierr = PCFieldSplitSetIS(pc,NULL,isrow[i]); PCHKERRQ(pc,ierr);
   }
   ierr = PetscFree(isrow); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

}  // namespace mfem

// auxiliary functions

#undef __FUNCT__
#define __FUNCT__ "mat_shell_apply"
static PetscErrorCode mat_shell_apply(Mat A, Vec x, Vec y)
{
   mat_shell_ctx     *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); PCHKERRQ(A,ierr);
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ctx->op->Mult(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "mat_shell_apply_transpose"
static PetscErrorCode mat_shell_apply_transpose(Mat A, Vec x, Vec y)
{
   mat_shell_ctx     *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); PCHKERRQ(A,ierr);
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ctx->op->MultTranspose(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "mat_shell_destroy"
static PetscErrorCode mat_shell_destroy(Mat A)
{
   mat_shell_ctx  *ctx;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); PCHKERRQ(A,ierr);
   delete ctx;
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pc_shell_apply"
static PetscErrorCode pc_shell_apply(PC pc, Vec x, Vec y)
{
   solver_shell_ctx  *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); PCHKERRQ(pc,ierr);
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ctx->op->Mult(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pc_shell_apply_transpose"
static PetscErrorCode pc_shell_apply_transpose(PC pc, Vec x, Vec y)
{
   solver_shell_ctx  *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); PCHKERRQ(pc,ierr);
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ctx->op->MultTranspose(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pc_shell_setup"
static PetscErrorCode pc_shell_setup(PC pc)
{
   PetscFunctionBeginUser;
   // TODO ask: is there a way to trigger the setup of ctx->op?
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pc_shell_destroy"
static PetscErrorCode pc_shell_destroy(PC pc)
{
   solver_shell_ctx *ctx;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); PCHKERRQ(pc,ierr);
   delete ctx;
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "array_container_destroy"
static PetscErrorCode array_container_destroy(void *ptr)
{
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = PetscFree(ptr); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Convert_Array_IS"
static PetscErrorCode Convert_Array_IS(MPI_Comm comm, mfem::Array<int> *list,
                                       PetscInt st, IS* is)
{
   PetscInt n = list->Size(),*idxs;
   const int *data = list->GetData();

   PetscFunctionBeginUser;
   ierr = PetscMalloc1(n,&idxs); CHKERRQ(ierr);
   for (PetscInt i=0; i<n; i++) { idxs[i] = data[i] + st; }
   ierr = ISCreateGeneral(comm,n,idxs,PETSC_OWN_POINTER,is);
   PetscFunctionReturn(0);
}

// TODO: These functions can be eventually moved to PETSc source code.
#include "_hypre_parcsr_mv.h"
#undef __FUNCT__
#define __FUNCT__ "MatConvert_hypreParCSR_AIJ"
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix* hA,Mat* pA)
{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(hA);
   hypre_CSRMatrix *hdiag,*hoffd;
   PetscScalar     *da,*oa,*aptr;
   PetscInt        *dii,*djj,*oii,*ojj,*iptr;
   PetscInt        i,dnnz,onnz,m,n;
   PetscMPIInt     size;
   PetscErrorCode  ierr;

   PetscFunctionBeginUser;
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   m     = hypre_CSRMatrixNumRows(hdiag);
   n     = hypre_CSRMatrixNumCols(hdiag);
   dnnz  = hypre_CSRMatrixNumNonzeros(hdiag);
   onnz  = hypre_CSRMatrixNumNonzeros(hoffd);
   ierr  = PetscMalloc1(m+1,&dii); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&djj); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&da); CHKERRQ(ierr);
   ierr  = PetscMemcpy(dii,hypre_CSRMatrixI(hdiag),(m+1)*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(djj,hypre_CSRMatrixJ(hdiag),dnnz*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(da,hypre_CSRMatrixData(hdiag),dnnz*sizeof(PetscScalar));
   CHKERRQ(ierr);
   /* TODO should we add a case when the J pointer is already sorted? */
   iptr = djj;
   aptr = da;
   for (i=0; i<m; i++)
   {
      PetscInt nc = dii[i+1]-dii[i];
      ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
      iptr += nc;
      aptr += nc;
   }
   ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
   if (size > 1)
   {
      PetscInt *offdj,*coffd;

      ierr  = PetscMalloc1(m+1,&oii); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&ojj); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&oa); CHKERRQ(ierr);
      ierr  = PetscMemcpy(oii,hypre_CSRMatrixI(hoffd),(m+1)*sizeof(PetscInt));
      CHKERRQ(ierr);
      offdj = hypre_CSRMatrixJ(hoffd);
      coffd = hypre_ParCSRMatrixColMapOffd(hA);
      for (i=0; i<onnz; i++) { ojj[i] = coffd[offdj[i]]; }
      ierr  = PetscMemcpy(oa,hypre_CSRMatrixData(hoffd),onnz*sizeof(PetscScalar));
      CHKERRQ(ierr);
      /* TODO should we add a case when the J pointer is already sorted? */
      iptr = ojj;
      aptr = oa;
      for (i=0; i<m; i++)
      {
         PetscInt nc = oii[i+1]-oii[i];
         ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
         iptr += nc;
         aptr += nc;
      }
      ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,dii,
                                            djj,da,oii,ojj,oa,pA); CHKERRQ(ierr);
   }
   else
   {
      oii = ojj = NULL;
      oa = NULL;
      ierr = MatCreateSeqAIJWithArrays(comm,m,n,dii,djj,da,pA); CHKERRQ(ierr);
   }
   /* We are responsible to free the CSR arrays.
      However, since we can take references of a PetscParMatrix
      but we cannot take reference of PETSc arrays,
      we need to create a PetscContainer object
      to take reference of these arrays in
      reference objects */
   void *ptrs[6] = {dii,djj,da,oii,ojj,oa};
   const char *names[6] = {"_mfem_csr_dii",
                           "_mfem_csr_djj",
                           "_mfem_csr_da",
                           "_mfem_csr_oii",
                           "_mfem_csr_ojj",
                           "_mfem_csr_oa"
                          };
   for (i=0; i<6; i++)
   {
      PetscContainer c;

      ierr = PetscContainerCreate(comm,&c); CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]); CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,array_container_destroy);
      CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*pA),names[i],(PetscObject)c);
      CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c); CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_hypreParCSR_IS"
static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix* hA,Mat* pA)
{
   Mat                    lA;
   ISLocalToGlobalMapping rl2g,cl2g;
   IS                     is;
   hypre_CSRMatrix        *hdiag,*hoffd;
   MPI_Comm               comm = hypre_ParCSRMatrixComm(hA);
   void                   *ptrs[2];
   const char             *names[2] = {"_mfem_csr_aux",
                                       "_mfem_csr_data"
                                      };
   PetscScalar            *hdd,*hod,*aa,*data;
   PetscInt               *col_map_offd,*hdi,*hdj,*hoi,*hoj;
   PetscInt               *aux,*ii,*jj;
   PetscInt               cum,dr,dc,oc,str,stc,nnz,i,jd,jo;
   PetscErrorCode         ierr;

   PetscFunctionBeginUser;
   /* access relevant information in ParCSR */
   str   = hypre_ParCSRMatrixFirstRowIndex(hA);
   stc   = hypre_ParCSRMatrixFirstColDiag(hA);
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   dr    = hypre_CSRMatrixNumRows(hdiag);
   dc    = hypre_CSRMatrixNumCols(hdiag);
   nnz   = hypre_CSRMatrixNumNonzeros(hdiag);
   hdi   = hypre_CSRMatrixI(hdiag);
   hdj   = hypre_CSRMatrixJ(hdiag);
   hdd   = hypre_CSRMatrixData(hdiag);
   oc    = hypre_CSRMatrixNumCols(hoffd);
   nnz  += hypre_CSRMatrixNumNonzeros(hoffd);
   hoi   = hypre_CSRMatrixI(hoffd);
   hoj   = hypre_CSRMatrixJ(hoffd);
   hod   = hypre_CSRMatrixData(hoffd);

   /* generate l2g maps for rows and cols */
   ierr = ISCreateStride(comm,dr,str,1,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);
   col_map_offd = hypre_ParCSRMatrixColMapOffd(hA);
   ierr = PetscMalloc1(dc+oc,&aux); CHKERRQ(ierr);
   for (i=0; i<dc; i++) { aux[i]    = i+stc; }
   for (i=0; i<oc; i++) { aux[i+dc] = col_map_offd[i]; }
   ierr = ISCreateGeneral(comm,dc+oc,aux,PETSC_OWN_POINTER,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);

   /* create MATIS object */
   ierr = MatCreate(comm,pA); CHKERRQ(ierr);
   ierr = MatSetSizes(*pA,dr,dc,PETSC_DECIDE,PETSC_DECIDE); CHKERRQ(ierr);
   ierr = MatSetType(*pA,MATIS); CHKERRQ(ierr);
   ierr = MatSetLocalToGlobalMapping(*pA,rl2g,cl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&rl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&cl2g); CHKERRQ(ierr);

   /* merge local matrices */
   ierr = PetscMalloc1(nnz+dr+1,&aux); CHKERRQ(ierr);
   ierr = PetscMalloc1(nnz,&data); CHKERRQ(ierr);
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   *ii  = *(hdi++) + *(hoi++);
   for (jd=0,jo=0,cum=0; *ii<nnz; cum++)
   {
      PetscScalar *aold = aa;
      PetscInt    *jold = jj,nc = jd+jo;
      for (; jd<*hdi; jd++) { *jj++ = *hdj++;      *aa++ = *hdd++; }
      for (; jo<*hoi; jo++) { *jj++ = *hoj++ + dc; *aa++ = *hod++; }
      *(++ii) = *(hdi++) + *(hoi++);
      ierr = PetscSortIntWithScalarArray(jd+jo-nc,jold,aold); CHKERRQ(ierr);
   }
   for (; cum<dr; cum++) { *(++ii) = nnz; }
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,dr,dc+oc,ii,jj,aa,&lA);
   CHKERRQ(ierr);
   ptrs[0] = aux;
   ptrs[1] = data;
   for (i=0; i<2; i++)
   {
      PetscContainer c;

      ierr = PetscContainerCreate(PETSC_COMM_SELF,&c); CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]); CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,array_container_destroy);
      CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)lA,names[i],(PetscObject)c);
      CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c); CHKERRQ(ierr);
   }
   ierr = MatISSetLocalMat(*pA,lA); CHKERRQ(ierr);
   ierr = MatDestroy(&lA); CHKERRQ(ierr);
   ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

#endif
#endif
