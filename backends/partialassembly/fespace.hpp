// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BACKENDS_PA_FESPACE_HPP
#define MFEM_BACKENDS_PA_FESPACE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "util.hpp"
#include "engine.hpp"
#include "array.hpp"
#include "vector.hpp"
#include "../../fem/fem.hpp"

namespace mfem
{

namespace pa
{

/// TODO: doxygen
template <Location Device>
class PAFiniteElementSpace : public mfem::PFiniteElementSpace
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // mfem::FiniteElementSpace *fes;

   LayoutType<Device> e_layout;

   mfem::Array<int> *tensor_offsets, *tensor_indices;

   void BuildDofMaps();

public:
   PAFiniteElementSpace() = delete;

   /// Nearly-empty class that stores a pointer to a mfem::FiniteElementSpace instance and the engine
   PAFiniteElementSpace(const PAEngine<Device> &e, mfem::FiniteElementSpace &fespace);

   /// Virtual destructor
   virtual ~PAFiniteElementSpace()
   {
      delete tensor_offsets;
      delete tensor_indices;
   }

   LayoutType<Device>& GetELayout() { return e_layout; }

   /// Return the engine as an OpenMP engine
   const PAEngine<Device>& GetEngine() { return static_cast<const PAEngine<Device>&>(*engine); }

   /// Convert an E vector to L vector
   void ToLVector(const VectorType<Device, double>& e_vector, VectorType<Device, double>& l_vector);

   /// Covert an L vector to E vector
   void ToEVector(const VectorType<Device, double>& l_vector, VectorType<Device, double>& e_vector);

   const FiniteElement *GetFE(int i) const { return fes->GetFE(i); }

   /// Returns number of degrees of freedom in each direction.
   inline const int GetNDofs1d() const { return GetFE(0)->GetOrder() + 1; }

   /// Returns number of quadrature points in each direction.
   inline const int GetNQuads1d(const int order) const
   {
      const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);
      return ir1d.GetNPoints();
   }
};

template <Location Device>
PAFiniteElementSpace<Device>::PAFiniteElementSpace(const PAEngine<Device> &e,
      mfem::FiniteElementSpace &fespace)
   : PFiniteElementSpace(e, fespace),
     e_layout(e, 0),
     tensor_offsets(NULL),
     tensor_indices(NULL)
{
   std::size_t lsize = 0;
   for (int e = 0; e < fespace.GetNE(); e++) { lsize += fespace.GetFE(e)->GetDof(); }
   e_layout.Resize(lsize);
   e_layout.DontDelete();
}

template <Location Device>
void PAFiniteElementSpace<Device>::BuildDofMaps()
{
   mfem::FiniteElementSpace *mfem_fes = fes;

   const int local_size = GetELayout().Size();
   const int global_size = mfem_fes->GetVLayout()->Size();
   const int vdim = mfem_fes->GetVDim();

   // Now we can allocate and fill the global map
   tensor_offsets = new mfem::Array<int>(*(new LayoutType<Device>(GetEngine(), global_size + 1)));
   tensor_indices = new mfem::Array<int>(*(new LayoutType<Device>(GetEngine(), local_size)));

   mfem::Array<int> offsets(tensor_offsets->Size());
   mfem::Array<int> indices(tensor_indices->Size());

   mfem::Array<int> global_map(local_size);
   mfem::Array<int> elem_vdof;

   int offset = 0;
   for (int e = 0; e < mfem_fes->GetNE(); e++)
   {
      const FiniteElement *fe = mfem_fes->GetFE(e);
      const int dofs = fe->GetDof();
      const int vdofs = dofs * vdim;
      const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement *>(fe);
      const mfem::Array<int> &dof_map = tfe->GetDofMap();

      mfem_fes->GetElementVDofs(e, elem_vdof);

      if (dof_map.Size() == 0)
      {
         for (int vd = 0; vd < vdim; vd++)
            for (int i = 0; i < vdofs; i++)
            {
               global_map[offset + dofs * vd + i] = elem_vdof[dofs * vd + i];
            }
      } else {
         for (int vd = 0; vd < vdim; vd++)
            for (int i = 0; i < vdofs; i++)
            {
               global_map[offset + dofs * vd + i] = elem_vdof[dofs * vd + dof_map[i]];
            }
      }
      offset += vdofs;
   }

   // global_map[i] = index in global vector for local dof i
   // NOTE: multiple i values will yield same global_map[i] for shared DOF.

   // We want to now invert this map so we have indices[j] = (local dof for global dof j).

   // Zero the offset vector
   offsets = 0;

   // Keep track of how many local dof point to its global dof
   // Count how many times each dof gets hit
   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      ++offsets[g + 1];
   }
   // Aggregate the offsets
   for (int i = 1; i <= global_size; i++)
   {
      offsets[i] += offsets[i - 1];
   }

   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      indices[offsets[g]++] = i;
   }

   // Shift the offset vector back by one, since it was used as a
   // counter above.
   for (int i = global_size; i > 0; i--)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;

   tensor_offsets->PushData(offsets.GetData());
   tensor_indices->PushData(indices.GetData());
   // offsets.Push();
   // indices.Push();
}

void toLVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const HostVector<double>& e_vector, HostVector<double>& l_vector);
#ifdef __NVCC__
void toLVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const CudaVector<double>& e_vector, CudaVector<double>& l_vector);
#endif

/// Convert an E vector to L vector
template <Location Device>
void PAFiniteElementSpace<Device>::ToLVector(const VectorType<Device, double>& e_vector, VectorType<Device, double>& l_vector)
{
   if (tensor_indices == NULL) BuildDofMaps();
   if (l_vector.Size() != (std::size_t) GetFESpace()->GetVSize())
   {
      l_vector.template Resize<double>(GetFESpace()->GetVLayout(), NULL);
   }
   toLVector(*tensor_offsets, *tensor_indices, e_vector, l_vector);
}

void toEVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const HostVector<double>& l_vector, HostVector<double>& e_vector);
#ifdef __NVCC__
void toEVector(const mfem::Array<int>& tensor_offsets, const mfem::Array<int>& tensor_indices
               , const CudaVector<double>& l_vector, CudaVector<double>& e_vector);
#endif

/// Covert an L vector to E vector
template <Location Device>
void PAFiniteElementSpace<Device>::ToEVector(const VectorType<Device, double>& l_vector, VectorType<Device, double>& e_vector)
{
   if (tensor_indices == NULL) BuildDofMaps();
   if (e_vector.Size() != (std::size_t) e_layout.Size())
   {
      e_vector.template Resize<double>(GetELayout(), NULL);
   }
   toEVector(*tensor_offsets, *tensor_indices, l_vector, e_vector);
}

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_FESPACE_HPP