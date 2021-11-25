////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include <array>

namespace matx {

/**
 * @brief Type-erased generic tensor descriptor for strides and sizes
 *
 * @tparam SizeType type of sizes
 * @tparam StrideType type of strides
 */
template <typename SizeType, typename StrideType, int RANK> class tensor_desc_t {
public:

  tensor_desc_t(SizeType &&size, StrideType &&stride)
      : size_(std::forward<SizeType>(size)),
        stride_(std::forward<StrideType>(stride)) {
    MATX_STATIC_ASSERT(size.size() == stride.size(),
                       "Size and stride array sizes must match");
    MATX_STATIC_ASSERT(size.size() == RANK,
                       "Rank parameter must match array size");                       
  }

  auto __MATX_INLINE__ Size(int dim) const { return size_[dim]; }
  auto __MATX_INLINE__ Stride(int dim) const { return stride_[dim]; }
  auto __MATX_INLINE__ Shape() const { return size_; }
  auto constexpr Rank() { return RANK; }

private:
  SizeType size_;
  StrideType stride_;
};

/**
 * @brief Constant rank, dynamic size, dynamic strides
 *
 */
template <typename SizeType, typename StrideType, int RANK>
using tensor_desc_cr_ds_t =
    tensor_desc_t<std::array<SizeType, RANK>, std::array<StrideType, RANK>,
                  RANK>;

// 32-bit size and strides
template <int RANK>
using tensor_desc_cr_ds_32_32_t =
    tensor_desc_cr_dsi_dst<int32_t, int32_t, RANK>;

// 64-bit size and strides
template <int RANK>
using tensor_desc_cr_ds_64_64_t =
    tensor_desc_cr_dsi_dst<int64_t, int64_t, RANK>;

// 32-bit size and 64-bit strides
template <int RANK>
using tensor_desc_cr_ds_32_64_t =
    tensor_desc_cr_dsi_dst<int32_t, int64_t, RANK>;

// index_t based size and stride
template <int RANK>
using tensor_desc_cr_disi_dist = tensor_desc_cr_ds_t<index_t, index_t, RANK>;
} // namespace matx