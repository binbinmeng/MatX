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

#include "matx_storage.h"

namespace matx {


template <typename C, typename S, std::enable_if_t<!std::is_pointer<C>>>
__MATX_INLINE__ make_tensor(C &&container, S &&shape) {
  auto storage = basic_storage(std::forward<C>(container),
                               std::forward<S>(shape), owning{});
}

template <typename C, typename S, typename O,
          std::enable_if_t<!std::is_pointer<C>>>
__MATX_INLINE__ make_tensor(C &&container, S &&shape, O &&own) {
  auto storage = basic_storage(std::forward<C>(container),
                               std::forward<S>(shape), std::forward<O>(own));
}

template <typename T, typename S, typename O>
__MATX_INLINE__ make_tensor(T *ptr, S &&shape, O &&own) {
  auto storage = basic_storage(pointer_buffer{ptr}, std::forward<S>(shape),
                               std::forward<O>(own));
}


// make_tensor helpers
/**
 * Create a 0D tensor with managed memory
 *
 **/
template <typename T>
auto make_tensor() {
  return tensor_t<T,0>{};
}

/**
 * Create a 0D tensor with user memory
 *
 * @param data
 *   Pointer to device data
 **/
template <typename T>
auto make_tensor(T *const data) {
  return tensor_t<T,0>{data};
}

/**
 * Create a tensor with managed memory
 *
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK>
auto make_tensor(const index_t (&shape)[RANK]) {
  return tensor_t<T,RANK>{shape};
}

/**
 * Create a tensor with managed memory
 *
 * @param shape
 *   Shape of tensor
 * @param strides
 *   Strides of tensor
 **/
template <typename T, int RANK>
auto make_tensor(const index_t (&shape)[RANK], const index_t (&strides)[RANK]) {
  return tensor_t<T,RANK>{shape, strides};
}


/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK>
tensor_t<T,RANK> make_tensor(T *const data, const index_t (&shape)[RANK]) {
  return tensor_t<T,RANK>{data, shape};
}


/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param strides
 *   Strides of tensor
 **/
template <typename T, int RANK>
tensor_t<T,RANK> make_tensor(T *const data, const index_t (&shape)[RANK], const index_t (&strides)[RANK]) {
  return tensor_t<T,RANK>{data, shape, strides};
}

} // namespace matx