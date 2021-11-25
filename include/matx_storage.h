////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "matx_shape.h"
#include "matx_type_utils.h"

namespace matx
{
  struct owning
  {
  };
  struct non_owning
  {
  };

  /**
   * @brief Legacy storage method
   * 
   * Used to signal the old semantics where everything is stored as a shared_ptr internally
   * that may or may not have ownership
   * 
   */
  struct default_storage{};

  template <typename T>
  class pointer_buffer
  {
  public:
    using value_type = T;
    using iterator = T *;
    using citerator = T const *;

    pointer_buffer() = delete;
    pointer_buffer(T *ptr, size_t alloc) : ldata_(ptr), alloc_(alloc);
    pointer_buffer(T &&ptr, size_t alloc) : ldata_(std::move(ptr)), alloc_(alloc);
    ~pointer_buffer() {}

    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return ldata_;
    }

    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return data() + alloc;
    }

    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return data() + alloc;
    }

    size_t size() const
    {
      return alloc_;
    }

  private:
    size_t alloc_;
    T *ldata_;
  };

  template <typename T, std::enable_if_t<is_smart_ptr<T>>>
  class smart_pointer_buffer
  {
  public:
    using value_type = T;
    using iterator = T *;
    using citerator = T const *;

    pointer_buffer() = delete;
    pointer_buffer(T ptr, size_t alloc) : ldata_(ptr), alloc_(alloc);
    pointer_buffer(T &&ptr, size_t alloc) : ldata_(std::move(ptr)), alloc_(alloc);
    ~pointer_buffer() {}

    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return ldata_.get();
    }

    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return data() + alloc;
    }

    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return data() + alloc;
    }

    size_t size() const
    {
      return alloc_;
    }

  private:
    size_t alloc_;
    T ldata_;
  };

  /**
   * @brief Basic storage class to hold pointers to data
   * 
   */
  template <typename C, typename O>
  class basic_storage
  {
  public:
    using value_type = typename C::value_type;
    using T = value_type;
    using iterator = value_type*;
    using citerator = value_type const *;

    template <typename C>
    inline basic_storage(C &&obj) : container_(std::forward<C>(container))
    {
    }

    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return container_.data();
    }    

    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return container_.begin();
    }

    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return container_.end();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return container_.cbegin();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return container_.cend();
    }

  private:
    C container_;
  }
}