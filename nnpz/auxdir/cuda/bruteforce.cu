/*
 * Copyright (C) 2022 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 3.0 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <cstdint>

/**
 * Swap a and b
 */
template <typename T> __device__ void swap(T &a, T &b) {
  T c(a);
  a = b;
  b = c;
}

/**
 * Implementation of a max-heap over an array: the element at position 0
 * has a distance higher than any other element of the array.
 */
class MaxHeap {
public:
  __device__ MaxHeap(int k, double *__restrict__ distances,
                     int32_t *__restrict__ indexes, double *__restrict__ scales)
      : m_k(k), m_count(0), m_distances(distances), m_indexes(indexes),
        m_scales(scales){};

  __device__ int count() const { return m_count; }

  __device__ int size() const { return m_k; }

  __device__ double top() const { return m_distances[0]; };

  __device__ void add(int idx, double distance, double scale) {
    int i = m_count;
    m_distances[i] = distance;
    m_indexes[i] = idx;
    m_scales[i] = scale;
    ++m_count;
    int parent = int((i - 1) / 2);
    while (i > 0 && m_distances[parent] < m_distances[i]) {
      swap(m_distances[i], m_distances[parent]);
      swap(m_indexes[i], m_indexes[parent]);
      swap(m_scales[i], m_scales[parent]);
      i = parent;
      parent = int((i - 1) / 2);
    }
  }

  __device__ void pop() {
    --m_count;
    m_distances[0] = m_distances[m_count];
    m_indexes[0] = m_indexes[m_count];
    m_indexes[m_count] = -1;

    int largest = 0, i = 0;
    do {
      i = largest;

      int left = 2 * i + 1;
      int right = left + 1;
      if (left < m_count && m_distances[left] >= m_distances[largest]) {
        largest = left;
      }
      if (right < m_count && m_distances[right] >= m_distances[largest]) {
        largest = right;
      }
      if (largest != i) {
        swap(m_distances[i], m_distances[largest]);
        swap(m_indexes[i], m_indexes[largest]);
        swap(m_scales[i], m_scales[largest]);
      }
    } while (largest != i);
  }

private:
  int m_k, m_count;
  double *__restrict__ m_distances;
  int32_t *__restrict__ m_indexes;
  double *__restrict__ m_scales;
};

/**
 * To avoid sorting the distances of all the reference objects,
 * we use a MaxHeap to keep the k-th closest elements.
 * The root of the heap is the last found k-th neighbor.
 * If a new object is closer, then we remove the top of the heap
 * and add a new neighbor.
 * In our case, k << nreference, so it is *way* more performant to do it
 * this way.
 * @param heap
 * @param index
 * @param distance
 */
__device__ void insert_if_best(MaxHeap &heap, int index, double distance,
                               double scale) {
  if (heap.count() < heap.size()) {
    heap.add(index, distance, scale);
  } else if (heap.top() > distance) {
    heap.pop();
    heap.add(index, distance, scale);
  }
}

__device__ double compute_scale(double const *__restrict__ reference,
                               double const *__restrict__ target, int nbands) {
  double nom = 0., den = 0.;

  for (int band_idx = 0; band_idx < nbands; ++band_idx) {
    double ref_val = reference[band_idx * 2];
    double tar_val = target[band_idx * 2];
    double tar_err = target[band_idx * 2 + 1];
    double err_sqr = (tar_err * tar_err);

    nom += (ref_val * tar_val) / err_sqr;
    den += (ref_val * ref_val) / err_sqr;
  }

  return min(1e5, max(1e-5, nom / den));
}

/**
 * Compute the chi2 distance between the target catalog and the reference
 * sample. Each target object gets assigned to a single GPU thread.
 * Both reference and catalog photometries are expected to be 32 bits
 * floats in row major. The first axis corresponds to the different objects,
 * the second axis to the different bands, and the third axis has only two
 * positions: value / error
 *
 * Note that we use __restrict__ for the output since it must not
 * overlap with any of the inputs. The compiler can use this hint for
 * optimizing memory access, saving more than 10% computing time.
 *
 * all_distances and all_closest are expected to be a row-major matrix
 * of (ntarget x k) elements.
 */
extern "C" __global__ void chi2_bruteforce(
    double const *__restrict__ reference, double const *__restrict__ all_targets,
    double *__restrict__ all_distances, double *__restrict__ all_scales,
    int32_t *__restrict__ all_closest, int const k, int const nbands,
    int const nreference, int const ntarget, bool const scaling) {
  unsigned target_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (target_idx < ntarget) {
    MaxHeap heap(k, all_distances + target_idx * k,
                 all_closest + target_idx * k, all_scales + target_idx * k);

    double const *__restrict__ target = all_targets + target_idx * nbands * 2;

    for (int ref_idx = 0; ref_idx < nreference; ++ref_idx) {
      double acc = 0.;
      int ref_offset = ref_idx * nbands * 2;

      double scale = 1.;
      if (scaling) {
        scale = compute_scale(reference + ref_offset, target, nbands);
      }

      for (int band_idx = 0; band_idx < nbands; ++band_idx) {
        double ref_val = reference[ref_offset + band_idx * 2];
        double ref_err = reference[ref_offset + band_idx * 2 + 1];
        double tar_val = target[band_idx * 2];
        double tar_err = target[band_idx * 2 + 1];
        double nom = (ref_val - tar_val) * (ref_val - tar_val);
        double den = ref_err * ref_err + tar_err * tar_err;
        acc += nom / den;
      }

      insert_if_best(heap, ref_idx, acc, 1.);
    }
  }
}
