/*
 * Copyright (C) 2022 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under the terms of
 * the GNU Lesser General Public License as published by the Free Software Foundation;
 * either version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301 USA
 */

#ifndef PHZ_NNPZ_MAXHEAP_H
#define PHZ_NNPZ_MAXHEAP_H

#include "Types.h"

namespace Nnpz {

/**
 * Implementation of a max-heap over an array: the element at position 0
 * has a distance higher than any other element of the array.
 */
class MaxHeap {
public:
  /**
   * Constructor
   * @param k
   *    Size of the heap
   * @param distances
   *    Memory area where to keep the distances
   * @param indexes
   *    Memory area where to keep the indexes
   * @param scales
   *    Memory area where to keep the scales
   */
  MaxHeap(index_t k, float* distances, index_t* indexes, scale_t* scales)
      : m_k(k), m_count(0), m_distances(distances), m_indexes(indexes), m_scales(scales){};

  /**
   * @return How many elements have been pushed into the heap
   */
  unsigned count() const {
    return m_count;
  }

  /**
   * @return Size of the heap
   */
  unsigned size() const {
    return m_k;
  }

  /**
   * @return The top of the heap: the maximum stored distance
   * @warning If no elements are present, the value is undefined!
   */
  float top() const {
    return m_distances[0];
  };

  /**
   * Add a new element to the heap
   * @param idx
   *    Element index (i.e. reference sample index)
   * @param distance
   *    Element distance, used to keep the heap
   * @param scale
   *    Element scaling factor
   * @warning
   *    There is no bound check! Make sure no elements are added passed the end of the heap.
   */
  void add(index_t idx, float distance, scale_t scale) {
    index_t i      = m_count;
    m_distances[i] = distance;
    m_indexes[i]   = idx;
    m_scales[i]    = scale;
    ++m_count;
    int parent = int((i - 1) / 2);
    while (i > 0 && m_distances[parent] < m_distances[i]) {
      std::swap(m_distances[i], m_distances[parent]);
      std::swap(m_indexes[i], m_indexes[parent]);
      std::swap(m_scales[i], m_scales[parent]);
      i      = parent;
      parent = int((i - 1) / 2);
    }
  }

  /**
   * Remove the top of the heap
   * @warning
   *    There is no bound check! Make sure no elements are removed from an empty heap
   */
  void pop() {
    --m_count;
    m_distances[0]     = m_distances[m_count];
    m_indexes[0]       = m_indexes[m_count];
    m_scales[0]        = m_scales[m_count];
    m_indexes[m_count] = -1;

    index_t largest = 0, i = 0;
    do {
      i = largest;

      index_t left  = 2 * i + 1;
      index_t right = left + 1;
      if (left < m_count && m_distances[left] >= m_distances[largest]) {
        largest = left;
      }
      if (right < m_count && m_distances[right] >= m_distances[largest]) {
        largest = right;
      }
      if (largest != i) {
        std::swap(m_distances[i], m_distances[largest]);
        std::swap(m_indexes[i], m_indexes[largest]);
        std::swap(m_scales[i], m_scales[largest]);
      }
    } while (largest != i);
  }

private:
  index_t  m_k, m_count;
  float*   m_distances;
  index_t* m_indexes;
  scale_t* m_scales;
};

/**
 * Insert an element into the heap *only if* its distance is closer than the furthest element
 * of the heap, which is then removed.
 * @param heap
 * @param index
 * @param distance
 * @param scale
 */
inline void insert_if_best(MaxHeap& heap, index_t index, float distance, float scale) {
  if (heap.count() < heap.size()) {
    heap.add(index, distance, scale);
  } else if (heap.top() > distance) {
    heap.pop();
    heap.add(index, distance, scale);
  }
}

}  // namespace Nnpz

#endif  // PHZ_NNPZ_MAXHEAP_H
