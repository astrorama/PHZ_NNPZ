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

#include "Nnpz/Scaling.h"
#include "Nnpz/Priors.h"
#include "Nnpz/Types.h"
#include <AlexandriaKernel/memory_tools.h>
#include <MathUtils/distances/Distances.h>
#include <MathUtils/function/Function.h>
#include <MathUtils/root/SecantMethod.h>
#include <sstream>

using namespace Euclid::MathUtils;
using Euclid::make_unique;

namespace Nnpz {

template <typename TDistance>
class DistanceDerivative final : public Function {
public:
  DistanceDerivative(PhotoPtrIterator ref_begin, PhotoPtrIterator ref_end, PhotoPtrIterator target_begin)
      : m_ref_begin(ref_begin), m_ref_end(ref_end), m_target_begin(target_begin) {}

  ~DistanceDerivative() override = default;

  double operator()(double scale) const override {
    return TDistance::daDistance(scale, m_ref_begin, m_ref_end, m_target_begin);
  }

  using Function::operator();

  std::unique_ptr<Function> clone() const override {
    return make_unique<DistanceDerivative>(*this);
  }

private:
  PhotoPtrIterator const m_ref_begin;
  PhotoPtrIterator const m_ref_end;
  PhotoPtrIterator       m_target_begin;
};

template <typename TDistance, typename TPrior>
class ScaleWithPrior final : public Function {
public:
  ScaleWithPrior(DistanceDerivative<TDistance> const& da_distance, TPrior const& prior)
      : m_da_distance(da_distance), m_prior(prior) {}

  double operator()(double scale) const override {
    return m_da_distance(scale) / 2. - m_prior.dx(scale) / m_prior(scale);
  }

  std::unique_ptr<NAryFunction> clone() const override {
    return Euclid::make_unique<ScaleWithPrior>(*this);
  }

  using Function::operator();

private:
  DistanceDerivative<TDistance> const& m_da_distance;
  TPrior const&                        m_prior;
};

template <typename TDistance, typename TPrior>
class ScaleCalculator final : public ScaleFunction {
public:
  template <typename... Args>
  ScaleCalculator(ScaleFunctionParams const& params, Args&&... args)
      : m_prior(std::forward<Args>(args)...), m_secant_params{params.maxiter, params.tolerance, 1, 1} {
    std::tie(m_secant_params.min, m_secant_params.max) = m_prior.getValidRange();
  }

  ~ScaleCalculator() override = default;

  double operator()(NdArray<photo_t> const& ref_photo, NdArray<photo_t> const& target_photo) const override {
    if (m_secant_params.min == m_secant_params.max) {
      return m_secant_params.min;
    }

    PhotoPtrIterator ref_begin(&ref_photo.at(0, 0));
    PhotoPtrIterator ref_end(ref_begin + ref_photo.shape(0));
    PhotoPtrIterator target_begin(&target_photo.at(0, 0));
    auto             guess = TDistance::guessScale(ref_begin, ref_end, target_begin);

    if (guess <= m_secant_params.min) {
      return m_secant_params.min;
    }
    if (guess >= m_secant_params.max) {
      return m_secant_params.max;
    }

    DistanceDerivative<TDistance> target_func(ref_begin, ref_end, target_begin);
    double                        x0 = guess - EPS;
    double                        x1 = guess;
    return secantMethod(ScaleWithPrior<TDistance, TPrior>(target_func, m_prior), x0, x1, m_secant_params).root;
  }

private:
  static constexpr double EPS = 1e-4;
  TPrior                  m_prior;
  SecantParams            m_secant_params;
};

std::shared_ptr<ScaleFunction> scaleFunctionFactory(std::string const& prior, ScaleFunctionParams const& params) {
  std::istringstream stream(prior);
  std::string        prior_type;
  stream >> prior_type;
  if (prior_type == "uniform") {
    return std::make_shared<ScaleCalculator<Chi2Distance, Uniform>>(params);
  } else if (prior_type == "tophat") {
    double min;
    double max;
    stream >> min >> max;
    if (stream.fail()) {
      throw Elements::Exception() << "Failed to parse the tophat parameters!";
    }
    return std::make_shared<ScaleCalculator<Chi2Distance, Tophat>>(params, min, max);
  } else if (prior_type == "delta") {
    double d;
    stream >> d;
    if (stream.fail()) {
      throw Elements::Exception() << "Failed to parse the delta parameters!";
    }
    return std::make_shared<ScaleCalculator<Chi2Distance, Delta>>(params, d);
  }
  throw Elements::Exception() << "Unknown prior " << prior;
}

}  // namespace Nnpz
