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
#include "AlexandriaKernel/memory_tools.h"
#include "MathUtils/function/Function.h"
#include "MathUtils/root/SecantMethod.h"
#include "Nnpz/Distances.h"
#include "Nnpz/Priors.h"
#include "Nnpz/Types.h"
#include <iostream>

using namespace Euclid::MathUtils;
using Euclid::make_unique;
namespace py = pybind11;

namespace Nnpz {

template <typename TDistance>
class DistanceDerivative : public Function {
public:
  DistanceDerivative(photo_t const* reference, photo_t const* target, py::ssize_t nbands)
      : m_reference(reference), m_target(target), m_nbands(nbands){};

  virtual ~DistanceDerivative() = default;

  double operator()(double scale) const override {
    return TDistance::daDistance(scale, m_reference, m_target, m_nbands);
  }

  using Function::operator();

  std::unique_ptr<Function> clone() const override {
    return make_unique<DistanceDerivative>(*this);
  }

private:
  photo_t const *m_reference, *m_target;
  py::ssize_t    m_nbands;
};

template <typename TDistance, typename TPrior>
class ScaleWithPrior final : public Function {
public:
  ScaleWithPrior(DistanceDerivative<TDistance>& da_distance, TPrior& prior)
      : m_da_distance(da_distance), m_prior(prior) {}

  double operator()(double scale) const override {
    return m_da_distance(scale) / 2. - m_prior.dx(scale) / m_prior(scale);
  }

  std::unique_ptr<NAryFunction> clone() const override {
    return Euclid::make_unique<ScaleWithPrior>(*this);
  }

  using Function::operator();

private:
  DistanceDerivative<TDistance>& m_da_distance;
  TPrior&                        m_prior;
};

template <typename TDistance, typename TPrior>
class ScaleCalculator final : public ScaleFunction {
public:
  template <typename... Args>
  ScaleCalculator(ScaleFunctionParams const& params, Args... args)
      : m_prior(std::forward<Args>(args)...), m_secant_params{params.maxiter, params.tolerance, 1, 1} {
    std::tie(m_secant_params.min, m_secant_params.max) = m_prior.getValidRange();
  }

  virtual ~ScaleCalculator() = default;

  float operator()(photo_t const* ref_photo, photo_t const* target_photo, py::ssize_t nbands) override {
    if (m_secant_params.min == m_secant_params.max) {
      return m_secant_params.min;
    }

    DistanceDerivative<TDistance> target(ref_photo, target_photo, nbands);
    auto                          guess = TDistance::guessScale(ref_photo, target_photo, nbands);

    if (guess <= m_secant_params.min) {
      return m_secant_params.min;
    }
    if (guess >= m_secant_params.max) {
      return m_secant_params.max;
    }

    double x0 = guess - EPS, x1 = guess + EPS;
    return secantMethod(ScaleWithPrior<TDistance, TPrior>(target, m_prior), x0, x1, m_secant_params).first;
  }

private:
  static constexpr double EPS = 1e-4;
  TPrior                  m_prior;
  SecantParams            m_secant_params;
};

std::shared_ptr<ScaleFunction> scaleFunctionFactory(std::string const& prior, ScaleFunctionParams const& params) {
  if (prior == "uniform") {
    return std::make_shared<ScaleCalculator<Chi2Distance, Uniform>>(params);
  }
  throw Elements::Exception() << "Unknown prior " << prior;
}

}  // namespace Nnpz