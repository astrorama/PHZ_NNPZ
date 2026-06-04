/*
 * Copyright (C) 2012-2020 Euclid Science Ground Segment
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

/**
 * @file Prior_test.cpp
 * @date 2025/05/08
 * @author Florian Dubath
 */

#include <boost/test/unit_test.hpp>
#include "Nnpz/Priors.h"
#include <cmath>


using namespace Euclid;

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(Prior_test)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Uniform_test) {
  auto prior = Nnpz::Uniform();
  
  auto range = prior.getValidRange();
  BOOST_CHECK(isinf(range.first));
  BOOST_CHECK(range.first<0);
  BOOST_CHECK(isinf(range.second));
  BOOST_CHECK(range.second>0);
  
  BOOST_CHECK_CLOSE(prior(0.0),1.0, 1e-4);
  BOOST_CHECK_CLOSE(prior(1.0),1.0, 1e-4);
  BOOST_CHECK_CLOSE(prior(10.0),1.0, 1e-4);
  
  BOOST_CHECK_SMALL(prior.dx(0.0), 1e-4);
  BOOST_CHECK_SMALL(prior.dx(1.0), 1e-4);
  BOOST_CHECK_SMALL(prior.dx(10.0), 1e-4);
}

BOOST_AUTO_TEST_CASE(Tophat_test) {
  auto prior = Nnpz::Tophat(0.5,2.0);
  
  auto range = prior.getValidRange();
  BOOST_CHECK_CLOSE(range.first,0.5, 1e-4);
  BOOST_CHECK_CLOSE(range.second,2.0, 1e-4);
  
  BOOST_CHECK_SMALL(prior(0.0), 1e-4);
  BOOST_CHECK_CLOSE(prior(1.0),1.0, 1e-4);
  BOOST_CHECK_SMALL(prior(10.0), 1e-4);
  
  BOOST_CHECK_SMALL(prior.dx(0.0), 1e-4);
  BOOST_CHECK_SMALL(prior.dx(1.0), 1e-4);
  BOOST_CHECK_SMALL(prior.dx(10.0), 1e-4);
}

BOOST_AUTO_TEST_CASE(Delta_test) {
  auto prior = Nnpz::Delta(1.0);
  
  auto range = prior.getValidRange();
  BOOST_CHECK_CLOSE(range.first,1.0, 1e-4);
  BOOST_CHECK_CLOSE(range.second,1.0, 1e-4);
  
  BOOST_CHECK_SMALL(prior(0.9), 1e-4);
  BOOST_CHECK_CLOSE(prior(1.0),1.0, 1e-4);
  BOOST_CHECK_SMALL(prior(1.1), 1e-4);
  
  BOOST_CHECK_SMALL(prior.dx(0.0), 1e-4);
  BOOST_CHECK_SMALL(prior.dx(1.0), 1e-4);
  BOOST_CHECK_SMALL(prior.dx(10.0), 1e-4);
}

BOOST_AUTO_TEST_CASE(Gaussian_test) {
  auto prior = Nnpz::GaussianPrior(1.0,0.05);
  
  auto range = prior.getValidRange();
  BOOST_CHECK_CLOSE(range.first,0.66069623274918077, 1e-4);
  BOOST_CHECK_CLOSE(range.second,1.3393037672508192, 1e-4);
  
  BOOST_CHECK_CLOSE(prior(0.8),0.0003354626279025136, 1e-4);
  BOOST_CHECK_CLOSE(prior(1.0),1.0, 1e-4);
  BOOST_CHECK_CLOSE(prior(1.1),0.13533528323661226, 1e-4);
  
  BOOST_CHECK_CLOSE(prior.dx(0.8),0.001341850511610054, 1e-4);
  BOOST_CHECK_SMALL(prior.dx(1.0), 1e-4);
  BOOST_CHECK_CLOSE(prior.dx(1.1),-0.27067056647322474, 1e-4);
}

BOOST_AUTO_TEST_CASE(LogNormal_test) {
  auto prior = Nnpz::LogNormalPrior(0.1,sqrt(0.1));
  
  auto range = prior.getValidRange();
  BOOST_CHECK_CLOSE(range.first,0.1, 1e-4);
  BOOST_CHECK_CLOSE(range.second,8.6, 1e-4);
  
  BOOST_CHECK_CLOSE(prior(0.8),0.7796070857255001, 1e-4);
  BOOST_CHECK_CLOSE(prior(1.0),1.0, 1e-4);
  BOOST_CHECK_CLOSE(prior(1.1),0.9555959020699539, 1e-4);

  
  BOOST_CHECK_CLOSE(prior.dx(0.8),2.068499437706946, 1e-4);
  BOOST_CHECK_SMALL(prior.dx(1.0), 1e-4);
  BOOST_CHECK_CLOSE(prior.dx(1.1),-0.7876008175464965, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()
