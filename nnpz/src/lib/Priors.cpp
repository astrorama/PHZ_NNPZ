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

#include "Nnpz/Priors.h"
#include <cmath>

namespace Nnpz {

  LogNormalPrior::LogNormalPrior(double mu, double sigma): 
           m_mu{mu}, m_sigma{sigma} {
     getBound();
  }

  std::pair<double, double> LogNormalPrior::getValidRange() const {
    return std::make_pair(min_s,max_s);
  }
  
  double LogNormalPrior::operator()(double x) const {
    return computeValue(x);
  };

  double LogNormalPrior::dx(double x) const {
    return - exp(-(log(x)-m_mu)*(log(x)-m_mu)/(2*m_sigma*m_sigma))*(-m_mu + m_sigma*m_sigma + log(x))/(m_sigma*m_sigma*x*x);
  }
  
  double LogNormalPrior::computeValue(double x) const{
    return  exp(-(log(x)-m_mu)*(log(x)-m_mu)/(2*m_sigma*m_sigma)+m_mu-m_sigma*m_sigma/2)/x;
  }
  
  void LogNormalPrior::getBound(){
     double min_prior = 1e-10;   
     // No analytic function for getting the bound: scan the scale range
     min_s = 1e-10;
     max_s = -1;
     for (int n=1;n<12;n++){
         for (int item=9; item>0; item--){
            double scale = item*pow(10,-n);
            double value = computeValue(scale);
            if (value <= min_prior){
               min_s = scale;
               break;
            }
         }
         if (min_s != 1e-10){
             break;
         }
     }  
 
     double scale=1.0;
     while (max_s<0){
        double value = computeValue(scale);
        if (value <= min_prior){
               max_s = scale;
        }
        scale+=0.1;
     }
  }
}
