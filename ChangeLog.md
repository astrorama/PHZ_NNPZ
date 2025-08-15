# Changelog

Project Coordinators: Florian Dubath @fdubath

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [1.4.4]
### Fixed
- Unified Photometry output set to 0 when input is missing in BruteForce case, to NaN in other cases. Fixed by stick to NaN.
- Unified Photometry computation: apply the Scaling where it was missing
- Unified Photometry computation: use reddened neighbor Photometry to compute the correction ratio

## [1.4.3] 2025-05-08
### Changed
- Add Gaussian and LogNormal Scaling Prior
- Update the related section in the User Manual
- Add Unit Tests for the scaling priors

## [1.4.2] 2024-11-04
### Changed
- Update the Doc and move the QuickStart data in another project 

## [1.4.1] 2024-07-05
### Changed
- Switch to Alexandria 2.31.3 

## [1.2.2] 2023-07-19
### Fixed
- Fix the scaling case: the scale was applied twice causing the weight to be off 


