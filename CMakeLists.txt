CMAKE_MINIMUM_REQUIRED(VERSION 3.25)

#===============================================================================
# Load macros and functions for Elements-based projects
#===============================================================================

find_package(ElementsProject)

#===============================================================================
# Declare project name and version
# Example with dependency:
#                         elements_project(MyProject 1.0 USE Element 3.9)
#===============================================================================

elements_project(PHZ_NNPZ 1.4.4 USE Alexandria 2.32.2)
