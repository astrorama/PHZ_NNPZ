CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

#===============================================================================
# Load elements_subdir macro here
# Examples:
#   For declaring a project module:
#         elements_subdir(ElementsExamples)
#===============================================================================
elements_subdir(nnpz)

#===============================================================================
# Load elements_depends_on_subdirs macro here
#   For creating a dependency onto an other accessible module
#         elements_depends_on_subdirs(ElementsKernel)
#===============================================================================
elements_depends_on_subdirs(ElementsKernel)
elements_depends_on_subdirs(MathUtils)
elements_depends_on_subdirs(XYDataset)

#===============================================================================
# Add the find_package macro (a pure CMake command) here to locate the
# libraries.
# Examples:
#          find_package(CppUnit)
#===============================================================================
find_package(NumPy)
find_package(pybind11 REQUIRED)

#===============================================================================
# Declare the library dependencies here
# Example:
#         elements_add_library(ElementsExamples src/Lib/*.cpp
#                     LINK_LIBRARIES Boost ElementsKernel
#                     INCLUDE_DIRS Boost ElementsKernel
#                     PUBLIC_HEADERS ElementsExamples)
#===============================================================================
elements_add_library(Nnpz src/lib/*.cpp
        LINK_LIBRARIES ElementsKernel MathUtils NdArray XYDataset
        PUBLIC_HEADERS Nnpz)
elements_add_pybind11_module(_Nnpz src/_Nnpz.cpp
        LINK_LIBRARIES Nnpz)

#===============================================================================
# Declare the executables here
# Example
# elements_add_executable(ElementsProgramExample src/Program/ProgramExample.cpp
#                        LINK_LIBRARIES Boost ElementsExamples
#                        INCLUDE_DIRS Boost ElementsExamples)
#===============================================================================

#===============================================================================
# Declare the Boost tests here
# Example:
# elements_add_unit_test(BoostClassExample tests/src/Boost/ClassExample_test.cpp
#                       EXECUTABLE BoostClassExample_test
#                       INCLUDE_DIRS ElementsExamples
#                       LINK_LIBRARIES ElementsExamples TYPE Boost)
#===============================================================================

#===============================================================================
# Use the following macro for python modules, scripts and aux files:
#  elements_install_python_modules()
#  elements_install_scripts()
#===============================================================================
elements_install_python_modules()
# elements_install_scripts()
elements_install_aux_files()


#===============================================================================
# Declare the Python programs here
# Examples :
# elements_add_python_program(PythonProgramExample
#                             ElementsExamples.PythonProgramExample)
#===============================================================================
elements_add_python_program(NnpzBuildPhotometry
        nnpz.program.BuildPhotometry)
elements_add_python_program(NnpzComputeNeighbors
        nnpz.program.ComputeNeighbors)
elements_add_python_program(NnpzCorrectPhotometry
        nnpz.program.CorrectPhotometry)
elements_add_python_program(NnpzComputeWeights
        nnpz.program.ComputeWeights)
elements_add_python_program(NnpzGenerateOutput
        nnpz.program.GenerateOutput)
elements_add_python_program(nnpz
        nnpz.program.Nnpz)

#===============================================================================
# Add the elements_install_conf_files macro
# Examples:
#          elements_install_conf_files()
#===============================================================================
elements_install_conf_files()
