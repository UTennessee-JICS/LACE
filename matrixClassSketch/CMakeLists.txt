cmake_minimum_required (VERSION 2.6)
project (MatrixClassTest)
# The version number.
set (MatrixClassTest_VERSION_MAJOR 1)
set (MatrixClassTest_VERSION_MINOR 0)
set (CMAKE_INSTALL_PREFIX /home/slw/Documents/repositories/matrixClassSketch01)
set (CMAKE_CXX_STANDARD 11)

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_matrix test_matrix.cpp)

# Link test executable against gtest & gtest_main
target_link_libraries(test_matrix ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
