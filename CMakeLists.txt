cmake_minimum_required(VERSION 3.15)
project(MTPL VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library
add_library(mtpl INTERFACE)
target_include_directories(mtpl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Tests (optional, only build if this is the main project)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    add_executable(test_mtpl_core tests/test_mtpl_core.cpp)
    target_link_libraries(test_mtpl_core mtpl)
    add_test(NAME test_mtpl_core COMMAND test_mtpl_core)
    
    add_executable(test_signal_transform tests/test_signal_transform.cpp)
    target_link_libraries(test_signal_transform mtpl)
    add_test(NAME test_signal_transform COMMAND test_signal_transform)
    
    add_executable(test_immutability tests/test_immutability.cpp)
    target_link_libraries(test_immutability mtpl)
    add_test(NAME test_immutability COMMAND test_immutability)
endif()