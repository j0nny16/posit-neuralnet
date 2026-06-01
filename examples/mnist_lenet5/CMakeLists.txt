cmake_minimum_required(VERSION 3.9)
project(posit-neuralnet)

# USER flags (change here) ################################################
# Threads
cmake_host_system_information(RESULT N QUERY NUMBER_OF_LOGICAL_CORES)
add_definitions(-D LL_THREADS=${N})
add_definitions(-D HL_THREADS=1)

# Quire mode (0 = disabled, 1 = old standard, 2 = new standard)
add_definitions(-D QUIRE_MODE=1)

# Underflow mode (0 = disabled, -1 = round, 1 = underflows <minpos/2^1, 2 = underflow <minpos/2^2, ...)
add_definitions(-D UNDERFLOW_MODE=0)

# Optimization
set(USE_SSE ON)
set(USE_AVX ON)
set(USE_AVX2 ON)
###########################################################################

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags ######################################################
# Unix
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)

    # set Streaming SIMD Extension (SSE) instructions
	if(USE_SSE AND COMPILER_HAS_SSE_FLAG)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
	endif(USE_SSE AND COMPILER_HAS_SSE_FLAG)
    # set Advanced Vector Extensions (AVX)
	if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
	endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)
    # set Advanced Vector Extensions 2 (AVX2)
	if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -march=core-avx2")
	endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
endif()

find_package (Threads)
###########################################################################

# Compile flags ###########################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g -pg")
SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -pg")
###########################################################################

# PyTorch (LibTorch) ######################################################
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/../../include/libtorch/share/cmake/Torch")
set(CMAKE_CUDA_ARCHITECTURES "all")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-D USING_PYTORCH)
###########################################################################

# Set source and include folder ###########################################
set(SRC_FOLDER "${CMAKE_CURRENT_LIST_DIR}/src")
set(HEADER_FOLDER "${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${HEADER_FOLDER}")
###########################################################################

# Include universal (posits) ##############################################
include_directories("${CMAKE_CURRENT_LIST_DIR}/../../include/universal/include")
###########################################################################

# Include PositNN (PyTorch for posits) ####################################
include_directories("${CMAKE_CURRENT_LIST_DIR}/../../include")
###########################################################################

# Setup executables #######################################################
add_executable(train_float ${SRC_FOLDER}/train_float.cpp)
target_link_libraries(train_float ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_float "${TORCH_LIBRARIES}")
set_property(TARGET train_float PROPERTY CXX_STANDARD 17)

add_executable(train_float16 ${SRC_FOLDER}/train_float16.cpp)
target_link_libraries(train_float16 ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_float16 "${TORCH_LIBRARIES}")
set_property(TARGET train_float16 PROPERTY CXX_STANDARD 17)

add_executable(train_float8 ${SRC_FOLDER}/train_float8.cpp)
target_link_libraries(train_float8 ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_float8 "${TORCH_LIBRARIES}")
set_property(TARGET train_float8 PROPERTY CXX_STANDARD 17)

add_executable(train_bfloat ${SRC_FOLDER}/train_bfloat.cpp)
target_link_libraries(train_bfloat ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_bfloat "${TORCH_LIBRARIES}")
set_property(TARGET train_bfloat PROPERTY CXX_STANDARD 17)

add_executable(train_posit ${SRC_FOLDER}/train_posit.cpp)
target_link_libraries(train_posit ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_posit "${TORCH_LIBRARIES}")
set_property(TARGET train_posit PROPERTY CXX_STANDARD 17)

add_executable(train_posit16 ${SRC_FOLDER}/train_posit16.cpp)
target_link_libraries(train_posit16 ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_posit16 "${TORCH_LIBRARIES}")
set_property(TARGET train_posit16 PROPERTY CXX_STANDARD 17)

add_executable(test_posit ${SRC_FOLDER}/test_posit.cpp)
target_link_libraries(test_posit ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_posit "${TORCH_LIBRARIES}")
set_property(TARGET test_posit PROPERTY CXX_STANDARD 17)
###########################################################################
