project(ConnectZero)
cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0003 NEW)
set(Python3_FIND_UNVERSIONED_NAMES FIRST)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
message(STATUS ${Python3_NumPy_VERSION})
find_package(TensorflowCC REQUIRED)
find_package(GSL REQUIRED)
add_library(MCTS_exten SHARED cpp/exten.cpp cpp/NN.cpp cpp/custom_NN.cpp cpp/custom_position.cpp cpp/custom_game.cpp cpp/treestate.cpp cpp/search.cpp)
set_target_properties(
	MCTS_exten
	PROPERTIES
		PREFIX ""
		OUTPUT_NAME "MCTS_exten"
		LINKER_LANGUAGE C
	)
target_include_directories(MCTS_exten PRIVATE ${Python3_INCLUDE_DIRS} PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(MCTS_exten ${PYTHON_LIBRARIES} TensorflowCC::TensorflowCC GSL::gsl GSL::gslcblas Python3::NumPy)
find_package(CUDA)
if(CUDA_FOUND)
	target_link_libraries(MCTS_exten ${CUDA_LIBRARIES})
endif()
set(CMAKE_CXX_FLAGS "-O3 -flto")
