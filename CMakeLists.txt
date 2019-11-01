cmake_minimum_required(VERSION 2.8)

project(test)



if(UNIX AND NOT APPLE)
	# find the package OpenCV
	find_package(OpenCV 4 REQUIRED)
	#link the local lib

	#message("    OPENCV_LIB  : "      ${OpenCV_LIBS})
	#message("    OPENCV_LIB  : "      ${OpenCV_INCLUDE_DIRS})
	
	set(INCLUDE_DIR ${OpenCV_INCLUDE_DIRS})
	include_directories(test ${INCLUDE_DIR} )

	# Build the main source code
	add_executable(./test ${PROJECT_SOURCE_DIR}/main.cpp 
						  ${PROJECT_SOURCE_DIR}/yolo.cpp
						  )

	# Combine and add the library directories
	set(LIB_DIR ${OpenCV_LIBS})
	target_link_libraries(./test ${LIB_DIR} /usr/lib/x86_64-linux-gnu/libyaml-cpp.so)

	add_library(yolo
				SHARED
				${PROJECT_SOURCE_DIR}/yolo.cpp
				)
	target_link_libraries(yolo ${OpenCV_LIBS} /usr/lib/x86_64-linux-gnu/libyaml-cpp.so)

	# Check the availability of C++ compilers
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	if(COMPILER_SUPPORTS_CXX11)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	elseif(COMPILER_SUPPORTS_CXX0X)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	endif()
endif()





