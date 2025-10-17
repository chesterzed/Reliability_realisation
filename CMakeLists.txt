cmake_minimum_required(VERSION 3.21)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

# Boost
set(BOOST_ROOT "A:/Clibs/boost_1_82_0")
find_package(BOOST REQUIRED COMPONENTS math)

if(Boost_FOUND)
        message(STATUS "Boost version: ${Boost_VERSION}")
        message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
        message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
else()
        message(WARNING "Boost not found")
endif()


# Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_VERSION}")
        message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
        message(STATUS "Python libraries: ${Python3_LIBRARIES}")

        # 🔧 ПОИСК ПУТИ К NUMPY
        execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NUMPY_INCLUDE_DIR)
                message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")
        else()
                message(WARNING "NumPy include directory not found!")
        endif()

else()
        message(FATAL_ERROR "Python not found!")
endif()

# 🔧 ПРОВЕРКА MATPLOTLIB
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import matplotlib; print(matplotlib.__version__)"
        OUTPUT_VARIABLE MATPLOTLIB_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)

if(MATPLOTLIB_VERSION)
        message(STATUS "Matplotlib version: ${MATPLOTLIB_VERSION}")
else()
        message(WARNING "Matplotlib not found. Run: pip3 install matplotlib")
endif()

# Anyway
add_executable(MyProject main.cpp
        distributions/beta/beta_dist.cpp
        distributions/beta/beta_dist.h
        matplotlibcpp.h
        )

# Boost
target_include_directories(MyProject PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(MyProject ${Boost_LIBRARIES})

# Python
target_link_libraries(MyProject ${Python3_LIBRARIES})
target_include_directories(MyProject PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
)