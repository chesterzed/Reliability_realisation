cmake_minimum_required(VERSION 3.21)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

set(BOOST_ROOT "A:/Clibs/boost_1_82_0")
find_package(BOOST REQUIRED COMPONENTS math)

if(Boost_FOUND)
        message(STATUS "Boost version: ${Boost_VERSION}")
        message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
        message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
else()
        message(WARNING "Boost not found")
endif()


add_executable(MyProject main.cpp
        distributions/beta/beta_dist.cpp
        distributions/beta/beta_dist.h
        )


target_include_directories(MyProject PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(MyProject ${Boost_LIBRARIES})

