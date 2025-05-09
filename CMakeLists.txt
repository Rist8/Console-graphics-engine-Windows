cmake_minimum_required(VERSION 3.10)
project(ray_marcher_sdl)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Create common source file variable
set(SRC OpenMPSDL.cpp)

# ----------------------------
# Build target: single-threaded
# ----------------------------
add_executable(single ${SRC})
target_compile_definitions(single PRIVATE SINGLE_THREADED)
target_include_directories(single PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(single PRIVATE ${SDL2_LIBRARIES})

# ----------------------------
# Build target: multi-threaded with OpenMP
# ----------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(multi ${SRC})
    target_compile_definitions(multi PRIVATE MULTI_THREADED)
    target_include_directories(multi PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(multi PRIVATE ${SDL2_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found. 'multi' target will not be available.")
endif()
