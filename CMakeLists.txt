cmake_minimum_required(VERSION 3.17)
project(StupidOmaha)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DNDEBUG)
endif()

# TODO: remove hard-coded paths
set(TORCH_DIR "/home/dshin/libtorch")
set(POKERSTOVE_DIR "/home/dshin/checkouts/pokerstove")

# Find the Torch package
find_package(Torch REQUIRED PATHS ${TORCH_DIR})
find_package(Boost COMPONENTS program_options filesystem REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-array-bounds ${TORCH_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")

include_directories("cpp")
include_directories(${Boost_INCLUDE_DIRS})
include_directories("${POKERSTOVE_DIR}/src/lib")

link_directories("${POKERSTOVE_DIR}/build/src/lib/pokerstove/penum")
link_directories("${POKERSTOVE_DIR}/build/src/lib/pokerstove/peval")
link_directories(${Boost_LIBRARY_DIRS})

# Add your source files
add_executable(equity-vs-random cpp/EquityVsRandom.cpp)
add_executable(sim cpp/Sim.cpp)

set(executables equity-vs-random sim)

foreach(tgt ${executables})
    # Link against the Torch libraries and pokerstove library
    target_link_libraries(${tgt} ${Boost_LIBRARIES} ${TORCH_LIBRARIES} penum peval)

    # Set the output directory
    set_target_properties(${tgt} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Set optimization level for Release builds
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-O3>)
endforeach()
