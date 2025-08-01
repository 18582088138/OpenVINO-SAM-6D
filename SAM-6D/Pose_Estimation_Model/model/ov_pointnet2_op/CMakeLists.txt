# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "openvino_operation_extension")

# The OpenVINO installed from PyPI can be used to find OpenVINO_DIR
find_package(Python3 REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
    OUTPUT_VARIABLE OpenVINO_DIR_PY
    ERROR_QUIET
)

find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")
find_package(Eigen3 REQUIRED)

set(SRC furthest_point_sampling.cpp gather_operation.cpp three_nn.cpp three_interpolate.cpp cylinder_query.cpp ball_query.cpp grouping_operation.cpp custom_svd.cpp custom_det.cpp custom_searchsorted.cpp custom_debug_node.cpp ov_extension.cpp)
# set(SRC furthest_point_sampling.cpp gather_operation.cpp three_nn.cpp three_interpolate.cpp cylinder_query.cpp ball_query.cpp grouping_operation.cpp ov_extension.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime Eigen3::Eigen)

# [cmake:extension]
install(TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

# Enable code style check
file(GLOB_RECURSE template_extension_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
# ov_add_clang_format_target(openvino_template_extension_clang FOR_SOURCES ${template_extension_src})