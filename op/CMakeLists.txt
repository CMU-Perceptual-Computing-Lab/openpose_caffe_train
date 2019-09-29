#add_executable(opcaffe "opcaffe.cpp")
#target_link_libraries(opcaffe ${Caffe_LINK})


#set(PYTHON_FILES
#    openpose.py
#    __init__.py
#    openpose_python.cpp)

add_subdirectory(pybind11)
pybind11_add_module(opcaffe opcaffe.cpp test.py)
target_link_libraries(opcaffe PRIVATE pybind11::module ${Caffe_LINK})
SET_TARGET_PROPERTIES(opcaffe PROPERTIES PREFIX "")
configure_file(test.py test.py)
