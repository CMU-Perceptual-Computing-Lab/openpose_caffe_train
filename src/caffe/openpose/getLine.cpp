#include "caffe/openpose/getLine.hpp"

namespace caffe
{
    std::string getLine(const int line, const std::string& function, const std::string& file)
    {
        return std::string{" at " + std::to_string(line) + ", " + function + ", " + file};
    }
}
