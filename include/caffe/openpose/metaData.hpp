#ifndef CAFFE_OPENPOSE_META_DATA_HPP
#define CAFFE_OPENPOSE_META_DATA_HPP
#ifdef USE_OPENCV

#include <string>
#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat, cv::Point, cv::Size
#include <caffe/openpose/poseModel.hpp>

namespace caffe {
    struct Joints
    {
        std::vector<cv::Point2f> points;
        std::vector<float> isVisible;
    };

    struct MetaData
    {
        bool filled;
        cv::Size imageSize;
        bool isValidation; // Just to check it is false
        int numberOtherPeople;
        int writeNumber;
        int totalWriteNumber;
        int epoch;
        cv::Point2f objPos; //objPos_x(float), objPos_y (float)
        float scaleSelf;
        Joints jointsSelf; //(3*16)
        std::vector<cv::Point2f> objPosOthers; //length is numberOtherPeople
        std::vector<float> scaleOthers; //length is numberOtherPeople
        std::vector<Joints> jointsOthers; //length is numberOtherPeople
        // Only for DomeDB
        std::string imageSource;
        bool depthEnabled = false;
        std::string depthSource;
        // Only for visualization
        std::string datasetString;
        int peopleIndex;
        int annotationListIndex;
        // For TAF
        std::vector<Joints> jointsOthersPrev;
        Joints jointsSelfPrev;
    };

    template<typename Dtype>
    bool readMetaData(
        MetaData& metaData, int& currentEpoch, std::string& datasetString,
        const char* data, const size_t offsetPerLine, const PoseCategory poseCategory,
        const PoseModel poseModel, const float cropSizeY);

}  // namespace caffe

#endif  // USE_OPENCV
#endif  // CAFFE_OPENPOSE_META_DATA_HPP
