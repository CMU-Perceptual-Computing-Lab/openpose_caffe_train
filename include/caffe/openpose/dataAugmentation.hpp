#ifndef CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP
#define CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP
#ifdef USE_OPENCV

#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat, cv::Point, cv::Size
#include "caffe/proto/caffe.pb.h"
#include "metaData.hpp"
#include "poseModel.hpp"

namespace caffe {
    // Swap center point
    void swapCenterPoint(MetaData& metaData, const OPTransformationParameter& param_, const float scale,
                         const PoseCategory poseCategory, const PoseModel poseModel);
    // Scale
    std::pair<float, float> estimateScale(
        const MetaData& metaData, const OPTransformationParameter& param_, const int index);
    // void applyScale(cv::Mat& imageAugmented, const float scale, const cv::Mat& image);
    void applyScale(MetaData& metaData, const float scale, const PoseModel poseModel);
    std::pair<float, float> estimateScale(const MetaData& metaData, const OPTransformationParameter& param_, const int datasetIndex);
    // Rotation
    std::pair<cv::Mat, cv::Size> estimateRotation(
        const MetaData& metaData, const cv::Size& imageSize, const OPTransformationParameter& param_,
        const int datasetIndex);
    std::pair<cv::Mat, cv::Size> estimateRotation(const MetaData& metaData, const cv::Size& imageSize,
                                                  const float rotation);
    float getRotRand(const OPTransformationParameter& param_);
    // void applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat, cv::Size>& RotAndFinalSize,
    //                    const cv::Mat& image, const unsigned char defaultBorderValue);
    void applyRotation(MetaData& metaData, const cv::Mat& Rot, const PoseModel poseModel);
    // Cropping
    cv::Point2i estimateCrop(const MetaData& metaData, const OPTransformationParameter& param_);
    void applyCrop(cv::Mat& imageAugmented, const cv::Point2i& cropCenter, const cv::Mat& image,
                   const unsigned char defaultBorderValue, const cv::Size& cropSize);
    void applyCrop(MetaData& metaData, const cv::Point2i& cropCenter,
                   const cv::Size& cropSize, const PoseModel poseModel);
    // Flipping
    bool estimateFlip(const float flipProb);
    void applyFlip(cv::Mat& imageAugmented, const bool flip, const cv::Mat& image);
    void applyFlip(MetaData& metaData, const bool flip, const int imageWidth,
                   const OPTransformationParameter& param_, const PoseModel poseModel);
    void rotatePoint(cv::Point2f& point2f, const cv::Mat& R);
    // Rotation + scale + cropping + flipping
    void applyAllAugmentation(cv::Mat& imageAugmented, const cv::Mat& rotationMatrix,
                              const float scale, const bool flip, const cv::Point2i& cropCenter,
                              const cv::Size& finalSize, const cv::Mat& image,
                              const unsigned char defaultBorderValue);
    // Other functions
    void keepRoiInside(cv::Rect& roi, const cv::Size& imageSize);
    void clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit);
    cv::Size estimatePO(const MetaData& metaData, const OPTransformationParameter& param_);
    cv::Point2i addPO(const MetaData& metaData, const cv::Size pointOffset);
    // Auxiliary functions
    const std::string DELIMITER = ";";
    std::vector<std::string> split(const std::string& stringToSplit, const std::string& delimiter = DELIMITER);
    template <typename Dtype>
    void splitFloating(std::vector<Dtype>& splitedText, const std::string& stringToSplit, const std::string& delimiter = DELIMITER)
    {
        splitedText.clear();
        const auto stringSplit = split(stringToSplit, delimiter);
        for (const auto& string : stringSplit)
            splitedText.emplace_back(std::stod(string));
    }
    template <typename Dtype>
    void splitUnsigned(std::vector<Dtype>& splitedText, const std::string& stringToSplit, const std::string& delimiter = DELIMITER)
    {
        splitedText.clear();
        const auto stringSplit = split(stringToSplit, delimiter);
        for (const auto& string : stringSplit)
            splitedText.emplace_back(std::stoull(string));
    }

}  // namespace caffe

#endif  // USE_OPENCV
#endif  // CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP_
