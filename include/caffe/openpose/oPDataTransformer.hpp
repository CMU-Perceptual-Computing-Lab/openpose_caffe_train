#ifndef CAFFE_OPENPOSE_OP_DATA_TRANSFORMER_HPP
#define CAFFE_OPENPOSE_OP_DATA_TRANSFORMER_HPP

// OpenPose: added
// This function has been originally copied from include/caffe/data_transformer.hpp (both hpp and cpp) at Sep 7th, 2017
// OpenPose: added end

#include <vector>
// OpenPose: added
#ifdef USE_OPENCV
    #include <opencv2/core/core.hpp> // cv::Mat, cv::Point, cv::Size
#endif  // USE_OPENCV
#include "dataAugmentation.hpp"
#include "metaData.hpp"
#include "poseModel.hpp"
// OpenPose: added end
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class OPDataTransformer {
public:
    explicit OPDataTransformer(const OPTransformationParameter& param, Phase phase,
        const std::string& modelString, const std::string& inputType = "image"); // OpenPose: Added last 2
    virtual ~OPDataTransformer() {}

    /**
     * @brief Initialize the Random number generations if needed by the
     *    transformation.
     */
    // void InitRand();

#ifdef USE_OPENCV
    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a cv::Mat
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See image_data_layer.cpp for an example.
     */
    // void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob); // OpenPose: commented
#endif  // USE_OPENCV

protected:
     /**
     * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
     *
     * @param n
     *    The upperbound (exclusive) value of the random number.
     * @return
     *    A uniformly random integer value from ({0, 1, ..., n-1}).
     */
    // virtual int Rand(int n);

    // void Transform(const Datum& datum, Dtype* transformedData); // OpenPose: commented
    // OpenPose: added
    // Image and label
public:
    void Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                   std::vector<long double>& distanceAverageNew,
                   std::vector<long double>& distanceSigmaNew,
                   std::vector<unsigned long long>& distanceAverageNewCounter,
                   const int datasetIndex,
                   const Datum* datum, const Datum* const datumNegative = nullptr);
    int getNumberChannels() const;

    // For Video
    void TransformVideoSF(int vid, int frames, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel, const Datum& datum,
                   const Datum& datumNegative, const int datasetIndex);
    void TestVideo(int frames, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel);
protected:
    // OpenPose: added end
    // Tranformation parameters
    // TransformationParameter param_; // OpenPose: commented
    OPTransformationParameter param_; // OpenPose: added

    // shared_ptr<Caffe::RNG> rng_; // OpenPose: commented
    Phase phase_;
    // Blob<Dtype> data_mean_; // OpenPose: commented
    // vector<Dtype> mean_values_; // OpenPose: commented

    // OpenPose: added
protected:
    PoseModel mPoseModel;
    PoseCategory mPoseCategory;
    int mCurrentEpoch;
    std::string mDatasetString;
    std::string mModelString;
    std::string mInputType;

    // Label generation
    void generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel, const Datum* datum,
                              const Datum* const datumNegative, const int datasetIndex,
                              std::vector<long double>& distanceAverageNew,
                              std::vector<long double>& distanceSigmaNew,
                              std::vector<unsigned long long>& distanceCounterNew);
    void generateLabelMap(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                          const MetaData& metaData, const int datasetIndex,
                          const std::vector<float>& distanceAverage,
                          const std::vector<float>& distanceSigma,
                          std::vector<long double>& distanceAverageNew,
                          std::vector<long double>& distanceSigmaNew,
                          std::vector<unsigned long long>& distanceCounterNew);
    // // For Distance
    // void putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* entryD, Dtype* entryDMask, cv::Mat& count,
    //                    const cv::Point2f& centerA, const cv::Point2f& centerB, const int stride, const int gridX,
    //                    const int gridY, const float sigma, const int thre) const;
    // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_OP_DATA_TRANSFORMER_HPP
