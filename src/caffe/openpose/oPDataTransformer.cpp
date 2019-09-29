#ifdef USE_OPENCV
    #include <opencv2/core/core.hpp>
    // OpenPose: added
    // #include <opencv2/contrib/contrib.hpp>
    // #include <opencv2/contrib/imgproc.hpp>
    // #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/opencv.hpp>
    // OpenPose: added end
#endif  // USE_OPENCV

// OpenPose: added
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
// OpenPose: added end
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
// OpenPose: added
#include "caffe/util/benchmark.hpp"
#include "caffe/openpose/getLine.hpp"
#include "caffe/openpose/oPDataTransformer.hpp"
// OpenPose: added end

namespace caffe {
// OpenPose: added ended
std::vector<unsigned long long> sNumberMaxOcclusions;
std::vector<double> sKeypointSigmas;
std::mutex sOcclusionsMutex;

struct AugmentSelection
{
    bool flip = false;
    std::pair<cv::Mat, cv::Size> RotAndFinalSize;
    cv::Point2i cropCenter;
    float scale = 1.f;
    // Video: Temp Data
    float rotation = 0.f;
    cv::Size pointOffset;
};

void doOcclusions(cv::Mat& imageAugmented, cv::Mat& backgroundImageAugmented, const MetaData& metaData,
                  const unsigned int numberMaxOcclusions, const PoseModel poseModel)
{
    // Only do occlusions if no other people (so I avoid issue of completely blocking other people)
    if (metaData.objPosOthers.empty())
    {
        // For all visible keypoints --> [0, numberMaxOcclusions] oclusions
        // For 1/n visible keypoints --> [0, numberMaxOcclusions/n] oclusions
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        const auto numberBodyParts = getNumberBodyParts(poseModel);
        int detectedParts = 0;
        for (auto i = 0 ; i < numberBodyParts ; i++)
            if (metaData.jointsSelf.isVisible[i] < 1.5f)
                detectedParts++;
        const auto numberOcclusions = (int)std::round(numberMaxOcclusions * dice * detectedParts / numberBodyParts);
        if (numberOcclusions > 0)
        {
            for (auto i = 0 ; i < numberOcclusions ; i++)
            {
                // Select occluded part
                int occludedPart = -1;
                do
                    occludedPart = std::rand() % numberBodyParts; // [0, #BP-1]
                while (metaData.jointsSelf.isVisible[occludedPart] > 1.5f);
                // Select random cropp around it
                const auto width = (int)std::round(imageAugmented.cols * metaData.scaleSelf/2
                                 * (1+(std::rand() % 1001 - 500)/1000.)); // +- [0.5-1.5] random
                const auto height = (int)std::round(imageAugmented.rows * metaData.scaleSelf/2
                                  * (1+(std::rand() % 1001 - 500)/1000.)); // +- [0.5-1.5] random
                const auto random = 1+(std::rand() % 1001 - 500)/500.; // +- [0-2] random
                // Estimate ROI rectangle to apply
                const auto point = metaData.jointsSelf.points[occludedPart];
                cv::Rect rectangle{(int)std::round(point.x - width/2*random),
                                   (int)std::round(point.y - height/2*random), width, height};
                keepRoiInside(rectangle, imageAugmented.size());
                // Apply crop
                if (rectangle.area() > 0)
                    backgroundImageAugmented(rectangle).copyTo(imageAugmented(rectangle));
            }
        }
    }
}

void setLabel(cv::Mat& image, const std::string& label, const cv::Point& org)
{
    const int fontface = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.5;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(image, org + cv::Point{0, baseline}, org + cv::Point{text.width, -text.height},
                  cv::Scalar{0,0,0}, CV_FILLED);
    cv::putText(image, label, org, fontface, scale, cv::Scalar{255,255,255}, thickness, 20);
}

template<typename Dtype>
int getType(Dtype dtype)
{
    (void)dtype;
    if (sizeof(Dtype) == sizeof(float))
        return CV_32F;
    else if (sizeof(Dtype) == sizeof(double))
        return CV_64F;
    else
    {
        throw std::runtime_error{"Only float or double" + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return CV_32F;
    }
}

template<typename Dtype>
void putGaussianMaps(Dtype* entry, Dtype* mask, const cv::Point2f& centerPoint, const int stride,
                     const int gridX, const int gridY, const float sigma, const float partRatio)
{
    // No distance
    // LOG(INFO) << "putGaussianMaps here we start for " << centerPoint.x << " " << centerPoint.y;
    const Dtype start = stride/2.f - 0.5f; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    const auto multiplier = 2.0 * sigma * sigma;
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        const Dtype y = start + gY * stride;
        const auto yMenosCenterPointSquared = (y-centerPoint.y)*(y-centerPoint.y);
        for (auto gX = 0; gX < gridX; gX++)
        {
            const Dtype x = start + gX * stride;
            const Dtype d2 = (x-centerPoint.x)*(x-centerPoint.x) + yMenosCenterPointSquared;
            const Dtype exponent = d2 / multiplier;
            //ln(100) = -ln(1%)
            if (exponent <= 4.6052)
            {
                const auto xyOffset = yOffset + gX;
                // Option a) Max
                entry[xyOffset] = std::min(Dtype(1), std::max(entry[xyOffset], std::exp(-exponent)));
                // // Option b) Average
                // entry[xyOffset] += std::exp(-exponent);
                // if (entry[xyOffset] > 1)
                //     entry[xyOffset] = 1;
                // Masks for this channel to 1
                mask[xyOffset] = Dtype(1.f/partRatio);
            }
        }
    }
}

#define sgn(x) x==0 ? 0 : x/abs(x)
template<typename Dtype>
void putDistanceMaps(Dtype* entryDistX, Dtype* entryDistY, Dtype* maskDistX, Dtype* maskDistY,
                     cv::Mat& count,
                     const cv::Point2f& rootPoint, const cv::Point2f& pointTarget, const int stride,
                     const int gridX, const int gridY, const float sigma, const float* averageUsed,
                     const float* sigmaUsed, long double* distanceAverageNew,
                     long double* distanceSigmaNew, unsigned long long& distanceCounterNew)
{
    // LOG(INFO) << "putDistanceMaps here we start for " << rootPoint.x << " " << rootPoint.y;
    const Dtype start = stride/2.f - 0.5f; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    const auto multiplier = 2.0 * sigma * sigma;
    const auto pointTargetScaledDown = 1/Dtype(stride)*pointTarget;
    // Distance average
    // Counter
    distanceCounterNew++;
    // Average & sigma
    const cv::Point2f directionNorm = pointTarget - rootPoint;
    const cv::Point2f valueNewPoint{(directionNorm.x/stride-averageUsed[0])/sigmaUsed[0],
                                    (directionNorm.y/stride-averageUsed[1])/sigmaUsed[1]};
    for (auto i = 0 ; i < 2 ; i++)
    {
        const auto valueNew = (i == 0 ? valueNewPoint.x : valueNewPoint.y);
        // Average
        distanceAverageNew[i] += valueNew;
        // Sigma
        const auto sigmaNew = (distanceAverageNew[i]/distanceCounterNew - valueNew);
        distanceSigmaNew[i] += sigmaNew*sigmaNew;
    }
    // Fill distance elements
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        const Dtype y = start + gY * stride;
        const auto yMenosCenterPointSquared = (y-rootPoint.y)*(y-rootPoint.y);
        for (auto gX = 0; gX < gridX; gX++)
        {
            const Dtype x = start + gX * stride;
            const Dtype d2 = (x-rootPoint.x)*(x-rootPoint.x) + yMenosCenterPointSquared;
            const Dtype exponent = d2 / multiplier;
            //ln(100) = -ln(1%)
            if (exponent <= 4.6052)
            {
                // Fill distance elements
                const auto xyOffset = yOffset + gX;
                const cv::Point2f directionAB = pointTargetScaledDown - cv::Point2f{(float)gX, (float)gY};
                // const cv::Point2f entryDValue{directionAB.x, directionAB.y};
                const cv::Point2f entryDValue{(directionAB.x-averageUsed[0])/sigmaUsed[0],
                                              (directionAB.y-averageUsed[1])/sigmaUsed[1]};
                auto& counter = count.at<uchar>(gY, gX);
                if (counter == 0)
                {
                    entryDistX[xyOffset] = Dtype(entryDValue.x);
                    entryDistY[xyOffset] = Dtype(entryDValue.y);
                    // Fill masks, it might solve the long-distance ones to be much less accurate
                    const auto maskBase = Dtype(0.333);
                    maskDistX[xyOffset] = maskBase;
                    maskDistY[xyOffset] = maskBase;
                }
                else
                {
                    entryDistX[xyOffset] = (entryDistX[xyOffset]*counter + Dtype(entryDValue.x)) / (counter + 1);
                    entryDistY[xyOffset] = (entryDistY[xyOffset]*counter + Dtype(entryDValue.y)) / (counter + 1);
                }
                // // Used so far, but it might provoke short distances to be good, while making long distances much
                // // worse. However the other one might harm the close ones
                // // Fill masks
                // const auto limit = Dtype(10); // Tried 10 in old code
                // const auto maskBase = Dtype(0.2);
                // // X
                // const auto oneOverAbsEntryDistX = maskBase/std::abs(entryDistX[xyOffset]);
                // maskDistX[xyOffset] = std::min(limit, oneOverAbsEntryDistX);
                // // Y
                // const auto oneOverAbsEntryDistY = maskBase/std::abs(entryDistY[xyOffset]);
                // maskDistY[xyOffset] = std::min(limit, oneOverAbsEntryDistY);

                // Avoid NaN - Clip masks
                const auto limit = Dtype(3); // 2 works, 5 explodes
                // X
                if (entryDistX[xyOffset] > limit)
                    maskDistX[xyOffset] = limit/std::abs(entryDistX[xyOffset]);
                // Y
                if (entryDistY[xyOffset] > limit)
                    maskDistY[xyOffset] = limit/std::abs(entryDistY[xyOffset]);
            }
        }
    }
}

template<typename Dtype>
void putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* maskX, Dtype* maskY,
                   cv::Mat& count, const cv::Point2f& centerA,
                   const cv::Point2f& centerB, const int stride, const int gridX,
                   const int gridY, const int threshold,
                   // const int diagonal, const float diagonalProportion,
                   Dtype* backgroundMask)
// void putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* entryD, Dtype* entryDMask,
//                    cv::Mat& count, const cv::Point2f& centerA,
//                    const cv::Point2f& centerB, const int stride, const int gridX,
//                    const int gridY, const int threshold)
{
    const auto scaleLabel = Dtype(1)/Dtype(stride);
    const auto centerALabelScale = scaleLabel * centerA;
    const auto centerBLabelScale = scaleLabel * centerB;
    cv::Point2f directionAB = centerBLabelScale - centerALabelScale;
    const auto distanceAB = std::sqrt(directionAB.x*directionAB.x + directionAB.y*directionAB.y);
    directionAB *= (Dtype(1) / distanceAB);

    // // For Distance
    // const auto dMin = Dtype(0);
    // const auto dMax = Dtype(std::sqrt(gridX*gridX + gridY*gridY));
    // const auto dRange = dMax - dMin;
    // const auto entryDValue = 2*(distanceAB - dMin)/dRange - 1; // Main range: [-1, 1],
    // -1 is 0px-distance, 1 is 368 / stride x sqrt(2) px of distance

    // If PAF is not 0 or NaN (e.g. if PAF perpendicular to image plane)
    if (!isnan(directionAB.x) && !isnan(directionAB.y))
    {
        const int minX = std::max(0,
                                  int(std::round(std::min(centerALabelScale.x, centerBLabelScale.x) - threshold)));
        const int maxX = std::min(gridX,
                                  int(std::round(std::max(centerALabelScale.x, centerBLabelScale.x) + threshold)));
        const int minY = std::max(0,
                                  int(std::round(std::min(centerALabelScale.y, centerBLabelScale.y) - threshold)));
        const int maxY = std::min(gridY,
                                  int(std::round(std::max(centerALabelScale.y, centerBLabelScale.y) + threshold)));
        // alpha*1 + (1-alpha)*realProportion
        // const auto weight = (1-diagonalProportion) + diagonalProportion * diagonal/distanceAB;
        for (auto gY = minY; gY < maxY; gY++)
        {
            const auto yOffset = gY*gridX;
            const auto gYMenosCenterALabelScale = gY - centerALabelScale.y;
            for (auto gX = minX; gX < maxX; gX++)
            {
                const auto xyOffset = yOffset + gX;
                const cv::Point2f ba{gX - centerALabelScale.x, gYMenosCenterALabelScale};
                const float distance = std::abs(ba.x*directionAB.y - ba.y*directionAB.x);
                if (distance <= threshold)
                {
                    auto& counter = count.at<uchar>(gY, gX);
                    if (counter == 0)
                    {
                        entryX[xyOffset] = directionAB.x;
                        entryY[xyOffset] = directionAB.y;
                        // Weight makes small PAFs as important as big PAFs
                        // maskX[xyOffset] *= weight;
                        // maskY[xyOffset] *= weight;
                        // // For Distance
                        // entryD[xyOffset] = entryDValue;
                        // entryDMask[xyOffset] = Dtype(1);
                        if (backgroundMask != nullptr)
                            backgroundMask[xyOffset] = Dtype(1);
                        // Masks for this channel to 1
                        maskX[xyOffset] = Dtype(1);
                        maskY[xyOffset] = Dtype(1);
                    }
                    else
                    {
                        entryX[xyOffset] = (entryX[xyOffset]*counter + directionAB.x) / (counter + 1);
                        entryY[xyOffset] = (entryY[xyOffset]*counter + directionAB.y) / (counter + 1);
                        // // For Distance
                        // entryD[xyOffset] = (entryD[xyOffset]*counter + entryDValue) / (counter + 1);
                    }
                    counter++;
                }
            }
        }
    }
}

cv::Rect getObjROI(const int stride, const std::vector<cv::Point2f> points,
                   const std::vector<float>& isVisible, const int gridX, const int gridY,
                   const float sideRatioX = 0.3f, const float sideRatioY = 0.3f)
{
    // Get valid bounding box
    auto minX = std::numeric_limits<float>::max();
    auto maxX = std::numeric_limits<float>::lowest();
    auto minY = std::numeric_limits<float>::max();
    auto maxY = std::numeric_limits<float>::lowest();
    for (auto i = 0 ; i < points.size() ; i++)
    {
        if (isVisible[i] <= 1)
        {
            // X
            if (maxX < points[i].x)
                maxX = points[i].x;
            if (minX > points[i].x)
                minX = points[i].x;
            // Y
            if (maxY < points[i].y)
                maxY = points[i].y;
            if (minY > points[i].y)
                minY = points[i].y;
        }
    }
    minX /= stride;
    maxX /= stride;
    minY /= stride;
    maxY /= stride;
    // Objet position and initial scale
    const auto objPosX = (maxX + minX) / 2;
    const auto objPosY = (maxY + minY) / 2;
    auto scaleX = maxX - minX;
    auto scaleY = maxY - minY;
    // Sometimes width is too narrow (only 1-2 keypoints in width), then we use at least half the size of the opposite
    // direction (height)
    if (scaleX < scaleY / 2)
        scaleX = scaleY / 2;
    else if (scaleY < scaleX / 2)
        scaleY = scaleX / 2;
    // Get ROI
    cv::Rect roi{
        int(std::round(objPosX - scaleX/2 - sideRatioX*scaleX)),
        int(std::round(objPosY - scaleY/2 - sideRatioY*scaleY)),
        int(std::round(scaleX*(1+2*sideRatioX))),
        int(std::round(scaleY*(1+2*sideRatioY)))
    };
    keepRoiInside(roi, cv::Size{gridX, gridY});
    // Return results
    return roi;
}

template <typename Dtype>
void maskPerson(
    Dtype* transformedLabel, const std::vector<cv::Point2f> points, const std::vector<float>& isVisible,
    const int stride, const int gridX, const int gridY, const int backgroundMaskIndex,
    const bool bodyIndexes, const PoseModel poseModel, const std::vector<int>& missingBodyPartsBase,
    const float sideRatioX = 0.3f, const float sideRatioY = 0.3f, const int tafTopology = 0)
{
    // Get missing indexes taking into account visible==3
    const auto missingIndexes = (bodyIndexes
        ? getIndexesForParts(poseModel, missingBodyPartsBase, isVisible, 4.f, tafTopology) : missingBodyPartsBase);
    // If missing indexes --> Mask out whole person
    if (!missingIndexes.empty())
    {
        // Get valid ROI bounding box
        const auto roi = getObjROI(stride, points, isVisible, gridX, gridY, sideRatioX, sideRatioY);
        // Apply ROI
        const auto channelOffset = gridX * gridY;
        const auto type = getType(Dtype(0));
        if (roi.area() > 0)
        {
            // Apply ROI to all channels with missing indexes
            for (const auto& missingIndex : missingIndexes)
            {
                cv::Mat maskMissTemp(gridY, gridX, type, &transformedLabel[missingIndex*channelOffset]);
                maskMissTemp(roi).setTo(0.f); // For debugging use 0.5f
            }
            // Apply ROI to background channel
            if (backgroundMaskIndex >= 0)
            {
                cv::Mat maskMissTemp(gridY, gridX, type, &transformedLabel[backgroundMaskIndex*channelOffset]);
                maskMissTemp(roi).setTo(0.f); // For debugging use 0.5f
            }
        }
    }
}

template <typename Dtype>
void maskPersonIfVisibleIsX(
    Dtype* transformedLabel, const std::vector<cv::Point2f> points, const std::vector<float>& isVisible,
    const int stride, const int gridX, const int gridY, const int backgroundMaskIndex,
    const PoseModel poseModel, const int X,
    const float sideRatioX = 0.3f, const float sideRatioY = 0.3f, const int tafTopology = 0)
{
    // Fake neck, mid hip - Mask out the person bounding box for those PAFs/BP where isVisible == 3
    std::vector<int> missingBodyPartsBase;
    // Get visible == 3 parts
    for (auto part = 0; part < isVisible.size(); part++)
        if (isVisible[part] == X)
            missingBodyPartsBase.emplace_back(part);
    // Make out channels
    maskPerson(
        transformedLabel, points, isVisible, stride, gridX, gridY, backgroundMaskIndex, true, poseModel,
        missingBodyPartsBase, sideRatioX, sideRatioY, tafTopology);
}
// OpenPose: added ended

template<typename Dtype>
OPDataTransformer<Dtype>::OPDataTransformer(const OPTransformationParameter& param,
        Phase phase, const std::string& modelString, const std::string& inputType) // OpenPose: Added std::string
        // : param_(param), phase_(phase) {
        : param_(param), phase_(phase), mCurrentEpoch{-1} {
    // OpenPose: commented
    // // check if we want to use mean_file
    // if (param_.has_mean_file()) {
    //     CHECK_EQ(param_.mean_value_size(), 0) <<
    //         "Cannot specify mean_file and mean_value at the same time";
    //     const std::string& mean_file = param.mean_file();
    //     if (Caffe::root_solver()) {
    //         LOG(INFO) << "Loading mean file from: " << mean_file;
    //     }
    //     BlobProto blob_proto;
    //     ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    //     data_mean_.FromProto(blob_proto);
    // }
    // // check if we want to use mean_value
    // if (param_.mean_value_size() > 0) {
    //     CHECK(param_.has_mean_file() == false) <<
    //         "Cannot specify mean_file and mean_value at the same time";
    //     for (int c = 0; c < param_.mean_value_size(); ++c) {
    //         mean_values_.push_back(param_.mean_value(c));
    //     }
    // }
    // OpenPose: commented end
    // OpenPose: added
    LOG(INFO) << "OPDataTransformer constructor done.";
    // PoseModel
    std::tie(mPoseModel, mPoseCategory) = flagsToPoseModel(modelString);
    mModelString = modelString;
    mInputType = inputType;
    // OpenPose: added end
}

// OpenPose: commented
// template <typename Dtype>
// void OPDataTransformer<Dtype>::InitRand() {
//     const bool needs_rand = param_.mirror() ||
//             (phase_ == TRAIN && param_.crop_size());
//     if (needs_rand)
//     {
//         const unsigned int rng_seed = caffe_rng_rand();
//         rng_.reset(new Caffe::RNG(rng_seed));
//     }
//     else
//         rng_.reset();
// }

// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                                         std::vector<long double>& distanceAverageNew,
                                         std::vector<long double>& distanceSigmaNew,
                                         std::vector<unsigned long long>& distanceCounterNew,
                                         const int datasetIndex,
                                         const Datum* datum,
                                         const Datum* const datumNegative)
{
    // Secuirty checks
    if (datum != nullptr)
    {
        const int datumChannels = datum->channels();
        CHECK_GE(datumChannels, 1);
    }
    const int imageNum = transformedData->num();
    const int imageChannels = transformedData->channels();
    const int labelNum = transformedLabel->num();
    CHECK_EQ(imageChannels, 3);
    CHECK_EQ(imageNum, labelNum);
    CHECK_GE(imageNum, 1);

    auto* transformedDataPtr = transformedData->mutable_cpu_data();
    auto* transformedLabelPtr = transformedLabel->mutable_cpu_data();
    CPUTimer timer;
    timer.Start();
    generateDataAndLabel(transformedDataPtr, transformedLabelPtr, datum, datumNegative, datasetIndex,
                         distanceAverageNew, distanceSigmaNew, distanceCounterNew);
    VLOG(2) << "Transform: " << timer.MicroSeconds() / 1000.0  << " ms";
}

template <typename Dtype>
int OPDataTransformer<Dtype>::getNumberChannels() const
{
    if (mInputType == "image")
    {
        // If distance
        if (param_.add_distance())
            return 2 * (getNumberBodyBkgAndPAF(mPoseModel) + getDistanceAverage(mPoseModel).size()); // For any
            // return 2 * (getNumberBodyBkgAndPAF(mPoseModel) + 2*(getNumberBodyParts(mPoseModel)-1)); // Neck-star distance
        // If no distance
        else
            return 2 * (getNumberTafChannels(param_.taf_topology())+getNumberBodyBkgAndPAF(mPoseModel));
    }
    else if(mInputType == "video")
    {
        return 2 * (getNumberTafChannels(param_.taf_topology())+getNumberBodyBkgAndPAF(mPoseModel));
    }
    else
        throw std::runtime_error("Type not defined" + getLine(__LINE__, __FUNCTION__, __FILE__));
    return -1;
}
// OpenPose: end

// OpenPose: commented
// template <typename Dtype>
// int OPDataTransformer<Dtype>::Rand(int n) {
//     CHECK(rng_);
//     CHECK_GT(n, 0);
//     caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
//     return ((*rng)() % n);
// }

// OpenPose: added
cv::Mat readImage(const std::string& data, const PoseCategory& poseCategory, const std::string& mediaDirectory,
                  const std::string& imageSource, const int datumWidth, const int datumHeight, const int datumArea)
{
    // Read image (LMDB channel 1)
    cv::Mat image;
    // DOME
    if (poseCategory == PoseCategory::DOME)
    {
        const auto imageFullPath = mediaDirectory + imageSource;
        image = cv::imread(imageFullPath, CV_LOAD_IMAGE_COLOR);
        if (image.empty())
            throw std::runtime_error{"Empty image" + imageFullPath + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }
    // COCO & MPII
    else
    {
        // // Naive copy
        // image = cv::Mat(datumHeight, datumWidth, CV_8UC3);
        // const auto initImageArea = (int)(image.rows * image.cols);
        // CHECK_EQ(initImageArea, datumArea);
        // for (auto y = 0; y < image.rows; y++)
        // {
        //     const auto yOffset = (int)(y*image.cols);
        //     for (auto x = 0; x < image.cols; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         cv::Vec3b& bgr = image.at<cv::Vec3b>(y, x);
        //         for (auto c = 0; c < 3; c++)
        //         {
        //             const auto dIndex = (int)(c*initImageArea + xyOffset);
        //             // if (hasUInt8)
        //                 bgr[c] = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
        //             // else
        //                 // bgr[c] = datum.float_data(dIndex);
        //         }
        //     }
        // }
        // // Naive copy (slightly optimized)
        // image = cv::Mat(datumHeight, datumWidth, CV_8UC3);
        // auto* uCharPtrCvMat = (unsigned char*)(image.data);
        // for (auto y = 0; y < image.rows; y++)
        // {
        //     const auto yOffset = (int)(y*image.cols);
        //     for (auto x = 0; x < image.cols; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         const auto baseIndex = 3*xyOffset;
        //         uCharPtrCvMat[baseIndex] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset]));
        //         uCharPtrCvMat[baseIndex + 1] = static_cast<Dtype>(
        //             static_cast<uint8_t>(data[xyOffset + initImageArea]));
        //         uCharPtrCvMat[baseIndex + 2] = static_cast<Dtype>(
        //             static_cast<uint8_t>(data[xyOffset + 2*initImageArea]));
        //     }
        // }
        // // Security check - Assert
        // cv::Mat image2;
        // std::swap(image, image2);
        // OpenCV wrapping --> 1.7x speed up naive image.at<cv::Vec3b>, 1.25x speed up with smart speed up
        const cv::Mat b(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[0]);
        const cv::Mat g(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[datumArea]);
        const cv::Mat r(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[2*datumArea]);
        cv::merge(std::vector<cv::Mat>{b,g,r}, image);
        // // Security checks
        // const auto initImageArea = (int)(image.rows * image.cols);
        // CHECK_EQ(initImageArea, datumArea);
        // CHECK_EQ(cv::norm(image-image2), 0);
    }
    return image;
}

cv::Mat readMaskMiss(const PoseCategory poseCategory, const PoseModel poseModel, const int initImageHeight,
                     const int initImageWidth, const int datumArea, const std::string& data, const bool readMask)
{
    // Read mask miss (LMDB channel 2)
    const cv::Mat maskMiss = (poseCategory == PoseCategory::COCO || poseCategory == PoseCategory::CAR//CAR_22, no CAR_12
        || poseCategory == PoseCategory::MPII || poseCategory == PoseCategory::PT || poseCategory == PoseCategory::HAND
        || readMask
        // COCO & Car
        // TODO: Car23 needs a mask, Car12 does not have one!!!
        ? cv::Mat(initImageHeight, initImageWidth, CV_8UC1, (unsigned char*)&data[4*datumArea])
        // DOME & MPII & Face (except face mask out)
        : cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{255.}));
    // // Naive copy
    // cv::Mat maskMiss2;
    // // COCO
    // if (poseCategory == PoseCategory::COCO)
    // {
    //     maskMiss2 = cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{0});
    //     for (auto y = 0; y < maskMiss2.rows; y++)
    //     {
    //         const auto yOffset = (int)(y*initImageWidth);
    //         for (auto x = 0; x < initImageWidth; x++)
    //         {
    //             const auto xyOffset = yOffset + x;
    //             const auto dIndex = (int)(4*datumArea + xyOffset);
    //             Dtype dElement;
    //             // if (hasUInt8)
    //                 dElement = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
    //             // else
    //                 // dElement = datum.float_data(dIndex);
    //             if (std::round(dElement/255)!=1 && std::round(dElement/255)!=0)
    //                 throw std::runtime_error{"Value out of {0,1}" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    //             maskMiss2.at<uchar>(y, x) = dElement; //round(dElement/255);
    //         }
    //     }
    // }
    // // DOME & MPII
    // else
    //     maskMiss2 = cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{255});
    // // Security checks
    // CHECK_EQ(cv::norm(maskMiss-maskMiss2), 0);
    return maskMiss;
}

cv::Mat readBackgroundImage(const Datum* datumNegative, const int finalImageWidth, const int finalImageHeight,
                            const cv::Size finalCropSize)
{
    // Read background image
    cv::Mat backgroundImage;
    if (datumNegative != nullptr)
    {
        const std::string& data = datumNegative->data();
        const int datumNegativeWidth = datumNegative->width();
        const int datumNegativeHeight = datumNegative->height();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        // Background image
        // // Naive copy
        // backgroundImage = cv::Mat(datumNegativeHeight, datumNegativeWidth, CV_8UC3);
        // for (auto y = 0; y < datumNegativeHeight; y++)
        // {
        //     const auto yOffset = (int)(y*datumNegativeWidth);
        //     for (auto x = 0; x < datumNegativeWidth; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         cv::Vec3b& bgr = backgroundImage.at<cv::Vec3b>(y, x);
        //         for (auto c = 0; c < 3; c++)
        //         {
        //             const auto dIndex = (int)(c*datumNegativeArea + xyOffset);
        //             bgr[c] = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
        //         }
        //     }
        // }
        // // Naive copy (slightly optimized)
        // backgroundImage = cv::Mat(datumNegativeHeight, datumNegativeWidth, CV_8UC3);
        // auto* uCharPtrCvMat = (unsigned char*)(backgroundImage.data);
        // for (auto y = 0; y < datumNegativeHeight; y++)
        // {
        //     const auto yOffset = (int)(y*datumNegativeWidth);
        //     for (auto x = 0; x < datumNegativeWidth; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         const auto baseIndex = 3*xyOffset;
        //         uCharPtrCvMat[baseIndex] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset]));
        //         uCharPtrCvMat[baseIndex + 1] = static_cast<Dtype>(
        //             static_cast<uint8_t>(data[xyOffset + datumNegativeArea]));
        //         uCharPtrCvMat[baseIndex + 2] = static_cast<Dtype>(
        //             static_cast<uint8_t>(data[xyOffset + 2*datumNegativeArea]));
        //     }
        // }
        // // Security check - Assert
        // cv::Mat image2;
        // std::swap(backgroundImage, image2);
        // OpenCV wrapping --> 1.7x speed up naive image.at<cv::Vec3b>, 1.25x speed up with smart speed up
        const cv::Mat b(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[0]);
        const cv::Mat g(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[datumNegativeArea]);
        const cv::Mat r(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[2*datumNegativeArea]);
        cv::merge(std::vector<cv::Mat>{b,g,r}, backgroundImage);
        // // Security checks
        // const auto datumNegativeArea2 = (int)(backgroundImage.rows * backgroundImage.cols);
        // CHECK_EQ(datumNegativeArea2, datumNegativeArea);
        // CHECK_EQ(cv::norm(backgroundImage-image2), 0);
        // Included data augmentation: cropping
        // Disable data augmentation --> minX = minY = 0
        // Data augmentation: cropping
        if (datumNegativeWidth > finalImageWidth && datumNegativeHeight > finalImageHeight)
        {
            // Option a) Rescale down
            const auto xRatio = finalImageWidth / (float) backgroundImage.cols;
            const auto yRatio = finalImageHeight / (float) backgroundImage.rows;
            if (xRatio > yRatio)
                cv::resize(backgroundImage, backgroundImage,
                           cv::Size{finalImageWidth, (int)std::round(xRatio*backgroundImage.rows)},
                           0., 0., CV_INTER_CUBIC);
            else
                cv::resize(backgroundImage, backgroundImage,
                           cv::Size{(int)std::round(yRatio*backgroundImage.cols), finalImageHeight},
                           0., 0., CV_INTER_CUBIC);
            // Option b) Random crop
            // If resize
            const auto xDiff = backgroundImage.cols - finalImageWidth;
            const auto yDiff = backgroundImage.rows - finalImageHeight;
            const auto minX = (xDiff <= 0 ? 0 :
                (int)std::round(xDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
            );
            const auto minY = (xDiff <= 0 ? 0 :
                (int)std::round(yDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
            );
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            const cv::Point2i backgroundCropCenter{minX + finalImageWidth/2, minY + finalImageHeight/2};
            applyCrop(backgroundImage, backgroundCropCenter, backgroundImageTemp, 0, finalCropSize);
        }
        // Resize (if smaller than final crop size)
        // if (datumNegativeWidth < finalImageWidth || datumNegativeHeight < finalImageHeight)
        else
        {
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            cv::resize(backgroundImageTemp, backgroundImage, cv::Size{finalImageWidth, finalImageHeight},
                       0, 0, CV_INTER_CUBIC);
        }
    }
    return backgroundImage;
}

template<typename Dtype>
bool generateAugmentedImages(MetaData& metaData, int& currentEpoch, std::string& datasetString,
                             cv::Mat& imageAugmented, cv::Mat& maskMissAugmented,
                             const Datum* datum, const Datum* const datumNegative,
                             const OPTransformationParameter& param_, const PoseCategory poseCategory,
                             const PoseModel poseModel, const Phase phase_, const int datasetIndex)
{
    // Time measurement
    CPUTimer timer1;
    timer1.Start();

    cv::Mat image;
    cv::Mat maskMiss;
    bool validMetaData = true;
    if (datum != nullptr)
    {
        // Parameters
        const std::string& data = datum->data();
        const int datumHeight = datum->height();
        const int datumWidth = datum->width();
        const auto datumArea = (int)(datumHeight * datumWidth);

        // const bool hasUInt8 = data.size() > 0;
        CHECK(data.size() > 0);

        // Read meta data (LMDB channel 3)
        // DOME
        if (poseCategory == PoseCategory::DOME)
            validMetaData = readMetaData<Dtype>(metaData, currentEpoch, datasetString, data.c_str(), datumWidth,
                                                poseCategory, poseModel, param_.crop_size_y());
        // COCO & MPII
        else
            validMetaData = readMetaData<Dtype>(metaData, currentEpoch, datasetString, &data[3 * datumArea],
                                                datumWidth, poseCategory, poseModel, param_.crop_size_y());
        // If error reading meta data --> Labels to 0 and return
        if (validMetaData)
        {
            // Read image (LMDB channel 1)
            image = readImage(data, poseCategory, param_.media_directory(), metaData.imageSource, datumWidth,
                              datumHeight, datumArea);

            // Read mask miss (LMDB channel 2)
            const auto initImageWidth = (int)image.cols;
            const auto initImageHeight = (int)image.rows;
            const auto readMask = metaData.datasetString == "face70_mask_out";
            maskMiss = readMaskMiss(
                poseCategory, poseModel, initImageHeight, initImageWidth, datumArea, data,
                readMask);
        }
        else
            LOG(INFO) << "Invalid metaData" + getLine(__LINE__, __FUNCTION__, __FILE__);
    }
    else
    {
        metaData.filled = false;
        validMetaData = false;
        if (datumNegative == nullptr)
            throw std::runtime_error{"Both datum and datumNegative are nullptr" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }

    // Parameters
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};
    const auto stride = (int)param_.stride();
    const auto gridX = finalImageWidth / stride;
    const auto gridY = finalImageHeight / stride;
    const auto initImageWidth = (int)image.cols;
    const auto initImageHeight = (int)image.rows;

    // Read background image
    const auto backgroundImage = readBackgroundImage(datumNegative, finalImageWidth, finalImageHeight, finalCropSize);

    // Time measurement
    VLOG(2) << "  bgr[:] = datum: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Depth image
    // const bool depthEnabled = metaData.depthEnabled;

    // timer1.Start();
    // // Clahe
    // if (param_.do_clahe())
    //     clahe(image, param_.clahe_tile_size(), param_.clahe_clip_limit());
    // BGR --> Gray --> BGR
    // if image is grey
    // cv::cvtColor(image, image, CV_GRAY2BGR);
    // VLOG(2) << "  cvtColor and CLAHE: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Data augmentation
    timer1.Start();
    AugmentSelection augmentSelection;
    // Augmentation
    cv::Mat backgroundImageAugmented;
    VLOG(2) << "   input size (" << initImageWidth << ", " << initImageHeight << ")";
    // We only do random transform augmentSelection augmentation when training.
    if (phase_ == TRAIN) // 80% time is spent here
    {
        // backgroundImage augmentation (no scale/rotation)
        const cv::Point2i backgroundCropCenter{backgroundImage.cols/2, backgroundImage.rows/2};
        cv::Mat backgroundImageTemp;
        // Rotation: make it visible
        applyCrop(backgroundImageTemp, backgroundCropCenter, backgroundImage, 0, finalCropSize);
        applyFlip(backgroundImageAugmented, 0.5f, backgroundImageTemp);
        // cv::Mats based on Datum
        if (datum != nullptr && validMetaData)
        {
            // Mask for background image
            // Image size, not backgroundImage
            cv::Mat maskBackgroundImage = (datumNegative != nullptr
                ? cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{0}) : cv::Mat());
            cv::Mat maskBackgroundImageAugmented;
            // Augmentation (scale, rotation, cropping, and flipping)
            // Order does matter, otherwise code will fail doing augmentation
            float scaleMultiplier;
            std::tie(augmentSelection.scale, scaleMultiplier) = estimateScale(metaData, param_, datasetIndex);
            // Swap center?
            swapCenterPoint(metaData, param_, scaleMultiplier, poseCategory, poseModel);
            // Apply scale
            applyScale(metaData, augmentSelection.scale, poseModel);
            augmentSelection.RotAndFinalSize = estimateRotation(
                metaData,
                cv::Size{(int)std::round(image.cols * augmentSelection.scale),
                         (int)std::round(image.rows * augmentSelection.scale)},
                param_, datasetIndex);
            applyRotation(metaData, augmentSelection.RotAndFinalSize.first, poseModel);
            augmentSelection.cropCenter = estimateCrop(metaData, param_);
            applyCrop(metaData, augmentSelection.cropCenter, finalCropSize, poseModel);
            augmentSelection.flip = estimateFlip(param_.flip_prob());
            applyFlip(metaData, augmentSelection.flip, finalImageHeight, param_, poseModel);
            // Aug on images - ~80% code time spent in the following `applyAllAugmentation` lines
            applyAllAugmentation(imageAugmented, augmentSelection.RotAndFinalSize.first, augmentSelection.scale,
                                 augmentSelection.flip, augmentSelection.cropCenter, finalCropSize, image,
                                 0);
            applyAllAugmentation(maskBackgroundImageAugmented, augmentSelection.RotAndFinalSize.first,
                                 augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                                 finalCropSize, maskBackgroundImage, 255);
            // MPII hands special cases (1/4)
            // For Car_v1 --> Not all cars labeled, so mask out everything but keypoints/PAFs
            if (poseModel == PoseModel::MPII_65_42 || poseModel == PoseModel::CAR_12)
                maskMissAugmented = maskBackgroundImageAugmented.clone();
            else
            {
                applyAllAugmentation(maskMissAugmented, augmentSelection.RotAndFinalSize.first,
                                     augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                                     finalCropSize, maskMiss,
                                     // Either show bkg image or mask out black region
                                     (backgroundImage.empty() ? 0 : 255));
            }
            // Lock
            std::unique_lock<std::mutex> lock{sOcclusionsMutex};
            // Get value
            const auto numberMaxOcclusions = sNumberMaxOcclusions[datasetIndex];
            lock.unlock();
            // Introduce occlusions
            doOcclusions(imageAugmented, backgroundImageAugmented, metaData, numberMaxOcclusions, poseModel);
            // Resize mask
            if (!maskMissAugmented.empty())
                cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
            // Final background image - elementwise multiplication
            if (!backgroundImageAugmented.empty() && !maskBackgroundImageAugmented.empty())
            {
                // Apply mask to background image
                cv::Mat backgroundImageAugmentedTemp;
                backgroundImageAugmented.copyTo(backgroundImageAugmentedTemp, maskBackgroundImageAugmented);
                // Add background image to image augmented
                cv::Mat imageAugmentedTemp;
                addWeighted(imageAugmented, 1., backgroundImageAugmentedTemp, 1., 0., imageAugmentedTemp);
                imageAugmented = imageAugmentedTemp;
            }
        }
        else
        {
            imageAugmented = backgroundImageAugmented;
            maskMissAugmented = cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{255});
        }
    }
    // Test
    else
    {
        imageAugmented = image;
        maskMissAugmented = maskMiss;
        // Resize mask
        if (!maskMissAugmented.empty())
            cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
    }
    // Augmentation time
    VLOG(2) << "  Aug: " << timer1.MicroSeconds()*1e-3 << " ms";

    return validMetaData;
}

template<typename Dtype>
void fillTransformedData(Dtype* transformedData, const cv::Mat& imageAugmented,
                         const OPTransformationParameter& param_)
{
    // Data copy
    // Copy imageAugmented into transformedData + mean-subtraction
    const int imageAugmentedArea = imageAugmented.rows * imageAugmented.cols;
    auto* uCharPtrCvMat = (unsigned char*)(imageAugmented.data);
    // VGG: x/256 - 0.5
    if (param_.normalization() == 0)
    {
        for (auto y = 0; y < imageAugmented.rows; y++)
        {
            const auto yOffset = y*imageAugmented.cols;
            for (auto x = 0; x < imageAugmented.cols; x++)
            {
                const auto xyOffset = yOffset + x;
                // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
                auto* bgr = &uCharPtrCvMat[3*xyOffset];
                transformedData[xyOffset] = (bgr[0] - 128) / 256.0;
                transformedData[xyOffset + imageAugmentedArea] = (bgr[1] - 128) / 256.0;
                transformedData[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 128) / 256.0;
            }
        }
    }
    // ResNet: x - channel average
    else if (param_.normalization() == 1)
    {
        for (auto y = 0; y < imageAugmented.rows ; y++)
        {
            const auto yOffset = y*imageAugmented.cols;
            for (auto x = 0; x < imageAugmented.cols ; x++)
            {
                const auto xyOffset = yOffset + x;
                // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
                auto* bgr = &uCharPtrCvMat[3*xyOffset];
                transformedData[xyOffset] = bgr[0] - 102.9801;
                transformedData[xyOffset + imageAugmentedArea] = bgr[1] - 115.9465;
                transformedData[xyOffset + 2*imageAugmentedArea] = bgr[2] - 122.7717;
            }
        }
    }
    // DenseNet: [x - channel average] * 0.017
    // https://github.com/shicai/DenseNet-Caffe#notes
    else if (param_.normalization() == 2)
    {
        const auto scaleDenseNet = 0.017;
        for (auto y = 0; y < imageAugmented.rows ; y++)
        {
            const auto yOffset = y*imageAugmented.cols;
            for (auto x = 0; x < imageAugmented.cols ; x++)
            {
                const auto xyOffset = yOffset + x;
                // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
                auto* bgr = &uCharPtrCvMat[3*xyOffset];
                transformedData[xyOffset] = (bgr[0] - 103.94)*scaleDenseNet;
                transformedData[xyOffset + imageAugmentedArea] = (bgr[1] - 116.78)*scaleDenseNet;
                transformedData[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 123.68)*scaleDenseNet;
            }
        }
    }
    // Unknown
    else
        throw std::runtime_error{"Unknown normalization" + getLine(__LINE__, __FUNCTION__, __FILE__)};
}

void putTextOnCvMat(cv::Mat& cvMat, const std::string& textToDisplay, const cv::Size& position,
                    const cv::Scalar& color, const bool normalizeWidth, const int imageWidth)
{
    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const auto ratio = imageWidth/1280.;
    // const auto fontScale = 0.75;
    const auto fontScale = 0.8 * ratio;
    const auto fontThickness = std::max(1, (int)std::round(2*ratio));
    const auto shadowOffset = std::max(1, (int)std::round(2*ratio));
    int baseline = 0;
    const auto textSize = cv::getTextSize(textToDisplay, font, fontScale, fontThickness, &baseline);
    const cv::Size finalPosition{position.width - (normalizeWidth ? textSize.width : 0),
                                 position.height + textSize.height/2};
    cv::putText(cvMat, textToDisplay,
                cv::Size{finalPosition.width + shadowOffset, finalPosition.height + shadowOffset},
                font, fontScale, cv::Scalar{0,0,0}, fontThickness);
    cv::putText(cvMat, textToDisplay, finalPosition, font, fontScale, color, fontThickness);
}

std::atomic<int> sCounterAuxiliary{0};
template<typename Dtype>
void visualize(
    const Dtype* const transformedLabel, const PoseModel poseModel, const PoseCategory poseCategory,
    const MetaData& metaData, const cv::Mat& imageAugmented, const int stride, const std::string& modelString,
    const bool addDistance, const int tafTopology = 0, const std::string folderName = "visualize")
{
    // Remove/re-create folder
    if (metaData.writeNumber < 1 && sCounterAuxiliary < 1)
    {
        // auto resultCode = std::system(std::string("rm -rf " + folderName).c_str()); // This would remove it every frame!
        const auto resultCode = std::system(std::string("mkdir " + folderName).c_str());
        (void)resultCode;
    }

    // Debugging - Visualize - Write on disk
    // if (poseCategory == PoseCategory::COCO)
    // if (poseCategory == PoseCategory::MPII)
    // if (poseCategory == PoseCategory::FACE)
    // if (poseCategory == PoseCategory::HAND)
    // if (poseCategory == PoseCategory::DOME)
    // if (poseCategory != PoseCategory::COCO)
    // if (poseCategory == PoseCategory::DOME || poseCategory == PoseCategory::FACE)
    // if (poseCategory == PoseCategory::DOME || poseCategory == PoseCategory::HAND)
    // if (false)
    {
        if (metaData.writeNumber < 1 && sCounterAuxiliary < 1)
        // if (metaData.writeNumber < 2 && sCounterAuxiliary < 2)
        // if (metaData.writeNumber < 3 && sCounterAuxiliary < 3)
        // if (metaData.writeNumber < 5 && sCounterAuxiliary < 5)
        // if (metaData.writeNumber < 10 && sCounterAuxiliary < 10)
        // if (metaData.writeNumber < 100 && sCounterAuxiliary < 100)
        {
            // 1. Create `visualize` folder in training folder (where train_pose.sh is located)
            // 2. Comment the following if statement
            const auto rezX = (int)imageAugmented.cols;
            const auto rezY = (int)imageAugmented.rows;
            const auto gridX = rezX / stride;
            const auto gridY = rezY / stride;
            const auto channelOffset = gridY * gridX;
            const auto numberTotalChannels = getNumberTafChannels(tafTopology) + getNumberBodyBkgAndPAF(poseModel)
                                           + addDistance * getDistanceAverage(poseModel).size();
            const auto bkgChannel = getNumberBodyBkgAndPAF(poseModel) - 1;
            (void)bkgChannel; // In case I do not use it inside the for loop
            (void)poseCategory; // In case I do not use it inside the for loop
            for (auto part = 0; part < numberTotalChannels; part++)
            {
                // Reduce #images saved (ideally mask images should be the same)
                // if (part < 1)
                // if (part==0 || part > 25)
                // if (part==335 || part==355) // Left/Right hand keypoint
                // if (part==435) // Face keypoint
                // if (part==335 || part==355 || part==435) // 1 left/right hand & face keypoint
                // if (part >= 436) // Face pupils
                // if (part==bkgChannel) // Background channel
                // if (part==bkgChannel || (part >= bkgChannel && metaData.writeNumber < 3)) // Bkg channel + even dist
                // if (part==bkgChannel || metaData.writeNumber < 3) // Bkg channel (for lots of images) + All channels (for few images)
                // if (part==bkgChannel || (part >= bkgChannel && part % 2 == 0)) // Bkg channel + distance
                // const auto numberPafChannels = getNumberPafChannels(poseModel); // 2 x #PAF
                // if (part < numberPafChannels || part == numberTotalChannels-1)
                // if (part < 3 || part >= numberTotalChannels - 3)
                {
                    // Mix X and Y channels
                    if (part % 2 == 1 && part < (getNumberTafChannels(tafTopology) + getNumberPafChannels(poseModel)))
                        continue;
                    // Fill output image
                    cv::Mat finalImage = cv::Mat::zeros(gridY, 2*gridX, CV_8UC1);
                    for (auto subPart = 0; subPart < 2; subPart++)
                    {
                        cv::Mat labelMap = finalImage(cv::Rect{subPart*gridX, 0, gridX, gridY});
                        for (auto gY = 0; gY < gridY; gY++)
                        {
                            const auto yOffset = gY*gridX;
                            for (auto gX = 0; gX < gridX; gX++)
                            {
                                const auto MAX_VALUE_VISUALIZE = 240; // So I can visualize when mask > 1
                                // Mix X and Y channels
                                if (part < (getNumberTafChannels(tafTopology) + getNumberPafChannels(poseModel)))
                                {
                                    const auto channelIndex1 = (part+numberTotalChannels*subPart)*channelOffset;
                                    const auto value1 = std::abs(
                                        transformedLabel[channelIndex1 + yOffset + gX]);
                                    const auto channelIndex2 = (part+1+numberTotalChannels*subPart)*channelOffset;
                                    const auto value2 = std::abs(
                                        transformedLabel[channelIndex2 + yOffset + gX]);
                                    labelMap.at<uchar>(gY,gX) = std::min(255,
                                        (int)(MAX_VALUE_VISUALIZE*(value1*value1+value2*value2)));
                                }
                                // Body part
                                else
                                {
                                    const auto channelIndex = (part+numberTotalChannels*subPart)*channelOffset;
                                    labelMap.at<uchar>(gY,gX) = std::min(255, (int)(MAX_VALUE_VISUALIZE*std::abs(
                                        transformedLabel[channelIndex + yOffset + gX])));
                                }
                            }
                        }
                    }
                    cv::resize(finalImage, finalImage, cv::Size{}, stride, stride, cv::INTER_LINEAR);
                    cv::applyColorMap(finalImage, finalImage, cv::COLORMAP_JET);
                    for (auto subPart = 0; subPart < 2; subPart++)
                    {
                        cv::Mat labelMap = finalImage(cv::Rect{subPart*rezX, 0, rezX, rezY});
                        cv::addWeighted(labelMap, 0.5, imageAugmented, 0.5, 0.0, labelMap);
                    }
                    // Add body part / PAF name to image
                    std::string textToDisplay = "textToDisplay";

                    if (tafTopology != 0)
                    {
                        if (part >= 0 && part < getNumberTafChannels(tafTopology))
                        {
                            textToDisplay = "TAF: ";
                            textToDisplay += getMapping(poseModel).at(getTafIndexA(tafTopology).at(part/2))
                                          + "->" + getMapping(poseModel).at(getTafIndexB(tafTopology).at(part/2));
                        }
                        else if (part >= getNumberTafChannels(tafTopology) && part < getNumberTafChannels(tafTopology)+getNumberPafChannels(poseModel)){
                            auto pafPart = part-getNumberTafChannels(tafTopology);
                            textToDisplay = getMapping(poseModel).at(getPafIndexA(poseModel).at(pafPart/2))
                                          + "->" + getMapping(poseModel).at(getPafIndexB(poseModel).at(pafPart/2));
                                          // + (part%2 == 0 ? " (X)" : " (Y)");

                        }
                        else
                        {
                            auto hmPart = part-getNumberPafChannels(poseModel)-getNumberTafChannels(tafTopology);
                            textToDisplay = getMapping(poseModel).at(hmPart);
                        }
                    }
                    else
                    {
                        if (part < getNumberPafChannels(poseModel))
                            textToDisplay = getMapping(poseModel).at(getPafIndexA(poseModel).at(part/2))
                                          + "->" + getMapping(poseModel).at(getPafIndexB(poseModel).at(part/2));
                                          // + (part%2 == 0 ? " (X)" : " (Y)");
                        else
                            textToDisplay = getMapping(poseModel).at(part-getNumberPafChannels(poseModel));
                    }
                    putTextOnCvMat(finalImage, textToDisplay, cv::Size{20,20}, cv::Scalar{255,255,255}, false, 3*368);
                    // Write on disk
                    const std::string randomC{char('a' + std::rand() % 26)};
                    char imagename [100];
                    if (metaData.filled)
                        sprintf(imagename, "%s/%s_augment_%04d_label_part_%02d%s.jpg",
                                folderName.c_str(), modelString.c_str(), metaData.writeNumber, part, randomC.c_str());
                    else
                        sprintf(imagename, "%s/%s_augment_%04d_negative_label_part_%02d%s.jpg",
                                folderName.c_str(), modelString.c_str(), sCounterAuxiliary.load(), part, randomC.c_str());
                    cv::imwrite(imagename, finalImage);
                }
            }
            if (!metaData.filled)
                sCounterAuxiliary++;
        }
    }
}

// OpenPose: added
template<typename Dtype>
void matToCaffe(Dtype* caffeImg, const cv::Mat& imgAug){
    const int imageAugmentedArea = imgAug.rows * imgAug.cols;
    auto* uCharPtrCvMat = (unsigned char*)(imgAug.data);
    //caffeImg = new Dtype[imgAug.channels()*imgAug.size().width*imgAug.size().height];
    for (auto y = 0; y < imgAug.rows; y++)
    {
        const auto yOffset = y*imgAug.cols;
        for (auto x = 0; x < imgAug.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
            auto* bgr = &uCharPtrCvMat[3*xyOffset];
            caffeImg[xyOffset] = (bgr[0] - 128) / 256.0;
            caffeImg[xyOffset + imageAugmentedArea] = (bgr[1] - 128) / 256.0;
            caffeImg[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 128) / 256.0;
        }
    }
}

template<typename Dtype>
void caffeToMat(cv::Mat& img, const Dtype* caffeImg, cv::Size imageSize){
    // Need a function to convert back
    img = cv::Mat(imageSize, CV_8UC3);
    const int imageAugmentedArea = img.rows * img.cols;
    auto* imgPtr = (unsigned char*)(img.data);
    for (auto y = 0; y < img.rows; y++)
    {
        const auto yOffset = y*img.cols;
        for (auto x = 0; x < img.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            auto* bgr = &imgPtr[3*xyOffset];
            bgr[0] = (caffeImg[xyOffset]*256.) + 128;
            bgr[1] = (caffeImg[xyOffset + imageAugmentedArea]*256.) + 128;
            bgr[2] = (caffeImg[xyOffset + 2*imageAugmentedArea]*256.) + 128;
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::TransformVideoSF(int vid, int frames, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                                                  const Datum& datum, const Datum& datumNegativeO, const int datasetIndex)
{
    // Parameters
    const Datum* datumNegative = &datumNegativeO;
    const std::string& data = datum.data();
    const int datumHeight = datum.height();
    const int datumWidth = datum.width();
    const auto datumArea = (int)(datumHeight * datumWidth);
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};
    const auto stride = (int)param_.stride();
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();
    const auto gridX = finalImageWidth / stride;
    const auto gridY = finalImageHeight / stride;

    // Lock
    std::unique_lock<std::mutex> lock{sOcclusionsMutex};
    // First time
    if (sNumberMaxOcclusions.empty())
    {
        splitUnsigned(sNumberMaxOcclusions, param_.number_max_occlusions(), DELIMITER);
        splitFloating(sKeypointSigmas, param_.sigmas(), DELIMITER);
    }
    // Dynamic resize
    if (sNumberMaxOcclusions.size() <= datasetIndex)
        sNumberMaxOcclusions.resize(datasetIndex+1, sNumberMaxOcclusions[0]);
    if (sKeypointSigmas.size() <= datasetIndex)
        sKeypointSigmas.resize(datasetIndex+1, sKeypointSigmas[0]);
    lock.unlock();

    // Dome doesnt work now

    // Read meta data (LMDB channel 3)
    MetaData metaData;
    bool validMetaData = readMetaData<Dtype>(metaData, mCurrentEpoch, mDatasetString, &data[3 * datumArea], datumWidth,
                                        mPoseCategory, mPoseModel, param_.crop_size_y());
    if(!validMetaData) throw std::runtime_error("Invalid Metadata");

    // Image
    cv::Mat image;
    const cv::Mat b(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[0]);
    const cv::Mat g(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[datumArea]);
    const cv::Mat r(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[2*datumArea]);
    std::vector<cv::Mat> bgr = {b,g,r};
    cv::merge(bgr, image);
    const auto initImageWidth = (int)image.cols;
    const auto initImageHeight = (int)image.rows;

    // Read background image
    cv::Mat backgroundImage;
    cv::Mat maskBackgroundImage = (datumNegative != nullptr
            ? cv::Mat(image.size().height, image.size().width, CV_8UC1, cv::Scalar{0}) : cv::Mat());
    if (datumNegative != nullptr)
    {
        const std::string& data = datumNegative->data();
        const int datumNegativeWidth = datumNegative->width();
        const int datumNegativeHeight = datumNegative->height();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        // OpenCV wrapping --> 1.7x speed up naive image.at<cv::Vec3b>, 1.25x speed up with smart speed up
        const cv::Mat b(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[0]);
        const cv::Mat g(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[datumNegativeArea]);
        const cv::Mat r(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[2*datumNegativeArea]);
        std::vector<cv::Mat> bgr = {b,g,r};
        cv::merge(bgr, backgroundImage);
        // // Security checks
        // const auto datumNegativeArea2 = (int)(backgroundImage.rows * backgroundImage.cols);
        // CHECK_EQ(datumNegativeArea2, datumNegativeArea);
        // CHECK_EQ(cv::norm(backgroundImage-image2), 0);
        // Included data augmentation: cropping
        // Disable data augmentation --> minX = minY = 0
        // Data augmentation: cropping
        if (datumNegativeWidth > finalImageWidth && datumNegativeHeight > finalImageHeight)
        {
            const auto xDiff = datumNegativeWidth - finalImageWidth;
            const auto yDiff = datumNegativeHeight - finalImageHeight;
            const auto minX = (xDiff <= 0 ? 0 :
                                            (int)std::round(xDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            const auto minY = (xDiff <= 0 ? 0 :
                                            (int)std::round(yDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            const cv::Point2i backgroundCropCenter{minX + finalImageWidth/2, minY + finalImageHeight/2};
            applyCrop(backgroundImage, backgroundCropCenter, backgroundImageTemp, 0, finalCropSize);
        }
        // Resize (if smaller than final crop size)
        // if (datumNegativeWidth < finalImageWidth || datumNegativeHeight < finalImageHeight)
        else
        {
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            cv::resize(backgroundImageTemp, backgroundImage, cv::Size{finalImageWidth, finalImageHeight}, 0, 0, CV_INTER_CUBIC);
        }
    }

    // Mask
    cv::Mat maskMiss = cv::Mat(initImageHeight, initImageWidth, CV_8UC1, (unsigned char*)&data[4*datumArea]);

    // Start Aug
    // metaData.objPos = cv::Point(image.size().width/2, image.size().height/2
    AugmentSelection startAug, endAug;
    startAug.scale = estimateScale(metaData, param_, datasetIndex).first;
    startAug.rotation = getRotRand(param_);
    startAug.pointOffset = estimatePO(metaData, param_);
    endAug.scale = estimateScale(metaData, param_, datasetIndex).first;
    endAug.rotation = getRotRand(param_);
    endAug.pointOffset = estimatePO(metaData, param_);
    bool to_flip = estimateFlip(param_.flip_prob());

    // Synthetic Motion
    bool motion = true;
    std::vector<AugmentSelection> augVec(frames);
    MetaData metaDataPrev;
    for(int i=0; i<frames; i++){
        float scale;
        float rotation;
        cv::Size ci;
        if(motion){
            scale = startAug.scale + (((endAug.scale - startAug.scale) / frames))*i;
            rotation = startAug.rotation + (((endAug.rotation - startAug.rotation) / frames))*i;
            ci = cv::Size(startAug.pointOffset.width + (((endAug.pointOffset.width - startAug.pointOffset.width) / frames))*i,
                                         startAug.pointOffset.height + (((endAug.pointOffset.height - startAug.pointOffset.height) / frames))*i);
        }else{
            scale = startAug.scale;
            rotation = startAug.rotation;
            ci = cv::Size(startAug.pointOffset.width,startAug.pointOffset.height);
        }
        MetaData metaDataCopy = metaData;

        // Augment
        cv::Mat& img = image;
        const cv::Mat& mask = maskMiss;
        AugmentSelection augmentSelection;
        // Augment here
        cv::Mat imgAug, maskAug, maskBgAug, bgImgAug;
        augmentSelection.scale = scale;
        applyScale(metaDataCopy, augmentSelection.scale, mPoseModel);
        augmentSelection.RotAndFinalSize = estimateRotation(
                    metaDataCopy,
                    cv::Size{(int)std::round(metaDataCopy.imageSize.width * startAug.scale),
                             (int)std::round(metaDataCopy.imageSize.height * startAug.scale)},
                    rotation);
        applyRotation(metaDataCopy, augmentSelection.RotAndFinalSize.first, mPoseModel);
        augmentSelection.cropCenter = addPO(metaDataCopy, ci);
        applyCrop(metaDataCopy, augmentSelection.cropCenter, finalCropSize, mPoseModel);
        //if(i==0) augmentSelection.flip = estimateFlip(metaData, param_);
        augmentSelection.flip = to_flip;
        applyFlip(metaDataCopy, augmentSelection.flip, finalImageHeight, param_, mPoseModel);
        applyAllAugmentation(imgAug, augmentSelection.RotAndFinalSize.first, augmentSelection.scale,
                             augmentSelection.flip, augmentSelection.cropCenter, finalCropSize, img, 0);
        applyAllAugmentation(maskAug, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, mask, 255);
        applyAllAugmentation(maskBgAug, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, maskBackgroundImage, 255);
        const cv::Point2i backgroundCropCenter{backgroundImage.cols/2, backgroundImage.rows/2};
        cv::Mat backgroundImageTemp;
        applyCrop(backgroundImageTemp, backgroundCropCenter, backgroundImage, 0, finalCropSize);
        applyFlip(bgImgAug, augmentSelection.flip, backgroundImageTemp);
        // Resize mask
        if (!maskAug.empty()){
            cv::Mat maskAugTemp;
            cv::resize(maskAug, maskAugTemp, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
            maskAug = maskAugTemp;
        }
        // Final background image - elementwise multiplication
        if (!bgImgAug.empty() && !maskBgAug.empty())
        {
            // Apply mask to background image
            cv::Mat backgroundImageAugmentedTemp;
            bgImgAug.copyTo(backgroundImageAugmentedTemp, maskBgAug);
            // Add background image to image augmented
            cv::Mat imageAugmentedTemp;
            cv::addWeighted(imgAug, 1., backgroundImageAugmentedTemp, 1., 0., imageAugmentedTemp);
            imgAug = imageAugmentedTemp;
        }

        // Save Prev
        if(i != 0){
            metaDataCopy.jointsOthersPrev = metaDataPrev.jointsOthers;
            metaDataCopy.jointsSelfPrev = metaDataPrev.jointsSelf;
        }
        metaDataPrev = metaDataCopy;

        // Create Label for frame
        Dtype* labelmapTemp = new Dtype[getNumberChannels() * gridY * gridX];
        const std::vector<float> a;
        const std::vector<float> b;
        std::vector<long double> c;
        std::vector<long double> d;
        std::vector<unsigned long long> e;
        generateLabelMap(labelmapTemp, imgAug.size(), maskAug, metaDataCopy, datasetIndex,
                         a, b, c, d, e);


//        if(i == 0 && vid == 4){
//            visualize(labelmapTemp, mPoseModel, mPoseCategory, metaDataCopy, imgAug, stride, mModelString, param_.add_distance(), param_.taf_topology(), "v0");
//        }
//        if(i==1 && vid == 4){
//            visualize(labelmapTemp, mPoseModel, mPoseCategory, metaDataCopy, imgAug, stride, mModelString, param_.add_distance(), param_.taf_topology(), "v1");
//            std::cout << "Done" << std::endl;
//            exit(-1);
//        }


        // Convert image to Caffe Format
        Dtype* imgaugTemp = new Dtype[imgAug.channels()*imgAug.size().width*imgAug.size().height];
        matToCaffe(imgaugTemp, imgAug);

        // Get pointers for all
        int dataOffset = imgAug.channels()*imgAug.size().width*imgAug.size().height;
        int labelOffset = getNumberChannels() * gridY * gridX;
        Dtype* transformedDataPtr = transformedData->mutable_cpu_data();
        Dtype* transformedLabelPtr = transformedLabel->mutable_cpu_data(); // Max 6,703,488

        // Copy label
        int totalVid = transformedLabel->shape()[0]/frames;
        std::copy(labelmapTemp, labelmapTemp + labelOffset, transformedLabelPtr + (i*totalVid*labelOffset + vid*labelOffset));
        delete labelmapTemp;

        // Copy data
        std::copy(imgaugTemp, imgaugTemp + dataOffset, transformedDataPtr + (i*totalVid*dataOffset + vid*dataOffset));
        delete imgaugTemp;
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::TestVideo(int frames, Blob<Dtype> *transformedData, Blob<Dtype> *transformedLabel)
{
    int totalVid = transformedLabel->shape()[0]/frames;
    int dataOffset = transformedData->shape()[1]*transformedData->shape()[2]*transformedData->shape()[3];
    int labelOffset = transformedLabel->shape()[1]*transformedLabel->shape()[2]*transformedLabel->shape()[3];
    Dtype* transformedDataPtr = transformedData->mutable_cpu_data();
    Dtype* transformedLabelPtr = transformedLabel->mutable_cpu_data(); // Max 6,703,488

    // Test Data
    for(int fid=0; fid<frames; fid++){
        for(int vid=0; vid<totalVid; vid++){
            Dtype* imgPtr = transformedDataPtr + totalVid*fid*dataOffset + vid*dataOffset;
            cv::Mat testImg;
            caffeToMat(testImg, imgPtr, cv::Size(transformedData->shape()[3], transformedData->shape()[2]));
            //int labelFrame = 2 * getNumberBodyBkgAndPAF(mPoseModel) - 1;


            // Channel Wanted
            int channelWanted = getNumberChannels()/2 + 2;
            //int channelWanted = getNumberChannels()/2 + getNumberTafChannels(param_.taf_topology()) + getNumberPafChannels(mPoseModel) + 18;
            Dtype* labelPtr = transformedLabelPtr + totalVid*fid*labelOffset + vid*labelOffset + (channelWanted)*transformedLabel->shape()[2]*transformedLabel->shape()[3];
            cv::Mat labelMat(cv::Size(transformedLabel->shape()[3], transformedLabel->shape()[2]), CV_32FC1);
            std::copy(labelPtr, labelPtr + labelMat.size().width*labelMat.size().height, &labelMat.at<float>(0,0));
            cv::resize(labelMat, labelMat, cv::Size(labelMat.size().width*8,labelMat.size().height*8));
            labelMat = cv::abs(labelMat);
            labelMat*=255;
            labelMat.convertTo(labelMat, CV_8UC1);
            cv::cvtColor(labelMat, labelMat, cv::COLOR_GRAY2BGR);
            testImg = testImg*0.2 + labelMat*0.8;

            cv::imwrite("/home/raaj/visualize/"+std::to_string(vid)+"-"+std::to_string(fid)+".png",testImg);
        }
    }

    std::cout << "TestVideo" << std::endl;
    exit(-1);
}


template<typename Dtype>
void OPDataTransformer<Dtype>::generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel,
                                                    const Datum* datum, const Datum* const datumNegative,
                                                    const int datasetIndex,
                                                    std::vector<long double>& distanceAverageNew,
                                                    std::vector<long double>& distanceSigmaNew,
                                                    std::vector<unsigned long long>& distanceCounterNew)
{
    // Parameters
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};
    const auto stride = (int)param_.stride();
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();
    const auto gridX = finalImageWidth / stride;
    const auto gridY = finalImageHeight / stride;
    // Lock
    std::unique_lock<std::mutex> lock{sOcclusionsMutex};
    // First time
    if (sNumberMaxOcclusions.empty())
    {
        splitUnsigned(sNumberMaxOcclusions, param_.number_max_occlusions(), DELIMITER);
        splitFloating(sKeypointSigmas, param_.sigmas(), DELIMITER);
    }
    // Dynamic resize
    if (sNumberMaxOcclusions.size() <= datasetIndex)
        sNumberMaxOcclusions.resize(datasetIndex+1, sNumberMaxOcclusions[0]);
    if (sKeypointSigmas.size() <= datasetIndex)
        sKeypointSigmas.resize(datasetIndex+1, sKeypointSigmas[0]);
    lock.unlock();

    MetaData metaData;
    cv::Mat imageAugmented;
    cv::Mat maskMissAugmented;
    const auto validMetaData = generateAugmentedImages<Dtype>(
        metaData, mCurrentEpoch, mDatasetString, imageAugmented, maskMissAugmented,
        datum, datumNegative, param_, mPoseCategory, mPoseModel, phase_, datasetIndex);
    // If error reading meta data --> Labels to 0 and return
    if (!validMetaData && datumNegative == nullptr)
    {
        const auto channelOffset = gridY * gridX;
        const auto addDistance = param_.add_distance();
        const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel)
                                       + addDistance * getDistanceAverage(mPoseModel).size();
        std::fill(transformedLabel, transformedLabel + 2*numberTotalChannels * channelOffset, 0.f);
        LOG(INFO) << "Invalid meta data, returned empty" + getLine(__LINE__, __FUNCTION__, __FILE__);
        return;
    }

    // Time measurement
    CPUTimer timer1;
    timer1.Start();

    // Fill transformerData
    fillTransformedData(transformedData, imageAugmented, param_);

    // Generate and copy label
    const auto& distanceAverage = getDistanceAverage(mPoseModel);
    const auto& sigmaAverage = getDistanceSigma(mPoseModel);
    generateLabelMap(transformedLabel, imageAugmented.size(), maskMissAugmented, metaData, datasetIndex,
                     distanceAverage, sigmaAverage, distanceAverageNew, distanceSigmaNew, distanceCounterNew);
    VLOG(2) << "  AddGaussian+CreateLabel: " << timer1.MicroSeconds()*1e-3 << " ms";

    // // Debugging - Visualize - Write on disk
    // visualize(
    //     transformedLabel, mPoseModel, mPoseCategory, metaData, imageAugmented, stride, mModelString,
    //     param_.add_distance());
}

float getNorm(const cv::Point2f& pointA, const cv::Point2f& pointB)
{
    const auto difference = pointA - pointB;
    return std::sqrt(difference.x*difference.x + difference.y*difference.y);
}

void maskHands(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points,
              const float stride, const float ratio)
{
    for (auto part = 0 ; part < 2 ; part++)
    {
        const auto shoulderIndex = (part == 0 ? 5:2);
        const auto elbowIndex = shoulderIndex+1;
        const auto wristIndex = elbowIndex+1;
        if (isVisible.at(shoulderIndex) != 2 && isVisible.at(elbowIndex) != 2 && isVisible.at(wristIndex) != 2)
        {
            const auto ratioStride = 1.f / stride;
            const auto wrist = ratioStride * points.at(wristIndex);
            const auto elbow = ratioStride * points.at(elbowIndex);
            const auto shoulder = ratioStride * points.at(shoulderIndex);

            const auto distance = (int)std::round(ratio*std::max(getNorm(wrist, elbow), getNorm(elbow, shoulder)));
            const cv::Point momentum = (wrist-elbow)*0.25f;
            cv::Rect roi{(int)std::round(wrist.x + momentum.x - distance /*- wrist.x/2.f*/),
                         (int)std::round(wrist.y + momentum.y - distance /*- wrist.y/2.f*/),
                         2*distance, 2*distance};
            // Apply ROI
            keepRoiInside(roi, maskMiss.size());
            if (roi.area() > 0)
                maskMiss(roi).setTo(0.f); // For debugging use 0.5f
        }
        // // If there is no visible desired keypoints, mask out the whole background
        // else
        //     maskMiss.setTo(0.f); // For debugging use 0.5f
    }
}

// Note: Not used for XXXX_95_XX models, instead maskPerson is used
void maskFeet(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points,
              const float stride, const float ratio, const PoseModel poseModel)
{
    const auto kneeIndexBase = (poseModel == PoseModel::COCO_23_17 ? 8 : 10);
    for (auto part = 0 ; part < 2 ; part++)
    {
        auto kneeIndex = kneeIndexBase+part*3;
        auto ankleIndex = kneeIndex+1;
        if (poseModel == PoseModel::COCO_25B_17 || poseModel == PoseModel::COCO_95_17 || poseModel == PoseModel::COCO_135_17)
        {
            kneeIndex = 13+part;
            ankleIndex = 15+part;
        }
        if (isVisible.at(kneeIndex) != 2 && isVisible.at(ankleIndex) != 2)
        {
            const auto ratioStride = 1.f / stride;
            const auto knee = ratioStride * points.at(kneeIndex);
            const auto ankle = ratioStride * points.at(ankleIndex);
            const auto distance = (int)std::round(ratio*getNorm(knee, ankle));
            const cv::Point momentum = (ankle-knee)*0.15f;
            cv::Rect roi{(int)std::round(ankle.x + momentum.x)-distance,
                         (int)std::round(ankle.y + momentum.y)-distance,
                         2*distance, 2*distance};
            // Apply ROI
            keepRoiInside(roi, maskMiss.size());
            if (roi.area() > 0)
                maskMiss(roi).setTo(0.f); // For debugging use 0.5f
        }
        // // If there is no visible desired keypoints, mask out the whole background
        // else
        //     maskMiss.setTo(0.f); // For debugging use 0.5f
    }
}

template<typename Dtype>
void fillMaskChannels(Dtype* transformedLabel, const int gridX, const int gridY, const int numberTotalChannels,
                      const int channelOffset, const cv::Mat& maskMiss)
{
    // Initialize labels to [0, 1] (depending on maskMiss)
    // // Naive version (very slow)
    // for (auto gY = 0; gY < gridY; gY++)
    // {
    //     const auto yOffset = gY*gridX;
    //     for (auto gX = 0; gX < gridX; gX++)
    //     {
    //         const auto xyOffset = yOffset + gX;
    //         const float weight = float(maskMiss.at<uchar>(gY, gX)) / 255.f;
    //         // Body part & PAFs & background channel & distance
    //         for (auto part = 0; part < numberTotalChannels; part++)
    //         // // For Distance
    //         // for (auto part = 0; part < numberTotalChannels - numberPafChannels/2; part++)
    //             transformedLabel[part*channelOffset + xyOffset] = weight;
    //     }
    // }
    // OpenCV wrapper: ~10x speed up with baseline
    cv::Mat maskMissFloat;
    const auto type = getType(Dtype(0));
    maskMiss.convertTo(maskMissFloat, type);
    maskMissFloat /= Dtype(255.f);
    // // For Distance
    // for (auto part = 0; part < numberTotalChannels - numberPafChannels/2; part++)
    for (auto part = 0; part < numberTotalChannels; part++)
    {
        auto* pointer = &transformedLabel[part*channelOffset];
        cv::Mat transformedLabel(gridY, gridX, type, (unsigned char*)(pointer));
        // // Not exactly 0 for limited floating precission
        // CHECK_LT(std::abs(cv::norm(transformedLabel-maskMissFloat)), 1e-6);
        maskMissFloat.copyTo(transformedLabel);
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateLabelMap(Dtype* transformedLabel, const cv::Size& imageSize,
                                                const cv::Mat& maskMiss, const MetaData& metaData,
                                                const int datasetIndex,
                                                const std::vector<float>& distanceAverage,
                                                const std::vector<float>& distanceSigma,
                                                std::vector<long double>& distanceAverageNew,
                                                std::vector<long double>& distanceSigmaNew,
                                                std::vector<unsigned long long>& distanceCounterNew)
{
    // Label size = image size / stride
    const auto rezX = (int)imageSize.width;
    const auto rezY = (int)imageSize.height;
    const auto stride = (int)param_.stride();
    const auto gridX = rezX / stride;
    const auto gridY = rezY / stride;
    const auto channelOffset = gridY * gridX;
    const auto numberBodyParts = getNumberBodyParts(mPoseModel); // #BP
    const auto numberPafChannels = getNumberPafChannels(mPoseModel); // 2 x #PAF
    const auto numberTafChannels = getNumberTafChannels(param_.taf_topology());
    const auto addDistance = param_.add_distance();
    const auto numberTotalChannels = getNumberTafChannels(param_.taf_topology()) + getNumberBodyBkgAndPAF(mPoseModel)
                                   + addDistance * getDistanceAverage(mPoseModel).size();
    // // For old distance
    // const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel) + (numberPafChannels / 2);

    // Labels to 0
    // 2x to consider mask and channel itself
    std::fill(transformedLabel, transformedLabel + 2*numberTotalChannels * channelOffset, 0.f);

    // Initialize labels to [0, 1] (depending on maskMiss)
    fillMaskChannels(transformedLabel, gridX, gridY, numberTotalChannels, channelOffset, maskMiss);

    // Neck-part distance
    // Mask distance labels to 0
    auto* maskDistance = transformedLabel + (numberPafChannels + numberBodyParts+1) * channelOffset;
    if (addDistance)
    {
        // // Option a) Mask all people bounding boxes as 0
        // // Person itself
        // const auto& points = metaData.jointsSelf.points;
        // const auto& isVisible = metaData.jointsSelf.isVisible;
        // // Get valid ROI bounding box
        // const auto roi = getObjROI(stride, points, isVisible, gridX, gridY, 0.5f, 0.5f);
        // // Apply to each channel
        // if (roi.area() > 0)
        // {
        //     const auto type = getType(Dtype(0));
        //     for (auto part = 0; part < getDistanceAverage(mPoseModel).size(); part++)
        //     {
        //         cv::Mat maskMissTemp(gridY, gridX, type, &maskDistance[part*channelOffset]);
        //         maskMissTemp(roi).setTo(0.f); // For debugging use 0.5f
        //     }
        // }
        // // Person others
        // for (const auto& otherPerson : metaData.jointsOthers)
        // {
        //     // Person others
        //     const auto& points = otherPerson.points;
        //     const auto& isVisible = otherPerson.isVisible;
        //     // Get valid ROI bounding box
        //     const auto roi = getObjROI(stride, points, isVisible, gridX, gridY, 0.5f, 0.5f);
        //     // Apply to each channel
        //     if (roi.area() > 0)
        //     {
        //         const auto type = getType(Dtype(0));
        //         for (auto part = 0; part < getDistanceAverage(mPoseModel).size(); part++)
        //         {
        //             cv::Mat maskMissTemp(gridY, gridX, type, &maskDistance[part*channelOffset]);
        //             maskMissTemp(roi).setTo(0.f); // For debugging use 0.5f
        //         }
        //     }
        // }
        // Option b) Mask everything as 0
        std::fill(maskDistance,
                  maskDistance + getDistanceAverage(mPoseModel).size() * channelOffset,
                  0.f);
    }

    // Background channel
    const auto backgroundMaskIndex = numberTafChannels+numberPafChannels+numberBodyParts;

    // If no people on image (e.g., if pure background image)
    if (!metaData.filled)
    {
        // Mask = 1, i.e., not masked-out (while keeping all labels to 0)
        // Note: Distance labels masks kept to 0, they are not defined for non-keypoint locations
        std::fill(transformedLabel,
                  transformedLabel + getNumberBodyBkgAndPAF(mPoseModel) * channelOffset,
                  1.f);
        if(param_.taf_topology()) throw std::runtime_error("!metaData.filled not implemented");
    }

    // If image with people
    else
    {
        // Masking out channels - For COCO_YY_ZZ models (ZZ < YY)
        if (numberBodyParts > getNumberBodyPartsLmdb(mPoseModel) || mPoseModel == PoseModel::MPII_59)
        {
            // Remove BP/PAF non-labeled channels
            // MPII hands special cases (2/4)
            //     - If left or right hand not labeled --> mask out training of those channels
            const auto missingChannels = getEmptyChannels(
                mPoseModel, metaData.jointsSelf.isVisible,
                (mPoseModel == PoseModel::MPII_59 || mPoseModel == PoseModel::MPII_65_42
                    ? 2.f : 4.f), param_.taf_topology());

            for (const auto& index : missingChannels)
                std::fill(&transformedLabel[index*channelOffset],
                          &transformedLabel[index*channelOffset + channelOffset], 0);
            const auto type = getType(Dtype(0));
            // MPII hands special cases (3/4)
            if (mPoseModel == PoseModel::MPII_65_42)
            {
                // MPII hands without body --> Remove wrists masked out to avoid overfitting
                const auto numberPafChannels = getNumberPafChannels(mPoseModel);
                for (const auto& index : {4+numberPafChannels, 7+numberPafChannels})
                    std::fill(&transformedLabel[index*channelOffset],
                              &transformedLabel[index*channelOffset + channelOffset], 0);
            }
            // Background
            if (addBkgChannel(mPoseModel))
            {
                const auto backgroundIndex = numberPafChannels + numberBodyParts;
                cv::Mat maskMissTemp(gridY, gridX, type, &transformedLabel[backgroundIndex*channelOffset]);
                // If hands
                if (numberBodyParts == 59 && mPoseModel != PoseModel::MPII_59)
                {
                    maskHands(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
                    for (const auto& jointsOther : metaData.jointsOthers)
                        maskHands(maskMissTemp, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
                }
                // If foot
                if (mPoseCategory == PoseCategory::COCO
                    && (getNumberBodyParts(mPoseModel) > 70
                        || mPoseModel == PoseModel::COCO_23_17 || mPoseModel == PoseModel::COCO_25_17
                        || mPoseModel == PoseModel::COCO_25_17E || mPoseModel == PoseModel::COCO_25B_17))
                {
                    maskFeet(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.8f,
                             mPoseModel);
                    for (const auto& jointsOther : metaData.jointsOthers)
                        maskFeet(maskMissTemp, jointsOther.isVisible, jointsOther.points, stride, 0.8f, mPoseModel);
                }
            }
        }

        // Sigma
        // Lock
        std::unique_lock<std::mutex> lock{sOcclusionsMutex};
        // Get value
        const auto keypointSigma = sKeypointSigmas[datasetIndex];
        lock.unlock();

        // Fake neck, mid hip - Mask out the person bounding box for those PAFs/BP where isVisible == 3
        // Self
        const auto backgroundMaskIndexTemp = (addBkgChannel(mPoseModel) ? backgroundMaskIndex : -1);
        const auto& joints = metaData.jointsSelf;
        if (mPoseModel == PoseModel::CAR_22)
            maskPersonIfVisibleIsX(
                transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
                mPoseModel, 2, 0.3, 0.3, param_.taf_topology());
        maskPersonIfVisibleIsX(
            transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
            mPoseModel, 3, 0.3, 0.3, param_.taf_topology());
        maskPersonIfVisibleIsX(
            transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
            mPoseModel, 4, 0.3, 0.3, param_.taf_topology());
        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            const auto& joints = metaData.jointsOthers[otherPerson];
            if (mPoseModel == PoseModel::CAR_22)
                maskPersonIfVisibleIsX(
                    transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
                    mPoseModel, 2, 0.3, 0.3, param_.taf_topology());
            maskPersonIfVisibleIsX(
                transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
                mPoseModel, 3, 0.3, 0.3, param_.taf_topology());
            maskPersonIfVisibleIsX(
                transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
                mPoseModel, 4, 0.3, 0.3, param_.taf_topology());
        }

        // Body parts
        for (auto part = 0; part < numberBodyParts; part++)
        {
            const auto partRatio = getSigmaRatio(mPoseModel)[part];
            const auto keypointSigmaPart = keypointSigma*partRatio;
            // Self
            if (metaData.jointsSelf.isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsSelf.points[part];
                //cv::Point centerPoint = cv::Point(0,0);
                putGaussianMaps(
                    transformedLabel + (numberTotalChannels+numberTafChannels+numberPafChannels+part)*channelOffset,
                    transformedLabel + (numberTafChannels+numberPafChannels+part)*channelOffset,
                    centerPoint, stride, gridX, gridY, keypointSigmaPart, partRatio);
            }

            // For every other person
            for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
            {
                if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
                {
                    const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                    putGaussianMaps(
                        transformedLabel + (numberTotalChannels+numberTafChannels+numberPafChannels+part)*channelOffset,
                        transformedLabel + (numberTafChannels+numberPafChannels+part)*channelOffset,
                        centerPoint, stride, gridX, gridY, keypointSigmaPart, partRatio);
                }
            }
        }

        // For upper neck and top head --> mask out people bounding boxes, leave rest as mask = 1, neck/head value = 0
        if (mPoseCategory == PoseCategory::COCO && (mPoseModel == PoseModel::COCO_25B_23
            || mPoseModel == PoseModel::COCO_25B_17 || getNumberBodyParts(mPoseModel) > 70))
        {
            // Mask people
            // Indexes: Real neck, top head, foot (for COCO_X_17), etc...
            const auto minus1Channels = getMinus1Channels(mPoseModel, joints.isVisible, param_.taf_topology());
            maskPerson(
                transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
                false, mPoseModel, minus1Channels, 1.f, 1.f, param_.taf_topology());
                // false, mPoseModel, minus1Channels, 0.f, 0.f); // Debug
            // For every other person
            for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
            {
                const auto& joints = metaData.jointsOthers[otherPerson];
                maskPerson(
                    transformedLabel, joints.points, joints.isVisible, stride, gridX, gridY, backgroundMaskIndexTemp,
                    false, mPoseModel, minus1Channels, 1.f, 1.f, param_.taf_topology());
                    // false, mPoseModel, minus1Channels, 0.f, 0.f); // Debug
            }
        }

        // Neck-part distance
        if (addDistance)
        {
            // Estimate average distance between keypoints
            if (distanceAverageNew.empty())
            {
                distanceAverageNew.resize(distanceAverage.size(), 0.L);
                distanceSigmaNew.resize(distanceSigma.size(), 0.L);
                distanceCounterNew.resize(distanceAverage.size()/2, 0ull);
            }
            if (distanceAverage.size() != distanceAverageNew.size() || distanceSigma.size() != distanceSigmaNew.size())
                throw std::runtime_error{"DISTANCE_AVERAGE or SIGMA_AVERAGE not filled in poseModel.cpp, error"
                                         + getLine(__LINE__, __FUNCTION__, __FILE__)};
            auto* channelDistance = transformedLabel + (numberTotalChannels + numberPafChannels + numberBodyParts+1)
                                  * channelOffset;
            // Multi-star version
            for (auto partTarget = 0; partTarget < numberBodyParts; partTarget++)
            {
                cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
                for (auto rootPart = 0; rootPart < numberBodyParts; rootPart++)
                {
                    if (rootPart != partTarget)
                    {
                        // Self
                        if (metaData.jointsSelf.isVisible[rootPart] <= 1
                            && metaData.jointsSelf.isVisible[partTarget] <= 1)
                        {
                            const auto& rootPoint = metaData.jointsSelf.points[rootPart];
                            const auto& targetPoint = metaData.jointsSelf.points[partTarget];
                            putDistanceMaps(
                                channelDistance + 2*partTarget*channelOffset,
                                channelDistance + (2*partTarget+1)*channelOffset,
                                maskDistance + 2*partTarget*channelOffset,
                                maskDistance + (2*partTarget+1)*channelOffset,
                                count, rootPoint, targetPoint, stride, gridX, gridY, keypointSigma,
                                &distanceAverage[2*partTarget], &distanceSigma[2*partTarget],
                                &distanceAverageNew[2*partTarget], &distanceSigmaNew[2*partTarget],
                                distanceCounterNew[partTarget]
                            );
                        }
                        // For every other person
                        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
                        {
                            if (metaData.jointsOthers[otherPerson].isVisible[rootPart] <= 1
                                && metaData.jointsOthers[otherPerson].isVisible[partTarget] <= 1)
                            {
                                const auto& rootPoint = metaData.jointsOthers[otherPerson].points[rootPart];
                                const auto& targetPoint = metaData.jointsOthers[otherPerson].points[partTarget];
                                putDistanceMaps(
                                    channelDistance + 2*partTarget*channelOffset,
                                    channelDistance + (2*partTarget+1)*channelOffset,
                                    maskDistance + 2*partTarget*channelOffset,
                                    maskDistance + (2*partTarget+1)*channelOffset,
                                    count, rootPoint, targetPoint, stride, gridX, gridY, keypointSigma,
                                    &distanceAverage[2*partTarget], &distanceSigma[2*partTarget],
                                    &distanceAverageNew[2*partTarget], &distanceSigmaNew[2*partTarget],
                                    distanceCounterNew[partTarget]
                                );
                            }
                        }
                    }
                }
            }
            // // Neck-star version
            // const auto rootIndex = getRootIndex();
            // for (auto partOrigin = 0; partOrigin < numberBodyParts; partOrigin++)
            // {
            //     if (rootIndex != partOrigin)
            //     {
            //         cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
            //         const auto partTarget = (partOrigin > rootIndex ? partOrigin-1 : partOrigin);
            //         // Self
            //         if (metaData.jointsSelf.isVisible[partOrigin] <= 1
            //             && metaData.jointsSelf.isVisible[rootIndex] <= 1)
            //         {
            //             const auto& rootPoint = metaData.jointsSelf.points[rootIndex];
            //             const auto& centerPoint = metaData.jointsSelf.points[partOrigin];
            //             putDistanceMaps(
            //                 channelDistance + 2*partTarget*channelOffset,
            //                 channelDistance + (2*partTarget+1)*channelOffset,
            //                 maskDistance + 2*partTarget*channelOffset,
            //                 maskDistance + (2*partTarget+1)*channelOffset,
            //                 count, rootPoint, centerPoint, stride, gridX, gridY, keypointSigma,
            //                 &distanceAverage[2*partTarget], &distanceSigma[2*partTarget],
            //                 &distanceAverageNew[2*partTarget], &distanceSigmaNew[2*partTarget],
            //                 distanceCounterNew[partTarget]
            //             );
            //         }
            //         // For every other person
            //         for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
            //         {
            //             if (metaData.jointsOthers[otherPerson].isVisible[partOrigin] <= 1
            //                 && metaData.jointsOthers[otherPerson].isVisible[rootIndex] <= 1)
            //             {
            //                 const auto& rootPoint = metaData.jointsOthers[otherPerson].points[rootIndex];
            //                 const auto& centerPoint = metaData.jointsOthers[otherPerson].points[partOrigin];
            //                 putDistanceMaps(
            //                     channelDistance + 2*partTarget*channelOffset,
            //                     channelDistance + (2*partTarget+1)*channelOffset,
            //                     maskDistance + 2*partTarget*channelOffset,
            //                     maskDistance + (2*partTarget+1)*channelOffset,
            //                     count, rootPoint, centerPoint, stride, gridX, gridY, keypointSigma,
            //                     &distanceAverage[2*partTarget], &distanceSigma[2*partTarget],
            //                     &distanceAverageNew[2*partTarget], &distanceSigmaNew[2*partTarget],
            //                     distanceCounterNew[partTarget]
            //                 );
            //             }
            //         }
            //     }
            // }
        }
    }

    // Background channel
    // Naive implementation
    const auto backgroundIndex = numberTotalChannels+numberTafChannels+numberPafChannels+numberBodyParts;
    auto* transformedLabelBkg = &transformedLabel[backgroundIndex*channelOffset];
    if (addBkgChannel(mPoseModel))
    {
        for (auto gY = 0; gY < gridY; gY++)
        {
            const auto yOffset = gY*gridX;
            for (auto gX = 0; gX < gridX; gX++)
            {
                const auto xyOffset = yOffset + gX;
                Dtype maximum = 0.;
                for (auto part = numberTotalChannels+numberTafChannels+numberPafChannels ; part < backgroundIndex ; part++)
                {
                    const auto index = part * channelOffset + xyOffset;
                    maximum = (maximum > transformedLabel[index]) ? maximum : transformedLabel[index];
                }
                transformedLabelBkg[xyOffset] = std::max(Dtype(1.)-maximum, Dtype(0.));
            }
        }
    }
    // Background mask channel
    auto* transformedLabelBkgMask = &transformedLabel[backgroundMaskIndex*channelOffset];

    if (metaData.filled)
    {
        // TAFs
        if(param_.taf_topology()){
            const auto& tafMapA = getTafIndexA(param_.taf_topology());
            const auto& tafMapB = getTafIndexB(param_.taf_topology());
            const auto threshold = 1;
            for (auto i = 0 ; i < tafMapA.size() ; i++)
            {
                cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
                // Self
                const auto& joints = metaData.jointsSelf;
                const auto& jointsPrev = metaData.jointsSelfPrev;
                if(jointsPrev.isVisible.size()){
                    if (joints.isVisible[tafMapA[i]] <= 1 && jointsPrev.isVisible[tafMapB[i]] <= 1)
                    {
                        putVectorMaps(transformedLabel + (numberTotalChannels + 0 + 2*i)*channelOffset,
                                      transformedLabel + (numberTotalChannels + 0 + 2*i + 1)*channelOffset,
                                      transformedLabel + (0 + 2*i)*channelOffset,
                                      transformedLabel + (0 + 2*i + 1)*channelOffset,
                                      // // For Distance
                                      // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                                      // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                                      count, joints.points[tafMapA[i]], jointsPrev.points[tafMapB[i]],
                                      param_.stride(), gridX, gridY, threshold,
                                      // diagonal, diagonalProportion,
                                      // For Car_v1 --> Not all cars labeled, so mask out everything but keypoints/PAFs
                                      // transformedLabelBkgMask
                                      (mPoseModel == PoseModel::CAR_12 ? transformedLabelBkgMask : nullptr)
                                      );
                    }
                }

                // For every other person
                for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
                {
                    const auto& joints = metaData.jointsOthers[otherPerson];
                    if(metaData.jointsOthersPrev.size()){
                        const auto& jointsPrev = metaData.jointsOthersPrev[otherPerson];

                        if(jointsPrev.isVisible.size() && joints.isVisible.size()){
                            if (joints.isVisible[tafMapA[i]] <= 1 && jointsPrev.isVisible[tafMapB[i]] <= 1)
                            {
                                putVectorMaps(transformedLabel + (numberTotalChannels + 0 + 2*i)*channelOffset,
                                              transformedLabel + (numberTotalChannels + 0 + 2*i + 1)*channelOffset,
                                              transformedLabel + (0 + 2*i)*channelOffset,
                                              transformedLabel + (0 + 2*i + 1)*channelOffset,
                                              // // For Distance
                                              // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                                              // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                                              count, joints.points[tafMapA[i]], jointsPrev.points[tafMapB[i]],
                                              param_.stride(), gridX, gridY, threshold,
                                              // diagonal, diagonalProportion,
                                              // For Car_v1 --> Not all cars labeled, so mask out everything but keypoints/PAFs
                                              // transformedLabelBkgMask
                                              (mPoseModel == PoseModel::CAR_12 ? transformedLabelBkgMask : nullptr)
                                              );
                            }
                        }
                    }
                }
            }
        }

        // PAFs
        const auto& labelMapA = getPafIndexA(mPoseModel);
        const auto& labelMapB = getPafIndexB(mPoseModel);
        const auto threshold = 1;
        // const auto diagonal = sqrt(gridX*gridX + gridY*gridY);
        // const auto diagonalProportion = (
        //     mCurrentEpoch > 0 ? 1.f : metaData.writeNumber/(float)metaData.totalWriteNumber);
        for (auto i = 0 ; i < labelMapA.size() ; i++)
        {
            cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
            // Self
            const auto& joints = metaData.jointsSelf;
            if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
            {
                putVectorMaps(transformedLabel + (numberTotalChannels + numberTafChannels + 2*i)*channelOffset,
                              transformedLabel + (numberTotalChannels + numberTafChannels + 2*i + 1)*channelOffset,
                              transformedLabel + (numberTafChannels + 2*i)*channelOffset,
                              transformedLabel + (numberTafChannels + 2*i + 1)*channelOffset,
                              // // For Distance
                              // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                              param_.stride(), gridX, gridY, threshold,
                              // diagonal, diagonalProportion,
                              // For Car_v1 --> Not all cars labeled, so mask out everything but keypoints/PAFs
                              // transformedLabelBkgMask
                              (mPoseModel == PoseModel::CAR_12 ? transformedLabelBkgMask : nullptr)
                              );
            }

            // For every other person
            for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
            {
                const auto& joints = metaData.jointsOthers[otherPerson];
                if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
                {
                    putVectorMaps(transformedLabel + (numberTotalChannels + numberTafChannels + 2*i)*channelOffset,
                                  transformedLabel + (numberTotalChannels + numberTafChannels + 2*i + 1)*channelOffset,
                                  transformedLabel + (numberTafChannels + 2*i)*channelOffset,
                                  transformedLabel + (numberTafChannels + 2*i + 1)*channelOffset,
                                  // // For Distance
                                  // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                                  // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                                  count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                                  param_.stride(), gridX, gridY, threshold,
                                  // diagonal, diagonalProportion,
                                  // For Car_v1 --> Not all cars labeled, so mask out everything but keypoints/PAFs
                                  // transformedLabelBkgMask
                                  (mPoseModel == PoseModel::CAR_12 ? transformedLabelBkgMask : nullptr)
                                  );
                }
            }
        }
        // // Re-normalize masks (otherwise PAF explodes)
        // const auto finalImageArea = gridX*gridY;
        // for (auto i = 0 ; i < labelMapA.size() ; i++)
        // {
        //     auto* initPoint = &transformedLabel[2*i*channelOffset];
        //     const auto accumulation = std::accumulate(initPoint, initPoint+channelOffset, 0);
        //     const auto ratio = finalImageArea / (float)accumulation;
        //     if (ratio > 1.01 || ratio < 0.99)
        //         std::transform(initPoint, initPoint + 2*channelOffset, initPoint,
        //                        std::bind1st(std::multiplies<Dtype>(), ratio)) ;
        // }

        // MPII hands special cases (4/4)
        // Make background channel as non-masked out region for visible labels (for cases with no all people labeled)
        if (mPoseModel == PoseModel::MPII_65_42)
        {
            // MPII - If left or right hand not labeled --> mask out training of those channels
            for (auto part = 0 ; part < 2 ; part++)
            {
                const auto wristIndex = (part == 0 ? 7:4);
                // jointsOther not used --> assuming 1 person / image
                if (metaData.jointsSelf.isVisible[wristIndex] <= 1)
                {
                    std::vector<int> handIndexes;
                    // PAFs
                    for (auto i = 26 ; i < 46 ; i++)
                    {
                        handIndexes.emplace_back(2*(i+20*part));
                        handIndexes.emplace_back(handIndexes.back()+1);
                    }
                    // Body parts
                    for (auto i = 25 ; i < 45 ; i++)
                        handIndexes.emplace_back(i+20*part + numberPafChannels);
                    // Fill those channels
                    for (const auto& handIndex : handIndexes)
                    {
                        for (auto gY = 0; gY < gridY; gY++)
                        {
                            const auto yOffset = gY*gridX;
                            for (auto gX = 0; gX < gridX; gX++)
                            {
                                const auto xyOffset = yOffset + gX;
                                // transformedLabel[handIndex*channelOffset + xyOffset] = 1.0;
                                transformedLabel[handIndex*channelOffset + xyOffset] = (
                                    transformedLabel[backgroundIndex*channelOffset + xyOffset] < 1-1e-6
                                        ? 1 : transformedLabel[handIndex*channelOffset + xyOffset]);
                            }
                        }
                    }
                }
            }
        }
        // For Car_v1 --> Not all cars labeled, so mask out everything but keypoints/PAFs
        if (mPoseModel == PoseModel::CAR_12)
            for (auto i = 0 ; i < backgroundMaskIndex ; i++)
                std::copy(transformedLabelBkgMask, transformedLabelBkgMask+channelOffset,
                          &transformedLabel[i*channelOffset]);
    }

    // Background mask channel
    if (addBkgChannel(mPoseModel))
    {
        // Set to 1 bkg if some keypoint was annotated as true in that pixel
        // Because this would mean that the real bkg (even if there are some missing keypoints)
        // will not be 1, so this approximation is better than nothing.
        for (auto gY = 0; gY < gridY; gY++)
        {
            const auto yOffset = gY*gridX;
            for (auto gX = 0; gX < gridX; gX++)
            {
                const auto xyOffset = yOffset + gX;
                // // For Car_v1 --> Not all cars labeled, so mask out everything but keypoints/PAFs
                // if (mPoseModel == PoseModel::CAR_12 && transformedLabelBkg[xyOffset] < 1 - 1e-9)
                // if (transformedLabelBkg[xyOffset] < 1 - 1e-9)
                if (transformedLabelBkg[xyOffset] < 0.85)
                    transformedLabelBkgMask[xyOffset] = Dtype(1);
            }
        }
    }
}
// OpenPose: added end

INSTANTIATE_CLASS(OPDataTransformer);

}  // namespace caffe
