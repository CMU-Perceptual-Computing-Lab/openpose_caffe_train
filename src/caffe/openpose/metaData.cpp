#include <iostream>
#include <thread> // std::this_thread::sleep_for
#include <caffe/openpose/getLine.hpp>
#include <caffe/openpose/metaData.hpp>
#include <glog/logging.h>

namespace caffe {
    // Private functions
    template<typename Dtype>
    Dtype decodeNumber(const char* charPtr)
    {
        Dtype pf;
        memcpy(&pf, charPtr, sizeof(Dtype));
        return pf;
    }

    std::string decodeString(const char* charPtr)
    {
        std::string result = "";
        for (auto index = 0 ; charPtr[index] != 0 ; index++)
            result.push_back(char(charPtr[index]));
        return result;
    }

    void lmdbJointsToOurModel(Joints& joints, const PoseModel poseModel)
    {
        // Transform joints in metaData from getNumberBodyPartsLmdb(poseModel) (specified in prototxt)
        // to getNumberBodyAndPafChannels(poseModel) (specified in prototxt)
        auto jointsOld = joints;

        // Common operations
        const auto numberBodyParts = getNumberBodyParts(poseModel);
        joints.points.resize(numberBodyParts);
        joints.isVisible.resize(numberBodyParts);

        // From COCO/DomeDB to OP keypoint indexes
        const auto& lmdbToOurModel = getLmdbToOpenPoseKeypoints(poseModel);
        for (auto i = 0 ; i < lmdbToOurModel.size() ; i++)
        {
            // If point defined
            if (!lmdbToOurModel[i].empty())
            {
                // Initialize point
                joints.points[i] = cv::Point2f{0.f, 0.f};
                // Get and average joints.points[i]
                for (auto& lmdbToOurModelIndex : lmdbToOurModel[i])
                    joints.points[i] += jointsOld.points[lmdbToOurModelIndex];
                joints.points[i] *= (1.f / (float)lmdbToOurModel[i].size());
                // Get joints.isVisible[i]
                // Original COCO:
                //     v=0: not labeled
                //     v=1: labeled but not visible
                //     v=2: labeled and visible
                // OpenPose:
                //     v=0: labeled but not visible
                //     v=1: labeled and visible
                //     v=2: out of image / unlabeled
                //     v=3: on of it's parents is v=2
                //     v=4: Not labeled for this dataset
                joints.isVisible[i] = 1;
                // >1 elements
                if (lmdbToOurModel[i].size() > 1)
                {
                    // See whether to make it 2 (all of them are not visible) or 3 (>= 1 of them are visible)
                    bool thereAre0s = false;
                    bool thereAre1s = false;
                    bool thereAre2s = false;
                    bool thereAre4s = false;
                    for (auto& lmdbToOurModelIndex : lmdbToOurModel[i])
                    {
                        if (jointsOld.isVisible[lmdbToOurModelIndex] == 0)
                            thereAre0s = true;
                        else if (jointsOld.isVisible[lmdbToOurModelIndex] == 1)
                            thereAre1s = true;
                        else if (jointsOld.isVisible[lmdbToOurModelIndex] == 2)
                            thereAre2s = true;
                        else if (jointsOld.isVisible[lmdbToOurModelIndex] == 4)
                            thereAre4s = true;
                        // Debugging-only
                        else
                        {
                            LOG(INFO) << "Unexpected jointsOld.isVisible[lmdbToOurModelIndex] = "
                                      << jointsOld.isVisible[lmdbToOurModelIndex]
                                      << " on index i = " << i
                                      << " and lmdbToOurModelIndex = " << lmdbToOurModelIndex;
                            thereAre4s = true; // Ignored channel for image
                        }
                    }
                    // Set final visibility flag
                    // If point not annotated on LMDB file --> isVisible == 4
                    if (thereAre4s)
                        joints.isVisible[i] = 4;
                    // If some not labeled and some labeled --> isVisible == 3
                    else if (thereAre2s && (thereAre0s || thereAre1s))
                        joints.isVisible[i] = 3;
                    // If non labeled --> isVisible == 2
                    else if (thereAre2s)
                        joints.isVisible[i] = 2;
                    // If not non-visible and some occluded --> isVisible == 0
                    else if (thereAre0s)
                        joints.isVisible[i] = 0;
                    // Else 1 (if all are 1s) --> isVisible == 1
                }
                // Only 1 element
                else
                {
                    for (auto& lmdbToOurModelIndex : lmdbToOurModel[i])
                    {
                        // If point not annotated on LMDB file --> isVisible == 4
                        if (jointsOld.isVisible[lmdbToOurModelIndex] == 4)
                            joints.isVisible[i] = 4;
                        // If any of them is 2 --> 2 (not in the image or unlabeled)
                        else if (jointsOld.isVisible[lmdbToOurModelIndex] == 2)
                        {
                            // Keypoint to keypoint correspondence
                            if (lmdbToOurModel[i].size() < 2)
                                joints.isVisible[i] = 2;
                            // Fake neck, midhip: When interpolated from >=2 keypoints, and at least 1 of them is missing
                            else
                                joints.isVisible[i] = 3;
                            break;
                        }
                        // If no 2 but 0 -> 0 (ocluded but located)
                        else if (jointsOld.isVisible[lmdbToOurModelIndex] == 0)
                            joints.isVisible[i] = 0;
                        // Else 1 (if all are 1s)
                    }
                }
            }
            // If point not defined
            else
            {
                joints.points[i] = cv::Point2f{0.f, 0.f};
                joints.isVisible[i] = 4;
            }
        }
    }

    void lmdbJointsToOurModel(MetaData& metaData, const PoseModel poseModel)
    {
        lmdbJointsToOurModel(metaData.jointsSelf, poseModel);
        for (auto& joints : metaData.jointsOthers)
            lmdbJointsToOurModel(joints, poseModel);
    }

    // Public functions
    template<typename Dtype>
    bool readMetaData(
        MetaData& metaData, int& currentEpoch, std::string& datasetString, const char* data,
        const size_t offsetPerLine, const PoseCategory poseCategory, const PoseModel poseModel,
        const float cropSizeY)
    {
        metaData.filled = true;
        // Dataset name
        metaData.datasetString = decodeString(data);
        if (datasetString.empty())
            datasetString = metaData.datasetString;
        // Image Dimension
        metaData.imageSize = cv::Size{(int)decodeNumber<Dtype>(&data[offsetPerLine+4]),
                                      (int)decodeNumber<Dtype>(&data[offsetPerLine])};

        // Validation, #people, counters
        metaData.numberOtherPeople = (int)data[2*offsetPerLine];
        metaData.peopleIndex = (int)data[2*offsetPerLine+1];
        metaData.annotationListIndex = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+2]));
        metaData.writeNumber = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+6]));
        metaData.totalWriteNumber = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+10]));

        // Count epochs according to counters
        if (metaData.writeNumber == 0)
            currentEpoch++;
        metaData.epoch = currentEpoch;
        if (metaData.writeNumber % 1000 == 0)
        {
            LOG(INFO) << "datasetString: " << metaData.datasetString <<"; imageSize: " << metaData.imageSize
                      << "; metaData.annotationListIndex: " << metaData.annotationListIndex
                      << "; metaData.writeNumber: " << metaData.writeNumber
                      << "; metaData.totalWriteNumber: " << metaData.totalWriteNumber
                      << "; metaData.epoch: " << metaData.epoch;
        }

        // Objpos
        metaData.objPos.x = decodeNumber<Dtype>(&data[3*offsetPerLine]);
        metaData.objPos.y = decodeNumber<Dtype>(&data[3*offsetPerLine+4]);
        // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
        if (poseCategory == PoseCategory::COCO)
            metaData.objPos -= cv::Point2f{1.f,1.f};
        // scaleSelf, jointsSelf
        // 368 hardcoded for compatibility with the old LMDB...
        const auto imageRatio = 368.f/cropSizeY;
        metaData.scaleSelf = decodeNumber<Dtype>(&data[4*offsetPerLine])*imageRatio;
        auto& jointSelf = metaData.jointsSelf;
        const auto numberPartsInLmdb = getNumberBodyPartsLmdb(poseModel);
        jointSelf.points.resize(numberPartsInLmdb);
        jointSelf.isVisible.resize(numberPartsInLmdb);
        for (auto part = 0 ; part < numberPartsInLmdb; part++)
        {
            // Point
            auto& jointPoint = jointSelf.points[part];
            jointPoint.x = decodeNumber<Dtype>(&data[5*offsetPerLine+4*part]);
            jointPoint.y = decodeNumber<Dtype>(&data[6*offsetPerLine+4*part]);
            // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
            if (poseCategory == PoseCategory::COCO)
                jointPoint -= cv::Point2f{1.f,1.f};
            // isVisible flag
            const auto isVisible = decodeNumber<Dtype>(&data[7*offsetPerLine+4*part]);
            // if (datasetString != metaData.datasetString || isVisible > 2 || std::isnan(isVisible) || std::isnan(isVisible))
            if (datasetString != metaData.datasetString || isVisible > 4 || std::isnan(isVisible) || std::isnan(isVisible))
            {
                // LOG(INFO) << "CHECK_LE(isVisible, 2) failed!!!!!\n"
                //           << "datasetString: " << metaData.datasetString;
                LOG(INFO) << "\n"
                          << "----------------------\n"
                          << "----------------------\n"
                          << "----------------------\n"
                          << "ERROR READING METADATA\n"
                          << "----------------------\n"
                          << "imageSize: " << metaData.imageSize
                          << "; metaData.annotationListIndex: " << metaData.annotationListIndex
                          << "; metaData.writeNumber: " << metaData.writeNumber
                          << "; metaData.totalWriteNumber: " << metaData.totalWriteNumber
                          << "; metaData.epoch: " << metaData.epoch
                          << "; isVisible: " << isVisible << "\n"
                          // << "Data:\n" << data;
                          << "----------------------\n";
                return false;
                // CHECK(false);
            }
            // // Equivalent code to above code
            // CHECK_LE(isVisible, 2); // isVisible in range [0, 2]
            jointSelf.isVisible[part] = std::round(isVisible);
            if (jointSelf.isVisible[part] != 2)
            {
                if (jointPoint.x < 0 || jointPoint.y < 0
                    || jointPoint.x >= metaData.imageSize.width || jointPoint.y >= metaData.imageSize.height)
                {
                    jointSelf.isVisible[part] = 2; // 2 means cropped/unlabeled, 0 means occluded but in image
                }
            }
            // LOG(INFO) << jointPoint.x << " " << jointPoint.y << " " << jointSelf.isVisible[part];
        }

        // Others (7 lines loaded)
        metaData.objPosOthers.resize(metaData.numberOtherPeople);
        metaData.scaleOthers.resize(metaData.numberOtherPeople);
        metaData.jointsOthers.resize(metaData.numberOtherPeople);
        for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
        {
            metaData.objPosOthers[person].x = decodeNumber<Dtype>(&data[(8+person)*offsetPerLine]);
            metaData.objPosOthers[person].y = decodeNumber<Dtype>(&data[(8+person)*offsetPerLine+4]);
            // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
            if (poseCategory == PoseCategory::COCO)
                metaData.objPosOthers[person] -= cv::Point2f{1.f,1.f};
            metaData.scaleOthers[person]  = decodeNumber<Dtype>(&data[(8+metaData.numberOtherPeople)
                                                                *offsetPerLine+4*person])*imageRatio;
        }
        // 8 + numberOtherPeople lines loaded
        for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
        {
            auto& currentPerson = metaData.jointsOthers[person];
            currentPerson.points.resize(numberPartsInLmdb);
            currentPerson.isVisible.resize(numberPartsInLmdb);
            for (auto part = 0 ; part < numberPartsInLmdb; part++)
            {
                // Point
                // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
                auto& jointPoint = currentPerson.points[part];
                jointPoint.x = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person)
                                                   *offsetPerLine+4*part]);
                jointPoint.y = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person+1)
                                                   *offsetPerLine+4*part]);
                // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
                if (poseCategory == PoseCategory::COCO)
                    jointPoint -= cv::Point2f{1.f,1.f};
                // isVisible flag
                const auto isVisible = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person+2)
                                                           *offsetPerLine+4*part]);
                currentPerson.isVisible[part] = std::round(isVisible);
                if (currentPerson.isVisible[part] != 2)
                {
                    if (jointPoint.x < 0 || jointPoint.y < 0
                        || jointPoint.x >= metaData.imageSize.width || jointPoint.y >= metaData.imageSize.height)
                    {
                        currentPerson.isVisible[part] = 2; // 2 means cropped/unlabeled, 0 means occluded  but in image
                    }
                }
            }
        }
        if (poseCategory == PoseCategory::DOME)
        {
            // Image path
            int currentLine = 8;
            if (metaData.numberOtherPeople != 0)
                currentLine = 9+4*metaData.numberOtherPeople;
            metaData.imageSource = decodeString(&data[currentLine * offsetPerLine]);
            // // Depth enabled
            // metaData.depthEnabled = decodeNumber<Dtype>(&data[(currentLine+1) * offsetPerLine]) != Dtype(0);
            // // Depth path
            // if (metaData.depthEnabled)
            //     metaData.depthSource = decodeString(&data[(currentLine+2) * offsetPerLine]);
        }
        else
            metaData.depthEnabled = false;

        // Transform joints in metaData from getNumberBodyPartsLmdb(mPoseModel) (specified in prototxt)
        // to getNumberBodyAndPafChannels(mPoseModel) (specified in prototxt)
        lmdbJointsToOurModel(metaData, poseModel);

        return true;
    }

    template bool readMetaData<float>(MetaData& metaData, int& currentEpoch, std::string& datasetString,
                                      const char* data, const size_t offsetPerLine,
                                      const PoseCategory poseCategory, const PoseModel poseModel, const float cropSizeY);
    template bool readMetaData<double>(MetaData& metaData, int& currentEpoch, std::string& datasetString,
                                       const char* data, const size_t offsetPerLine,
                                       const PoseCategory poseCategory, const PoseModel poseModel, const float cropSizeY);
}  // namespace caffe
