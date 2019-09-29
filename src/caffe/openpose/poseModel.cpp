#include <algorithm>    // std::sort, std::unique, std::distance
#include <iostream>
#include <map>
#include <caffe/openpose/poseModel.hpp>
#include <caffe/openpose/getLine.hpp>

namespace caffe {
// General information:
// DOME_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "LowerAbs"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "RBigToe"},
//     {20, "RSmallToe"},
//     {21, "LBigToe"},
//     {22, "LSmallToe"},
//     {19/23, "Background"},
// };
// COCO_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "LEye"},
//     {2,  "REye"},
//     {3,  "LEar"},
//     {4,  "REar"},
//     {5,  "LShoulder"},
//     {6,  "RShoulder"},
//     {7,  "LElbow"},
//     {8,  "RElbow"},
//     {9,  "LWrist"},
//     {10, "RWrist"},
//     {11, "LHip"},
//     {12, "RHip"},
//     {13, "LKnee"},
//     {14, "RKnee"},
//     {15, "LAnkle"},
//     {16, "RAnkle"},
//     {17-21, "Background"},
//     {17, "LBigToe"},
//     {18, "LSmallToe"},
//     {19, "LHeel"},
//     {20, "RBigToe"},
//     {21, "RSmallToe"},
//     {22, "RHeel"},
// };
// MPII_BODY_PARTS { // http://human-pose.mpi-inf.mpg.de/#download
//     {0,  "RAnkle"},
//     {1,  "RKnee"},
//     {2,  "RHip"},
//     {3,  "LHip"},
//     {4,  "LKnee"},
//     {5,  "LAnkle"},
//     {6,  "MHip"}, // Pelvis in MPII website
//     {7,  "Neck"}, // Thorax in MPII website
//     {8,  "UpperNeck"},
//     {9,  "HeadTop"},
//     {10, "RWrist"},
//     {11, "RElbow"},
//     {12, "RShoulder"},
//     {13, "LShoulder"},
//     {14, "LElbow"},
//     {15, "LWrist"},
//     {16, "Background"},
// };
const std::map<unsigned int, std::string> POSE_BODY_18_BODY_PARTS {
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "RHip"},
    {9,  "RKnee"},
    {10, "RAnkle"},
    {11, "LHip"},
    {12, "LKnee"},
    {13, "LAnkle"},
    {14, "REye"},
    {15, "LEye"},
    {16, "REar"},
    {17, "LEar"},
    {18, "Background"},
};
const std::map<unsigned int, std::string> POSE_BODY_19_BODY_PARTS {
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "MHip"},
    {9,  "RHip"},
    {10, "RKnee"},
    {11, "RAnkle"},
    {12, "LHip"},
    {13, "LKnee"},
    {14, "LAnkle"},
    {15, "REye"},
    {16, "LEye"},
    {17, "REar"},
    {18, "LEar"},
    {19, "Background"},
};
const std::map<unsigned int, std::string> POSE_BODY_23_BODY_PARTS {
    {0,  "Nose"},
    {1,  "RShoulder"},
    {2,  "RElbow"},
    {3,  "RWrist"},
    {4,  "LShoulder"},
    {5,  "LElbow"},
    {6,  "LWrist"},
    {7,  "RHip"},
    {8,  "RKnee"},
    {9,  "RAnkle"},
    {10, "LHip"},
    {11, "LKnee"},
    {12, "LAnkle"},
    {13, "REye"},
    {14, "LEye"},
    {15, "REar"},
    {16, "LEar"},
    {17, "LBigToe"},
    {18, "LSmallToe"},
    {19, "LHeel"},
    {20, "RBigToe"},
    {21, "RSmallToe"},
    {22, "RHeel"},
    {23, "Background"},
};
const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "MHip"},
    {9,  "RHip"},
    {10, "RKnee"},
    {11, "RAnkle"},
    {12, "LHip"},
    {13, "LKnee"},
    {14, "LAnkle"},
    {15, "REye"},
    {16, "LEye"},
    {17, "REar"},
    {18, "LEar"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
    {25, "Background"},
};
const std::map<unsigned int, std::string> POSE_BODY_25B_BODY_PARTS {
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "RKnee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17, "UpperNeck"},
    {18, "HeadTop"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
};
const auto H59 = 19;
const std::map<unsigned int, std::string> POSE_BODY_59_BODY_PARTS {
    // Body
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "LowerAbs"},
    {9,  "RHip"},
    {10, "RKnee"},
    {11, "RAnkle"},
    {12, "LHip"},
    {13, "LKnee"},
    {14, "LAnkle"},
    {15, "REye"},
    {16, "LEye"},
    {17, "REar"},
    {18, "LEar"},
    // Left hand
    {H59+0, "LThumb1CMC"},       {H59+1, "LThumb2Knuckles"}, {H59+2, "LThumb3IP"},   {H59+3, "LThumb4FingerTip"},
    {H59+4, "LIndex1Knuckles"},  {H59+5, "LIndex2PIP"},      {H59+6, "LIndex3DIP"},  {H59+7, "LIndex4FingerTip"},
    {H59+8, "LMiddle1Knuckles"}, {H59+9, "LMiddle2PIP"},     {H59+10, "LMiddle3DIP"},{H59+11, "LMiddle4FingerTip"},
    {H59+12, "LRing1Knuckles"},  {H59+13, "LRing2PIP"},      {H59+14, "LRing3DIP"},  {H59+15, "LRing4FingerTip"},
    {H59+16, "LPinky1Knuckles"}, {H59+17, "LPinky2PIP"},     {H59+18, "LPinky3DIP"}, {H59+19, "LPinky4FingerTip"},
    // Right hand
    {H59+20, "RThumb1CMC"},      {H59+21, "RThumb2Knuckles"},{H59+22, "RThumb3IP"},  {H59+23, "RThumb4FingerTip"},
    {H59+24, "RIndex1Knuckles"}, {H59+25, "RIndex2PIP"},     {H59+26, "RIndex3DIP"}, {H59+27, "RIndex4FingerTip"},
    {H59+28, "RMiddle1Knuckles"},{H59+29, "RMiddle2PIP"},    {H59+30, "RMiddle3DIP"},{H59+31, "RMiddle4FingerTip"},
    {H59+32, "RRing1Knuckles"},  {H59+33, "RRing2PIP"},      {H59+34, "RRing3DIP"},  {H59+35, "RRing4FingerTip"},
    {H59+36, "RPinky1Knuckles"}, {H59+37, "RPinky2PIP"},     {H59+38, "RPinky3DIP"}, {H59+39, "RPinky4FingerTip"},
    // Background
    {59, "Background"},
};
const auto H65 = 25;
const std::map<unsigned int, std::string> POSE_BODY_65_BODY_PARTS {
    // Body
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "MidHip"},
    {9,  "RHip"},
    {10, "RKnee"},
    {11, "RAnkle"},
    {12, "LHip"},
    {13, "LKnee"},
    {14, "LAnkle"},
    {15, "REye"},
    {16, "LEye"},
    {17, "REar"},
    {18, "LEar"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
    // Left hand
    {H65+0, "LThumb1CMC"},       {H65+1, "LThumb2Knuckles"}, {H65+2, "LThumb3IP"},   {H65+3, "LThumb4FingerTip"},
    {H65+4, "LIndex1Knuckles"},  {H65+5, "LIndex2PIP"},      {H65+6, "LIndex3DIP"},  {H65+7, "LIndex4FingerTip"},
    {H65+8, "LMiddle1Knuckles"}, {H65+9, "LMiddle2PIP"},     {H65+10, "LMiddle3DIP"},{H65+11, "LMiddle4FingerTip"},
    {H65+12, "LRing1Knuckles"},  {H65+13, "LRing2PIP"},      {H65+14, "LRing3DIP"},  {H65+15, "LRing4FingerTip"},
    {H65+16, "LPinky1Knuckles"}, {H65+17, "LPinky2PIP"},     {H65+18, "LPinky3DIP"}, {H65+19, "LPinky4FingerTip"},
    // Right hand
    {H65+20, "RThumb1CMC"},      {H65+21, "RThumb2Knuckles"},{H65+22, "RThumb3IP"},  {H65+23, "RThumb4FingerTip"},
    {H65+24, "RIndex1Knuckles"}, {H65+25, "RIndex2PIP"},     {H65+26, "RIndex3DIP"}, {H65+27, "RIndex4FingerTip"},
    {H65+28, "RMiddle1Knuckles"},{H65+29, "RMiddle2PIP"},    {H65+30, "RMiddle3DIP"},{H65+31, "RMiddle4FingerTip"},
    {H65+32, "RRing1Knuckles"},  {H65+33, "RRing2PIP"},      {H65+34, "RRing3DIP"},  {H65+35, "RRing4FingerTip"},
    {H65+36, "RPinky1Knuckles"}, {H65+37, "RPinky2PIP"},     {H65+38, "RPinky3DIP"}, {H65+39, "RPinky4FingerTip"},
    // Left hand
    {25, "LThumb1CMC"},         {26, "LThumb2Knuckles"},{27, "LThumb3IP"},  {28, "LThumb4FingerTip"},
    {29, "LIndex1Knuckles"},    {30, "LIndex2PIP"},     {31, "LIndex3DIP"}, {32, "LIndex4FingerTip"},
    {33, "LMiddle1Knuckles"},   {34, "LMiddle2PIP"},    {35, "LMiddle3DIP"},{36, "LMiddle4FingerTip"},
    {37, "LRing1Knuckles"},     {38, "LRing2PIP"},      {39, "LRing3DIP"},  {40, "LRing4FingerTip"},
    {41, "LPinky1Knuckles"},    {42, "LPinky2PIP"},     {43, "LPinky3DIP"}, {44, "LPinky4FingerTip"},
    // Right hand
    {45, "RThumb1CMC"},         {46, "RThumb2Knuckles"},{47, "RThumb3IP"},  {48, "RThumb4FingerTip"},
    {49, "RIndex1Knuckles"},    {50, "RIndex2PIP"},     {51, "RIndex3DIP"}, {52, "RIndex4FingerTip"},
    {53, "RMiddle1Knuckles"},   {54, "RMiddle2PIP"},    {55, "RMiddle3DIP"},{56, "RMiddle4FingerTip"},
    {57, "RRing1Knuckles"},     {58, "RRing2PIP"},      {59, "RRing3DIP"},  {60, "RRing4FingerTip"},
    {61, "RPinky1Knuckles"},    {62, "RPinky2PIP"},     {63, "RPinky3DIP"}, {64, "RPinky4FingerTip"},
    // Background
    {65, "Background"},
};
const auto F95 = 25;
const std::map<unsigned int, std::string> POSE_BODY_95_BODY_PARTS {
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "RKnee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17, "UpperNeck"},
    {18, "HeadTop"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
    // Face
    {F95+0, "FaceContour0"},   {F95+1, "FaceContour1"},   {F95+2, "FaceContour2"},   {F95+3, "FaceContour3"},   {F95+4, "FaceContour4"},   {F95+5, "FaceContour5"},   // Contour 1/3
    {F95+6, "FaceContour6"},   {F95+7, "FaceContour7"},   {F95+8, "FaceContour8"},   {F95+9, "FaceContour9"},   {F95+10, "FaceContour10"}, {F95+11, "FaceContour11"}, // Contour 2/3
    {F95+12, "FaceContour12"}, {F95+13, "FaceContour13"}, {F95+14, "FaceContour14"}, {F95+15, "FaceContour15"}, {F95+16, "FaceContour16"},                            // Contour 3/3
    {F95+17, "REyeBrow0"},  {F95+18, "REyeBrow1"},  {F95+19, "REyeBrow2"},  {F95+20, "REyeBrow3"},  {F95+21, "REyeBrow4"}, // Right eyebrow
    {F95+22, "LEyeBrow4"},  {F95+23, "LEyeBrow3"},  {F95+24, "LEyeBrow2"},  {F95+25, "LEyeBrow1"},  {F95+26, "LEyeBrow0"}, // Left eyebrow
    {F95+27, "NoseUpper0"}, {F95+28, "NoseUpper1"}, {F95+29, "NoseUpper2"}, {F95+30, "NoseUpper3"}, // Upper nose
    {F95+31, "NoseLower0"}, {F95+32, "NoseLower1"}, {F95+33, "NoseLower2"}, {F95+34, "NoseLower3"}, {F95+35, "NoseLower4"}, // Lower nose
    {F95+36, "REye0"}, {F95+37, "REye1"}, {F95+38, "REye2"}, {F95+39, "REye3"}, {F95+40, "REye4"}, {F95+41, "REye5"}, // Right eye
    {F95+42, "LEye0"}, {F95+43, "LEye1"}, {F95+44, "LEye2"}, {F95+45, "LEye3"}, {F95+46, "LEye4"}, {F95+47, "LEye5"}, // Left eye
    {F95+48, "OMouth0"}, {F95+49, "OMouth1"}, {F95+50, "OMouth2"}, {F95+51, "OMouth3"}, {F95+52, "OMouth4"}, {F95+53, "OMouth5"}, // Outer mouth 1/2
    {F95+54, "OMouth6"}, {F95+55, "OMouth7"}, {F95+56, "OMouth8"}, {F95+57, "OMouth9"}, {F95+58, "OMouth10"}, {F95+59, "OMouth11"}, // Outer mouth 2/2
    {F95+60, "IMouth0"}, {F95+61, "IMouth1"}, {F95+62, "IMouth2"}, {F95+63, "IMouth3"}, {F95+64, "IMouth4"}, {F95+65, "IMouth5"}, {F95+66, "IMouth6"}, {F95+67, "IMouth7"}, // Inner mouth
    {F95+68, "RPupil"}, {F95+69, "LPupil"}, // Pupils
};
const auto H135 = 25;
const auto F135 = H135+40;
const std::map<unsigned int, std::string> POSE_BODY_135_BODY_PARTS {
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "RKnee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17, "UpperNeck"},
    {18, "HeadTop"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
    // Left hand
    {H135+0, "LThumb1CMC"},       {H135+1, "LThumb2Knuckles"}, {H135+2, "LThumb3IP"},   {H135+3, "LThumb4FingerTip"},
    {H135+4, "LIndex1Knuckles"},  {H135+5, "LIndex2PIP"},      {H135+6, "LIndex3DIP"},  {H135+7, "LIndex4FingerTip"},
    {H135+8, "LMiddle1Knuckles"}, {H135+9, "LMiddle2PIP"},     {H135+10, "LMiddle3DIP"},{H135+11, "LMiddle4FingerTip"},
    {H135+12, "LRing1Knuckles"},  {H135+13, "LRing2PIP"},      {H135+14, "LRing3DIP"},  {H135+15, "LRing4FingerTip"},
    {H135+16, "LPinky1Knuckles"}, {H135+17, "LPinky2PIP"},     {H135+18, "LPinky3DIP"}, {H135+19, "LPinky4FingerTip"},
    // Right hand
    {H135+20, "RThumb1CMC"},      {H135+21, "RThumb2Knuckles"},{H135+22, "RThumb3IP"},  {H135+23, "RThumb4FingerTip"},
    {H135+24, "RIndex1Knuckles"}, {H135+25, "RIndex2PIP"},     {H135+26, "RIndex3DIP"}, {H135+27, "RIndex4FingerTip"},
    {H135+28, "RMiddle1Knuckles"},{H135+29, "RMiddle2PIP"},    {H135+30, "RMiddle3DIP"},{H135+31, "RMiddle4FingerTip"},
    {H135+32, "RRing1Knuckles"},  {H135+33, "RRing2PIP"},      {H135+34, "RRing3DIP"},  {H135+35, "RRing4FingerTip"},
    {H135+36, "RPinky1Knuckles"}, {H135+37, "RPinky2PIP"},     {H135+38, "RPinky3DIP"}, {H135+39, "RPinky4FingerTip"},
    // Face
    {F135+0, "FaceContour0"},   {F135+1, "FaceContour1"},   {F135+2, "FaceContour2"},   {F135+3, "FaceContour3"},   {F135+4, "FaceContour4"},   {F135+5, "FaceContour5"},   // Contour 1/3
    {F135+6, "FaceContour6"},   {F135+7, "FaceContour7"},   {F135+8, "FaceContour8"},   {F135+9, "FaceContour9"},   {F135+10, "FaceContour10"}, {F135+11, "FaceContour11"}, // Contour 2/3
    {F135+12, "FaceContour12"}, {F135+13, "FaceContour13"}, {F135+14, "FaceContour14"}, {F135+15, "FaceContour15"}, {F135+16, "FaceContour16"},                             // Contour 3/3
    {F135+17, "REyeBrow0"},  {F135+18, "REyeBrow1"},  {F135+19, "REyeBrow2"},  {F135+20, "REyeBrow3"},  {F135+21, "REyeBrow4"}, // Right eyebrow
    {F135+22, "LEyeBrow4"},  {F135+23, "LEyeBrow3"},  {F135+24, "LEyeBrow2"},  {F135+25, "LEyeBrow1"},  {F135+26, "LEyeBrow0"}, // Left eyebrow
    {F135+27, "NoseUpper0"}, {F135+28, "NoseUpper1"}, {F135+29, "NoseUpper2"}, {F135+30, "NoseUpper3"}, // Upper nose
    {F135+31, "NoseLower0"}, {F135+32, "NoseLower1"}, {F135+33, "NoseLower2"}, {F135+34, "NoseLower3"}, {F135+35, "NoseLower4"}, // Lower nose
    {F135+36, "REye0"}, {F135+37, "REye1"}, {F135+38, "REye2"}, {F135+39, "REye3"}, {F135+40, "REye4"}, {F135+41, "REye5"}, // Right eye
    {F135+42, "LEye0"}, {F135+43, "LEye1"}, {F135+44, "LEye2"}, {F135+45, "LEye3"}, {F135+46, "LEye4"}, {F135+47, "LEye5"}, // Left eye
    {F135+48, "OMouth0"}, {F135+49, "OMouth1"}, {F135+50, "OMouth2"}, {F135+51, "OMouth3"}, {F135+52, "OMouth4"}, {F135+53, "OMouth5"}, // Outer mouth 1/2
    {F135+54, "OMouth6"}, {F135+55, "OMouth7"}, {F135+56, "OMouth8"}, {F135+57, "OMouth9"}, {F135+58, "OMouth10"}, {F135+59, "OMouth11"}, // Outer mouth 2/2
    {F135+60, "IMouth0"}, {F135+61, "IMouth1"}, {F135+62, "IMouth2"}, {F135+63, "IMouth3"}, {F135+64, "IMouth4"}, {F135+65, "IMouth5"}, {F135+66, "IMouth6"}, {F135+67, "IMouth7"}, // Inner mouth
    {F135+68, "RPupil"}, {F135+69, "LPupil"}, // Pupils
};
// Hand legend:
//     - Thumb:
//         - Carpometacarpal Joints (CMC)
//         - Interphalangeal Joints (IP)
//     - Other fingers:
//         - Knuckles or Metacarpophalangeal Joints (MCP)
//         - PIP (Proximal Interphalangeal Joints)
//         - DIP (Distal Interphalangeal Joints)
//     - All fingers:
//         - Fingertips
// More information: Page 6 of http://www.mccc.edu/~behrensb/documents/TheHandbig.pdf
const std::map<unsigned int, std::string> CAR_12_PARTS {
    {0,  "FRWheel"},
    {1,  "FLWheel"},
    {2,  "BRWheel"},
    {3,  "BLWheel"},
    {4,  "FRLight"},
    {5,  "FLLight"},
    {6,  "BRLight"},
    {7,  "BLLight"},
    {8,  "FRTop"},
    {9,  "FLTop"},
    {10, "BRTop"},
    {11, "BLTop"},
    {12, "Background"},
};
const std::map<unsigned int, std::string> CAR_22_PARTS {
    {0,  "FLWheel"},
    {1,  "BLWheel"},
    {2,  "FRWheel"},
    {3,  "BRWheel"},
    {4,  "FRFogLight"},
    {5,  "FLFogLight"},
    {6,  "FRLight"},
    {7,  "FLLight"},
    {8,  "Grilles"},
    {9,  "FBumper"},
    {10, "LMirror"},
    {11, "RMirror"},
    {12, "FRTop"},
    {13, "FLTop"},
    {14, "BLTop"},
    {15, "BRTop"},
    {16, "BLLight"},
    {17, "BRLight"},
    {18, "Trunk"},
    {19, "BBumper"},
    {20, "BLCorner"},
    {21, "BRCorner"},
    {22, "Tailpipe"},
    {23, "Background"},
};





    // Auxiliary functions
    const auto NUMBER_MODELS = 15; // How many are the same
    int poseModelToIndex(const PoseModel poseModel)
    {
        const auto numberBodyParts = getNumberBodyParts(poseModel);
        if (poseModel == PoseModel::COCO_19b)
            return 3;
        else if (poseModel == PoseModel::COCO_19_V2)
            return 4;
        else if (poseModel == PoseModel::CAR_12)
            return 7;
        else if (poseModel == PoseModel::COCO_25E || poseModel == PoseModel::COCO_25_17E)
            return 8;
        else if (poseModel == PoseModel::CAR_22)
            return 10;
        else if (poseModel == PoseModel::COCO_19E)
            return 11;
        else if (poseModel == PoseModel::COCO_25B_23 || poseModel == PoseModel::COCO_25B_17 || poseModel == PoseModel::MPII_25B_16 || poseModel == PoseModel::PT_25B_15)
            return 12;
        else if (numberBodyParts == 18)
            return 0;
        else if (numberBodyParts == 19)
            return 1;
        else if (numberBodyParts == 23)
            return 9;
        else if (numberBodyParts == 25)
            return 5;
        else if (numberBodyParts == 59) // COCO + Hand
            return 2;
        else if (numberBodyParts == 65) // COCO + Foot + Hand
            return 6;
        else if (numberBodyParts == 95) // COCO + MPII + Foot + Face
            return 13;
        else if (numberBodyParts == 135) // COCO + MPII + Foot + Hands + Face
            return 14;
        // else
        throw std::runtime_error{"PoseModel does not have corresponding index yet."
                                 + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return 0;
    }





    // Parameters and functions to change if new PoseModel
    const std::array<int, (int)PoseModel::Size> NUMBER_BODY_PARTS{
        18, 18, 19, 19, 59, 59, 59, 19, 19, 25, 25, 65, 12, 25, 25, 23, 23, 22, 19, 25,25,25,25, 95,95,95,95, 135,135,135,135,135,135,135};

    const std::array<int, (int)PoseModel::Size> NUMBER_PARTS_LMDB{
        17, 19, 17, 19, 59, 17, 59, 17, 17, 23, 17, 42, 14, 23, 17, 23, 17, 22, 17, 23,17,16,15, 23,17,16,70,  23, 17, 16, 21, 42, 70,135};

    const std::array<std::vector<std::vector<int>>, (int)PoseModel::Size> LMDB_TO_OPENPOSE_KEYPOINTS{
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}                   // COCO_18
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {9},{10},{11}, {12},{13},{14},  {15},{16},{17},{18}                // DOME_18
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18}         // DOME_19
        },
        std::vector<std::vector<int>>{                                                                              // DOME_59
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18},        // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // COCO_59_17
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3},         // Body
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{},                                        // Left hand
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}                                         // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // MPII_59
            {},{}, {2},{3},{4},  {5},{6},{7},  {},  {9},{10},{11},  {12},{13},{14},  {},{},{},{},                   // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_b
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_V2
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {17},{18},{19},{20},{21},{22} // COCO_25
            // {},{5,6}, {},{},{}, {},{},{}, {}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22}    // COCO_25
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17
        },
        std::vector<std::vector<int>>{                                                                              // MPII_65_42
            {},{}, {},{},{21}, {},{},{0}, {}, {},{},{}, {},{},{}, {},{},{},{}, {},{},{},{},{},{},                   // Body
            {1},{2},{3},{4}, {5},{6},{7},{8}, {9},{10},{11},{12}, {13},{14},{15},{16}, {17},{18},{19},{20},         // Left hand
            {22},{23},{24},{25}, {26},{27},{28},{29}, {30},{31},{32},{33}, {34},{35},{36},{37}, {38},{39},{40},{41} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // CAR_12
            {0},{1},{2},{3},{4},{5},{6},{7},{9},{10},{11},{12}                                                      // 8 and 13 are always empty
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {17},{18},{19},{20},{21},{22} // COCO_25E
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17E
        },
        std::vector<std::vector<int>>{
            {0}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {17},{18},{19},{20},{21},{22} // COCO_23
        },
        std::vector<std::vector<int>>{
            {0}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{}      // COCO_23_17
        },
        std::vector<std::vector<int>>{                                                                              // CAR_22
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}//,{22}
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19E
        },
        // COCO + MPII + Foot
        std::vector<std::vector<int>>{                                                                              // COCO_25B_23
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{},{},{17},{18},{19},{20},{21},{22}
        },
        std::vector<std::vector<int>>{                                                                              // COCO_25B_17
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{},{},{},{},{},{},{},{}
        },
        std::vector<std::vector<int>>{                                                                              // MPII_25B_16
            // {},{},{},{},{},{13},{12},{14},{11},{15},{10},{3},{2},{4},{1},{5},{0},{8},{9},{},{},{},{},{},{} // Uses all MPII keypoints
            {},{},{},{},{},{13},{12},{  },{  },{  },{  },{ },{ },{ },{ },{ },{ },{8},{9},{},{},{},{},{},{} // Does not use all MPII keypoints
        },
        std::vector<std::vector<int>>{                                                                              // PT_25B_15
            {13},{},{},{},{},{9},{8},{10},{7},{11},{6},{3},{2},{4},{1},{5},{0},{12},{14},{},{},{},{},{},{}
        },
        // COCO + MPII + Foot + Face
        std::vector<std::vector<int>>{                                                                              // COCO_95_23
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {},{},                                                                                                  // MPII
            {17},{18},{19},{20},{21},{22},                                                                          // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // COCO_95_17
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // MPII_95_16
            // {},{},{},{},{},{13},{12},{14},{11},{15},{10},{3},{2},{4},{1},{5},{0},{8},{9},{},{},{},{},{},{}
            {},{},{},{},{},{13},{12},{},{},{},{},{},{},{},{},{},{},                                                 // COCO
            {8},{9},                                                                                                // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // FACE_95_70
            // {27},
            // {F95+27,F95+28,F95+29,F95+30,F95+31,F95+32,F95+34,F95+35},
            {27,28,29,30,31,32,34,35},
            // {42},
            // {F95+42,F95+43,F95+44,F95+45,F95+46,F95+47},
            {42,43,44,45,46,47},
            // {36}, // COCO 1/2
            // {F95+36,F95+37,F95+38,F95+39,F95+40,F95+41}, // COCO 1/2
            {36,37,38,39,40,41}, // COCO 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{}, // COCO 2/2
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}, // Face contour
            {17},{18},{19},{20},{21},{22},{23},{24},{25},{26}, // Eyebrows
            {27},{28},{29},{30},{31},{32},{33},{34},{35}, // Nose
            {36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47}, // Eyes
            {48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60},{61},{62},{63},{64},{65},{66},{67}, // Mouth
            {68},{69} // Pupils
        },
        // COCO + MPII + Foot + Hands + Face
        std::vector<std::vector<int>>{                                                                              // COCO_135_23
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {},{},                                                                                                  // MPII
            {17},{18},{19},{20},{21},{22},                                                                          // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // COCO_135_17
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // MPII_135_16
            // {},{},{},{},{},{13},{12},{14},{11},{15},{10},{3},{2},{4},{1},{5},{0},{8},{9},{},{},{},{},{},{}
            {},{},{},{},{},{13},{12},{},{},{},{},{},{},{},{},{},{},                                                 // COCO
            {8},{9},                                                                                                // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // HAND_135_21
            {},{},{},{},{},{},{},{},{},{},{0},{},{},{},{},{},{},                                                    // COCO
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            {1},{2},{3},{4},{5},{6},{7},{8},{9},{10},   {11},{12},{13},{14},{15},{16},{17},{18},{19},{20},          // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // HAND_135_42
            {},{},{},{},{},{},{},{},{},{0},{21},{},{},{},{},{},{},                                                  // COCO
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {1},{2},{3},{4},{5},{6},{7},{8},{9},{10},   {11},{12},{13},{14},{15},{16},{17},{18},{19},{20},          // Left hand
            {22},{23},{24},{25},{26},{27},{28},{29},{30},{31},   {32},{33},{34},{35},{36},{37},{38},{39},{40},{41}, // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // FACE_135_70
            // {27},
            // {F95+27,F95+28,F95+29,F95+30,F95+31,F95+32,F95+34,F95+35},
            {27,28,29,30,31,32,34,35},
            // {42},
            // {F95+42,F95+43,F95+44,F95+45,F95+46,F95+47},
            {42,43,44,45,46,47},
            // {36}, // COCO 1/2
            // {F95+36,F95+37,F95+38,F95+39,F95+40,F95+41}, // COCO 1/2
            {36,37,38,39,40,41}, // COCO 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{}, // COCO 2/2
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Right hand
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}, // Face contour
            {17},{18},{19},{20},{21},{22},{23},{24},{25},{26}, // Eyebrows
            {27},{28},{29},{30},{31},{32},{33},{34},{35}, // Nose
            {36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47}, // Eyes
            {48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60},{61},{62},{63},{64},{65},{66},{67}, // Mouth
            // {68},{69} // Pupils
            {36,37,38,39,40,41},{42,43,44,45,46,47} // Pupils (our training sets do not have them)
        },
        std::vector<std::vector<int>>{                                                                              // DOME_135
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {17},{18},                                                                                              // MPII
            {19},{20},{21},{22},{23},{24},                                                                          // Foot
            {25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},    // Left hand
            {45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60},{61},{62},{63},{64},    // Right hand
            {65},{66},{67},{68},{69},{70},{71},{72},{73},{74},{75},{76},{77},{78},{79},{80},{81}, // Face contour
            {82},{83},{84},{85},{86},{87},{88},{89},{90},{91}, // Eyebrows
            {92},{93},{94},{95},{96},{97},{98},{99},{100}, // Nose
            {101},{102},{103},{104},{105},{106},{107},{108},{109},{110},{111},{112}, // Eyes
            {113},{114},{115},{116},{117},{118},{119},{120},{121},{122},{123},{124},{125},{126},{127},{128},{129},{130},{131},{132}, // Mouth
            {133},{134} // Pupils
        },
    };

    // Idea: Keypoint that is empty will be masked out (including the PAFs that use it)
    // For simplicity: Same than LMDB_TO_OPENPOSE_KEYPOINTS unless masked keypoints.
    // But the only thing that matters is whether it is empty (e.g., {}) or not (e.g., {1} or {99,92} or {-1}).
    // E.g., for foot dataset to avoid overfitting on duplicated body keypoints
    const std::array<std::vector<std::vector<int>>, (int)PoseModel::Size> CHANNELS_TO_MASK{
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}                   // COCO_18
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {9},{10},{11}, {12},{13},{14},  {15},{16},{17},{18}                // DOME_18
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18}         // DOME_19
        },
        std::vector<std::vector<int>>{                                                                              // DOME_59
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18},        // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // COCO_59_17
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3},         // Body
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{},                                        // Left hand
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}                                         // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // MPII_59
            {},{}, {2},{3},{4},  {5},{6},{7},  {},  {9},{10},{11},  {12},{13},{14},  {},{},{},{},                   // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_b
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_V2
        },
        std::vector<std::vector<int>>{
            {},{5,6}, {},{},{}, {},{},{}, {11,12}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22}// COCO_25
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17
        },
        std::vector<std::vector<int>>{                                                                              // MPII_65_42
            {},{}, {},{},{21}, {},{},{0}, {}, {},{},{}, {},{},{}, {},{},{},{}, {},{},{},{},{},{},                   // Body
            {1},{2},{3},{4}, {5},{6},{7},{8}, {9},{10},{11},{12}, {13},{14},{15},{16}, {17},{18},{19},{20},         // Left hand
            {22},{23},{24},{25}, {26},{27},{28},{29}, {30},{31},{32},{33}, {34},{35},{36},{37}, {38},{39},{40},{41} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // CAR_12
            {0},{1},{2},{3},{4},{5},{6},{7},{9},{10},{11},{12}                                                      // 8 and 13 are always empty
        },
        std::vector<std::vector<int>>{
            {},{5,6}, {},{},{}, {},{},{}, {11,12}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22}// COCO_25E
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17E
        },
        std::vector<std::vector<int>>{
            {}, {},{},{}, {},{},{}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22}              // COCO_23
        },
        std::vector<std::vector<int>>{
            {0}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{}      // COCO_23_17
        },
        std::vector<std::vector<int>>{                                                                              // CAR_22
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}//,{22}
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19E
        },
        std::vector<std::vector<int>>{                                                                              // COCO_25B_23
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{-1},{-1},{17},{18},{19},{20},{21},{22}
            // { },{ },{ },{ },{ },{ },{ },{ },{ },{ },{  },{  },{  },{  },{  },{15},{16},{},{},{17},{18},{19},{20},{21},{22}
        },
        std::vector<std::vector<int>>{                                                                              // COCO_25B_17
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{-1},{-1},{},{},{},{},{},{}
        },
        std::vector<std::vector<int>>{                                                                              // MPII_25B_16
            // {},{},{},{},{},{13},{12},{14},{11},{15},{10},{3},{2},{4},{1},{5},{0},{8},{9},{},{},{},{},{},{} // Uses all MPII keypoints
            {},{},{},{},{},{13},{12},{  },{  },{  },{  },{ },{ },{ },{ },{ },{ },{8},{9},{},{},{},{},{},{} // Does not use all MPII keypoints
        },
        std::vector<std::vector<int>>{                                                                              // PT_25B_15
            {13},{},{},{},{},{9},{8},{10},{7},{11},{6},{3},{2},{4},{1},{5},{0},{12},{14},{},{},{},{},{},{}
        },
        // COCO + MPII + Foot + Face
        std::vector<std::vector<int>>{                                                                              // COCO_95_23
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {-1},{-1},                                                                                              // MPII
            {17},{18},{19},{20},{21},{22},                                                                          // Foot
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},// Face 1/2
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // COCO_95_17
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {-1},{-1},                                                                                              // MPII
            {-1},{-1},{-1},{-1},{-1},{-1},                                                                          // Foot
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},// Face 1/2
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // MPII_95_16
            // {},{},{},{},{},{13},{12},{14},{11},{15},{10},{3},{2},{4},{1},{5},{0},{8},{9},{},{},{},{},{},{}
            {},{},{},{},{},{13},{12},{},{},{},{},{},{},{},{},{},{},                                                 // COCO
            {8},{9},                                                                                                // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // FACE_95_70
            {-1},{-1},{-1},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                               // COCO
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}, // Face contour
            {17},{18},{19},{20},{21},{22},{23},{24},{25},{26}, // Eyebrows
            {27},{28},{29},{30},{31},{32},{33},{34},{35}, // Nose
            {36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47}, // Eyes
            {48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60},{61},{62},{63},{64},{65},{66},{67}, // Mouth
            {68},{69} // Pupils
        },
        // COCO + MPII + Foot + Hands + Face
        std::vector<std::vector<int>>{                                                                              // COCO_135_23
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {-1},{-1},                                                                                              // MPII
            {17},{18},{19},{20},{21},{22},                                                                          // Foot
            // {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            // {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Right hand
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},    // Left hand
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},    // Right hand
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},// Face 1/2
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // COCO_135_17
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {-1},{-1},                                                                                              // MPII
            {-1},{-1},{-1},{-1},{-1},{-1},                                                                          // Foot
            // {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            // {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Right hand
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},    // Left hand
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},    // Right hand
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},// Face 1/2
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // MPII_135_16
            // {},{},{},{},{},{13},{12},{14},{11},{15},{10},{3},{2},{4},{1},{5},{0},{8},{9},{},{},{},{},{},{}
            {},{},{},{},{},{13},{12},{},{},{},{},{},{},{},{},{},{},                                                 // COCO
            {8},{9},                                                                                                // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Left hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},                                            // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // HAND_135_21
            {},{},{},{},{},{},{},{},{},{-1},{0},{},{},{},{},{},{},                                                  // COCO
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},   {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}, // Left hand
            {1},{2},{3},{4},{5},{6},{7},{8},{9},{10},   {11},{12},{13},{14},{15},{16},{17},{18},{19},{20},          // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // HAND_135_42
            {},{},{},{},{},{},{},{},{},{0},{21},{},{},{},{},{},{},                                                  // COCO
            {},{},                                                                                                  // MPII
            {},{},{},{},{},{},                                                                                      // Foot
            {1},{2},{3},{4},{5},{6},{7},{8},{9},{10},   {11},{12},{13},{14},{15},{16},{17},{18},{19},{20},          // Left hand
            {22},{23},{24},{25},{26},{27},{28},{29},{30},{31},   {32},{33},{34},{35},{36},{37},{38},{39},{40},{41}, // Right hand
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},// Face 1/2
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}// Face 2/2
        },
        std::vector<std::vector<int>>{                                                                              // DOME_135
            {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},                             // COCO
            {17},{18},                                                                                              // MPII
            {19},{20},{21},{22},{23},{24},                                                                          // Foot
            {25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},    // Left hand
            {45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60},{61},{62},{63},{64},    // Right hand
            {65},{66},{67},{68},{69},{70},{71},{72},{73},{74},{75},{76},{77},{78},{79},{80},{81}, // Face contour
            {82},{83},{84},{85},{86},{87},{88},{89},{90},{91}, // Eyebrows
            {92},{93},{94},{95},{96},{97},{98},{99},{100}, // Nose
            {101},{102},{103},{104},{105},{106},{107},{108},{109},{110},{111},{112}, // Eyes
            {113},{114},{115},{116},{117},{118},{119},{120},{121},{122},{123},{124},{125},{126},{127},{128},{129},{130},{131},{132}, // Mouth
            {133},{134} // Pupils
        },
    };

    std::pair<PoseModel,PoseCategory> flagsToPoseModel(const std::string& poseModeString)
    {
        // COCO
        if (poseModeString == "COCO_18")
            return std::make_pair(PoseModel::COCO_18, PoseCategory::COCO);
        else if (poseModeString == "COCO_19")
            return std::make_pair(PoseModel::COCO_19, PoseCategory::COCO);
        else if (poseModeString == "COCO_19b")
            return std::make_pair(PoseModel::COCO_19b, PoseCategory::COCO);
        else if (poseModeString == "COCO_19E")
            return std::make_pair(PoseModel::COCO_19E, PoseCategory::COCO);
        else if (poseModeString == "COCO_19_V2")
            return std::make_pair(PoseModel::COCO_19_V2, PoseCategory::COCO);
        else if (poseModeString == "COCO_23")
            return std::make_pair(PoseModel::COCO_23, PoseCategory::COCO);
        else if (poseModeString == "COCO_23_17")
            return std::make_pair(PoseModel::COCO_23_17, PoseCategory::COCO);
        else if (poseModeString == "COCO_25")
            return std::make_pair(PoseModel::COCO_25, PoseCategory::COCO);
        else if (poseModeString == "COCO_25_17")
            return std::make_pair(PoseModel::COCO_25_17, PoseCategory::COCO);
        else if (poseModeString == "COCO_25E")
            return std::make_pair(PoseModel::COCO_25E, PoseCategory::COCO);
        else if (poseModeString == "COCO_25_17E")
            return std::make_pair(PoseModel::COCO_25_17E, PoseCategory::COCO);
        else if (poseModeString == "COCO_59_17")
            return std::make_pair(PoseModel::COCO_59_17, PoseCategory::COCO);
        else if (poseModeString == "COCO_25B_17")
            return std::make_pair(PoseModel::COCO_25B_17, PoseCategory::COCO);
        else if (poseModeString == "COCO_25B_23")
            return std::make_pair(PoseModel::COCO_25B_23, PoseCategory::COCO);
        else if (poseModeString == "COCO_95_17")
            return std::make_pair(PoseModel::COCO_95_17, PoseCategory::COCO);
        else if (poseModeString == "COCO_95_23")
            return std::make_pair(PoseModel::COCO_95_23, PoseCategory::COCO);
        else if (poseModeString == "COCO_135_17")
            return std::make_pair(PoseModel::COCO_135_17, PoseCategory::COCO);
        else if (poseModeString == "COCO_135_23")
            return std::make_pair(PoseModel::COCO_135_23, PoseCategory::COCO);
        // Dome
        else if (poseModeString == "DOME_18")
            return std::make_pair(PoseModel::DOME_18, PoseCategory::DOME);
        else if (poseModeString == "DOME_19")
            return std::make_pair(PoseModel::DOME_19, PoseCategory::DOME);
        else if (poseModeString == "DOME_59")
            return std::make_pair(PoseModel::DOME_59, PoseCategory::DOME);
        else if (poseModeString == "DOME_135")
            return std::make_pair(PoseModel::DOME_135, PoseCategory::DOME);
        // MPII
        else if (poseModeString == "MPII_25B_16")
            return std::make_pair(PoseModel::MPII_25B_16, PoseCategory::MPII);
        else if (poseModeString == "MPII_95_16")
            return std::make_pair(PoseModel::MPII_95_16, PoseCategory::MPII);
        else if (poseModeString == "MPII_135_16")
            return std::make_pair(PoseModel::MPII_135_16, PoseCategory::MPII);
        // PT
        else if (poseModeString == "PT_25B_15")
            return std::make_pair(PoseModel::PT_25B_15, PoseCategory::PT);
        // Hand
        else if (poseModeString == "MPII_59")
            return std::make_pair(PoseModel::MPII_59, PoseCategory::HAND);
        else if (poseModeString == "MPII_65_42")
            return std::make_pair(PoseModel::MPII_65_42, PoseCategory::HAND);
        else if (poseModeString == "HAND_135_21")
            return std::make_pair(PoseModel::HAND_135_21, PoseCategory::HAND);
        else if (poseModeString == "HAND_135_42")
            return std::make_pair(PoseModel::HAND_135_42, PoseCategory::HAND);
        // Face
        else if (poseModeString == "FACE_95_70")
            return std::make_pair(PoseModel::FACE_95_70, PoseCategory::FACE);
        else if (poseModeString == "FACE_135_70")
            return std::make_pair(PoseModel::FACE_135_70, PoseCategory::FACE);
        // Car
        else if (poseModeString == "CAR_12")
            return std::make_pair(PoseModel::CAR_12, PoseCategory::CAR);
        else if (poseModeString == "CAR_22")
            return std::make_pair(PoseModel::CAR_22, PoseCategory::CAR);
        // Unknown
        throw std::runtime_error{"String (" + poseModeString
                                 + ") does not correspond to any model (COCO_18, DOME_18, ...)"
                                 + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return std::make_pair(PoseModel::COCO_18, PoseCategory::COCO);
    }





    // Parameters and functions to change if new number body parts
    const std::array<std::vector<std::array<int,2>>, NUMBER_MODELS> SWAP_LEFT_RIGHT_KEYPOINTS{
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{11,8},{12,9},{13,10},{15,14},{17,16}},                    // 18 (COCO_18, DOME_18)
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17}},                   // 19 (COCO_19(b), DOME_19)
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17},                    // 59 (DOME_59), COCO_59_17, MPII_59
                                       {19,39},{20,40},{21,41},{22,42},{23,43},{24,44},{25,45},{26,46},     // 2 fingers
                                       {27,47},{28,48},{29,49},{30,50},{31,51},{32,52},{33,53},{34,54},     // 2 fingers
                                       {35,55},{36,56},{37,57},{38,58}},                                    // 1 finger
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17}},                   // COCO_19b
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17}},                   // COCO_19_V2
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17},{19,22},{20,23},{21,24}}, // 25 (COCO_25, COCO_25_17)
        std::vector<std::array<int,2>>{{7,4},                                                               // 65 (MPII_65_42)
                                       {25,45},{26,46},{27,47},{28,48},{29,49},{30,50},{31,51},{32,52},     // 2 fingers
                                       {33,53},{34,54},{35,55},{36,56},{37,57},{38,58},{39,59},{40,60},     // 2 fingers
                                       {41,61},{42,62},{43,63},{44,64}},                                    // 1 finger
        std::vector<std::array<int,2>>{{0,1},{2,3},{4,5},{6,7},{8,9},{10,11}},                                      // CAR_12
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17},{19,22},{20,23},{21,24}}, // 25E (COCO_25E, COCO_25_17E)
        std::vector<std::array<int,2>>{{4,1},{5,2},{6,3},{10,7},{11,8}, {12,9}, {14,13},{16,15},{17,20},{18,21},{19,22}}, // 23 (COCO_23, COCO_23_17)
        std::vector<std::array<int,2>>{{0,2},{1,3},{4,5},{6,7},{10,11},{12,13},{14,15},{16,17},{20,21}},            // CAR_22
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17}},                   // COCO_19E
        std::vector<std::array<int,2>>{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12},{13,14},{15,16},{19,22},{20,23},{21,24}},// 25B (COCO_25B_23, COCO_25B_17, MPII_25B_16)
        // COCO + MPII + Foot + Face
        std::vector<std::array<int,2>>{                                                                             // 95 (COCO_95_23, COCO_95_17, MPII_95_16, FACE_95_70)
            {1,2},{3,4},{5,6},{7,8},{9,10},{11,12},{13,14},{15,16},{19,22},{20,23},{21,24}, // COCO + MPII + Foot
            {F95+0,F95+16},{F95+1,F95+15},{F95+2,F95+14},{F95+3,F95+13},{F95+4,F95+12},{F95+5,F95+11},{F95+6,F95+10},{F95+7,F95+9}, // Contour
            {F95+17,F95+26},{F95+18,F95+25},{F95+19,F95+24},{F95+20,F95+23},{F95+21,F95+22}, // Eyebrows
            {F95+31,F95+35},{F95+32,F95+34}, // Nose
            {F95+36,F95+45},{F95+37,F95+44},{F95+38,F95+43},{F95+39,F95+42},{F95+40,F95+47},{F95+41,F95+46}, // Eyes
            {F95+48,F95+54},{F95+49,F95+53},{F95+50,F95+52},{F95+55,F95+59},{F95+56,F95+58},{F95+60,F95+64},{F95+61,F95+63},{F95+65,F95+67}, // Mouth
            {F95+68,F95+69} // Pupils
        },
        // COCO + MPII + Foot + Hands + Face
        std::vector<std::array<int,2>>{                                                                             // 95 (COCO_95_23, COCO_95_17, MPII_95_16, FACE_95_70)
            {1,2},{3,4},{5,6},{7,8},{9,10},{11,12},{13,14},{15,16},{19,22},{20,23},{21,24}, // COCO + MPII + Foot
            {H135+0,H135+20},{H135+1,H135+21},{H135+2,H135+22},{H135+3,H135+23},{H135+4,H135+24},      // Hands 1/4
            {H135+5,H135+25},{H135+6,H135+26},{H135+7,H135+27},{H135+8,H135+28},{H135+9,H135+29},      // Hands 2/4
            {H135+10,H135+30},{H135+11,H135+31},{H135+12,H135+32},{H135+13,H135+33},{H135+14,H135+34}, // Hands 3/4
            {H135+15,H135+35},{H135+16,H135+36},{H135+17,H135+37},{H135+18,H135+38},{H135+19,H135+39}, // Hands 4/4
            {F135+0,F135+16},{F135+1,F135+15},{F135+2,F135+14},{F135+3,F135+13},{F135+4,F135+12},{F135+5,F135+11},{F135+6,F135+10},{F135+7,F135+9}, // Contour
            {F135+17,F135+26},{F135+18,F135+25},{F135+19,F135+24},{F135+20,F135+23},{F135+21,F135+22}, // Eyebrows
            {F135+31,F135+35},{F135+32,F135+34}, // Nose
            {F135+36,F135+45},{F135+37,F135+44},{F135+38,F135+43},{F135+39,F135+42},{F135+40,F135+47},{F135+41,F135+46}, // Eyes
            {F135+48,F135+54},{F135+49,F135+53},{F135+50,F135+52},{F135+55,F135+59},{F135+56,F135+58},{F135+60,F135+64},{F135+61,F135+63},{F135+65,F135+67}, // Mouth
            {F135+68,F135+69} // Pupils
        },
    };

    const std::array<std::vector<int>, NUMBER_MODELS> LABEL_MAP_A{
        std::vector<int>{1, 8,  9, 1,   11, 12, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  14, 15},                       // 18 (COCO_18, DOME_18)
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16},                       // 19 (COCO_19, DOME_19)
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16,                        // 59 (DOME_59), COCO_59_17, MPII_59
                         7,19,20,21, 7,23,24,25, 7,27,28,29, 7,31,32,33, 7,35,36,37, // Left hand
                         4,39,40,41, 4,43,44,45, 4,47,48,49, 4,51,52,53, 4,55,56,57},// Right hand
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16, 2, 5},                 // COCO_19b
        std::vector<int>{1,1,1,1,1,1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1},                                             // COCO_19_V2
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16, 14,19,14, 11,22,11},   // 25 (COCO_25, COCO_25_17)
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16, 14,19,14, 11,22,11,    // 65 (MPII_65_42)
                         7,25,26,27, 7,29,30,31, 7,33,34,35, 7,37,38,39, 7,41,42,43, // Left hand
                         4,45,46,47, 4,49,50,51, 4,53,54,55, 4,57,58,59, 4,61,62,63},// Right hand
        std::vector<int>{4, 4,4,0,4,8, 5,5,1,5,9},                                                                  // CAR_12
        std::vector<int>{                                                                                           // 25 (COCO_25E, COCO_25_17E)
            // Minimum spanning tree
            1,   1, 2, 3,   1, 5, 6,   8, 9,  10,    8, 12, 13,   1,  0, 15,  0, 16,   14,19,14, 11,22,11,
            // Redundant ones
            // Ears-shoulders, shoulders-hips, shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                   2, 5,            2, 5,             2, 5,         9, 12,       4,      11,        4, 7,        11, 14},
        std::vector<int>{                                                                                           // 23 (COCO_23, COCO_23_17)
            // Minimum spanning tree
                 0, 1, 2,   0, 4, 5,      7,   8,       10, 11,       0, 13,  0, 14,   12,17,12,  9,20,9,     1, 4,
            // Redundant ones
            // Ears-shoulders,   ears, hips,    shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                   1, 4,           15, 7,             1, 4,         7, 10,       3,       9,        3, 6,         9, 12},
        std::vector<int>{                                                                                           // CAR_22
        //   Wheels    Lights       Top      Tailpipe    Front     Mirrors     Back      Vertical   Back-light replacement
            0,1,3,2, 6,7,16,17, 12,13,14,15, /*1,3,*/ 6,7,6,7,6,7,  12,13, 16,17,16,17,  0,3,6,16,         6,7,3,20},
        std::vector<int>{                                                                                           // COCO_19E
            // Minimum spanning tree
            1,   1, 2, 3,   1, 5, 6,   8, 9,  10,    8, 12, 13,   1,  0, 15,  0, 16,   //14,19,14, 11,22,11,
            // Redundant ones
            // Ears-shoulders, shoulders-hips, shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                   2, 5,            2, 5,             2, 5,         9, 12,       4,      11,        4, 7,        },//11, 14},
        std::vector<int>{                                                                                           // 25B (COCO_25B_23, COCO_25B_17, MPII_25B_16)
            // Minimum spanning tree
            // |----------------------- COCO Body -----------------------|   |------ Foot ------|  | MPII |
                 0,0,1,2,   0,0,   5,6,   7, 8,    5, 6,   11,12,   13,14,   15,19,15,  16,22,16,    5, 5,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                    6, 6,       3,        3,4,              5, 6,         9,      9, 10,     11,    15},
            // Ignored: shoulders, hips-ankles (-0.1% COCO_23), ankles-small toes (-0.1% COCO_23)
            //              5,        11,12,                        15,16},
        // COCO + MPII + Foot + Face
        std::vector<int>{                                                                                           // 95 (COCO_95_23, COCO_95_17, MPII_95_16, FACE_95_70)
            // Minimum spanning tree
            // |----------------------- COCO Body -----------------------|   |------ Foot ------|  | MPII |
                 0,0,1,2,   0,0,   5,6,   7, 8,    5, 6,   11,12,   13,14,   15,19,15,  16,22,16,    5, 5,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                    6, 6,       3,        3,4,              5, 6,         9,      9, 10,     11,    15,
            // Ignored: shoulders, hips-ankles (-0.1% COCO_23), ankles-small toes (-0.1% COCO_23)
            //              5,        11,12,                        15,16,
               0, 2, 1, // COCO-Face (+1 extra, not 2)
               F95+0,F95+1,F95+2,F95+3,F95+4,F95+5,F95+6,F95+7,F95+8,F95+9, F95+10,F95+11,F95+12,F95+13,F95+14,F95+15, // Contour (+0)
               F95+0, F95+16,F95+17,F95+18,F95+19,F95+20,F95+21,F95+22,F95+23,F95+24,F95+25, // Countour-Eyebrow + Eyebrows (+1)
               F95+21,F95+22,F95+27,F95+28,F95+29,F95+30,F95+33,F95+32,F95+33,F95+34, // Eyebrow-Nose + Nose (+1)
               F95+27,F95+27,F95+36,F95+37,F95+38,F95+39,F95+40,F95+42,F95+43,F95+44,F95+45,F95+46, // Nose-Eyes + Eyes (+1)
               F95+33,F95+48,F95+49,F95+50,F95+51,F95+52,F95+53,F95+54,F95+55,F95+56,F95+57,F95+58, // Nose-Mouth + Outer Mouth (+0)
               F95+48,F95+54,F95+60,F95+61,F95+62,F95+63,F95+64,F95+65,F95+66, // Outer-Inner + Inner Mouth (+1)
               F95+36,F95+39,F95+42,F95+45, // Eyes-Pupils (+2)
        },
        // COCO + MPII + Foot + Hands + Face
        std::vector<int>{                                                                                           // 135 (COCO_135_23, COCO_135_17, MPII_135_16, HAND_135_42, FACE_135_70, DOME_135)
            // Minimum spanning tree
            // |----------------------- COCO Body -----------------------|   |------ Foot ------|  | MPII |
                 0,0,1,2,   0,0,   5,6,   7, 8,    5, 6,   11,12,   13,14,   15,19,15,  16,22,16,    5,17,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                      6,        3,        3,4,              5, 6,         9,      9, 10,     11,    15,
            // Ignored: shoulders, hips-ankles (-0.1% COCO_23), ankles-small toes (-0.1% COCO_23)
            //              5,        11,12,                        15,16,
            // Left hand
               9,H135+0,H135+1,H135+2, 9,H135+4,H135+5,H135+6, 9,H135+8,H135+9,H135+10, 9,H135+12,H135+13,H135+14, 9,H135+16,H135+17,H135+18,
            // Right hand
               10,H135+20,H135+21,H135+22, 10,H135+24,H135+25,H135+26, 10,H135+28,H135+29,H135+30, 10,H135+32,H135+33,H135+34, 10,H135+36,H135+37,H135+38,
            // Face
               0, 2, 1, // COCO-Face (+1 extra, not 2)
               F135+0,F135+1,F135+2,F135+3,F135+4,F135+5,F135+6,F135+7,F135+8,F135+9, F135+10,F135+11,F135+12,F135+13,F135+14,F135+15, // Contour (+0)
               F135+0, F135+16,F135+17,F135+18,F135+19,F135+20,F135+21,F135+22,F135+23,F135+24,F135+25, // Countour-Eyebrow + Eyebrows (+1)
               F135+21,F135+22,F135+27,F135+28,F135+29,F135+30,F135+33,F135+32,F135+33,F135+34, // Eyebrow-Nose + Nose (+1)
               F135+27,F135+27,F135+36,F135+37,F135+38,F135+39,F135+40,F135+42,F135+43,F135+44,F135+45,F135+46, // Nose-Eyes + Eyes (+1)
               F135+33,F135+48,F135+49,F135+50,F135+51,F135+52,F135+53,F135+54,F135+55,F135+56,F135+57,F135+58, // Nose-Mouth + Outer Mouth (+0)
               F135+48,F135+54,F135+60,F135+61,F135+62,F135+63,F135+64,F135+65,F135+66, // Outer-Inner + Inner Mouth (+1)
               F135+36,F135+39,F135+42,F135+45, // Eyes-Pupils (+2)
        },
    };

    const std::array<std::vector<int>, NUMBER_MODELS> LABEL_MAP_B{
        std::vector<int>{8, 9, 10, 11,  12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17},                       // 18 (COCO_18, DOME_18)
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18},                       // 19 (COCO_19, DOME_19)
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18,                        // 59 (DOME_59), COCO_59_17, MPII_59
                         19,20,21,22, 23,24,25,26, 27,28,29,30, 31,32,33,34, 35,36,37,38, // Left hand
                         39,40,41,42, 43,44,45,46, 47,48,49,50, 51,52,53,54, 55,56,57,58},// Right hand
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18, 9, 12},                // COCO_19b
        std::vector<int>{0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18},                                             // COCO_19_V2
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18, 19,20,21, 22,23,24},   // 25 (COCO_25, COCO_25_17)
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18, 19,20,21, 22,23,24,    // 65 (MPII_65_42)
                         25,26,27,28, 29,30,31,32, 33,34,35,36, 37,38,39,40, 41,42,43,44, // Left hand
                         45,46,47,48, 49,50,51,52, 53,54,55,56, 57,58,59,60, 61,62,63,64},// Right hand
        std::vector<int>{5, 6,0,2,8,10, 7,1,3,9,11},                                                                // CAR_12
        std::vector<int>{                                                                                           // 25 (COCO_25E, COCO_25_17E)
            // Minimum spanning tree
            8,   2, 3, 4,   5, 6, 7,   9, 10, 11,   12, 13, 14,   0, 15, 17, 16, 18,   19,20,21, 22,23,24,
            // Redundant ones
            // Ears-shoulders, shoulders-hips, shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                   17, 18,          9, 12,            4, 7,        11, 14,       7,      14,        9, 12,       23, 20},
        std::vector<int>{                                                                                           // 23 (COCO_23, COCO_23_17)
            // Minimum spanning tree
                 1, 2, 3,   4, 5, 6,       8,  9,       11, 12,      13, 15, 14, 16,   17,18,19, 20,21,22,     7, 10,
            // Redundant ones
            // Ears-shoulders,   ears, hips,    shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                   15, 16,         16, 10,            3, 6,         9, 12,       6,      12,        7, 10,       21, 18},
        std::vector<int>{                                                                                           // CAR_22
        //   Wheels    Lights       Top      Tailpipe    Front     Mirrors     Back      Vertical   Back-light replacement
            1,3,2,0, 7,16,17,6, 13,14,15,12,/*23,23,*/8,8,9,9,4,5,  11,10, 18,18,19,19, 7,17,12,14,      21,20,21,14},
        std::vector<int>{                                                                                           // COCO_19E
            // Minimum spanning tree
            8,   2, 3, 4,   5, 6, 7,   9, 10, 11,   12, 13, 14,   0, 15, 17, 16, 18,   //19,20,21, 22,23,24,
            // Redundant ones
            // Ears-shoulders, shoulders-hips, shoulders-wrists, hips-ankles, wrists,  ankles, wrists-hips, small toes-ankles)
                   17, 18,          9, 12,            4, 7,        11, 14,       7,      14,        9, 12,       },//23, 20},
        std::vector<int>{                                                                                           // 25B (COCO_25B_23, COCO_25B_17, MPII_25B_16)
            // Minimum spanning tree
            // |----------------------- COCO Body -----------------------|   |------ Foot ------|  | MPII |
                 1,2,3,4,   5,6,   7,8,   9,10,   11,12,   13,14,   15,16,   19,20,21,  22,23,24,   17,18,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                   17,18,       4,        5,6,              9,10,        10,      11,12,     12,    16},
            // Ignored: shoulders, hips-ankles (-0.1% COCO_23), small toes-ankles (-0.1% COCO_23)
            //              6,        15,16,                        20,23},
        // COCO + MPII + Foot + Face
        std::vector<int>{                                                                                           // 95 (COCO_95_23, COCO_95_17, MPII_95_16, FACE_95_70)
            // Minimum spanning tree
            // |----------------------- COCO Body -----------------------|   |------ Foot ------|  | MPII |
                 1,2,3,4,   5,6,   7,8,   9,10,   11,12,   13,14,   15,16,   19,20,21,  22,23,24,   17,18,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                   17,18,       4,        5,6,              9,10,        10,      11,12,     12,    16,
            // Ignored: shoulders, hips-ankles (-0.1% COCO_23), small toes-ankles (-0.1% COCO_23)
            //              6,        15,16,                        20,23,
               F95+30,F95+39,F95+42, // COCO-Face (+2 extra)
               F95+1,F95+2,F95+3,F95+4,F95+5,F95+6,F95+7,F95+8,F95+9,F95+10,F95+11,F95+12,F95+13,F95+14,F95+15,F95+16, // Contour (+0)
               F95+17,F95+26,F95+18,F95+19,F95+20,F95+21,F95+22,F95+23,F95+24,F95+25,F95+26, // Countour-Eyebrow + Eyebrows (+1)
               F95+27,F95+27,F95+28,F95+29,F95+30,F95+33,F95+32,F95+31,F95+34,F95+35, // Eyebrow-Nose + Nose (+1)
               F95+39,F95+42,F95+37,F95+38,F95+39,F95+40,F95+41,F95+43,F95+44,F95+45,F95+46,F95+47, // Nose-Eyes + Eyes (+1)
               F95+51,F95+49,F95+50,F95+51,F95+52,F95+53,F95+54,F95+55,F95+56,F95+57,F95+58,F95+59, // Nose-Mouth + Outer Mouth (+0)
               F95+60,F95+64,F95+61,F95+62,F95+63,F95+64,F95+65,F95+66,F95+67, // Outer-Inner + Inner Mouth (+1)
               F95+68,F95+68,F95+69,F95+69, // Eyes-Pupils (+2)
        },
        // COCO + MPII + Foot + Hands + Face
        std::vector<int>{                                                                                           // 135 (COCO_135_23, COCO_135_17, MPII_135_16, HAND_135_42, FACE_135_70, DOME_135)
            // Minimum spanning tree
            // |----------------------- COCO Body -----------------------|   |------ Foot ------|  | MPII |
                 1,2,3,4,   5,6,   7,8,   9,10,   11,12,   13,14,   15,16,   19,20,21,  22,23,24,   17,18,
            // Redundant ones
            // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
                     17,        4,        5,6,              9,10,        10,      11,12,     12,    16,
            // Ignored: shoulders, hips-ankles (-0.1% COCO_23), small toes-ankles (-0.1% COCO_23)
            //              6,        15,16,                        20,23,
            // Left hand
               H135+0,H135+1,H135+2,H135+3, H135+4,H135+5,H135+6,H135+7, H135+8,H135+9,H135+10,H135+11, H135+12,H135+13,H135+14,H135+15, H135+16,H135+17,H135+18,H135+19,
            // Right hand
               H135+20,H135+21,H135+22,H135+23, H135+24,H135+25,H135+26,H135+27, H135+28,H135+29,H135+30,H135+31, H135+32,H135+33,H135+34,H135+35, H135+36,H135+37,H135+38,H135+39,
            // Face
               F135+30,F135+39,F135+42, // COCO-Face (+2 extra)
               F135+1,F135+2,F135+3,F135+4,F135+5,F135+6,F135+7,F135+8,F135+9,F135+10,F135+11,F135+12,F135+13,F135+14,F135+15,F135+16, // Contour (+0)
               F135+17,F135+26,F135+18,F135+19,F135+20,F135+21,F135+22,F135+23,F135+24,F135+25,F135+26, // Countour-Eyebrow + Eyebrows (+1)
               F135+27,F135+27,F135+28,F135+29,F135+30,F135+33,F135+32,F135+31,F135+34,F135+35, // Eyebrow-Nose + Nose (+1)
               F135+39,F135+42,F135+37,F135+38,F135+39,F135+40,F135+41,F135+43,F135+44,F135+45,F135+46,F135+47, // Nose-Eyes + Eyes (+1)
               F135+51,F135+49,F135+50,F135+51,F135+52,F135+53,F135+54,F135+55,F135+56,F135+57,F135+58,F135+59, // Nose-Mouth + Outer Mouth (+0)
               F135+60,F135+64,F135+61,F135+62,F135+63,F135+64,F135+65,F135+66,F135+67, // Outer-Inner + Inner Mouth (+1)
               F135+68,F135+68,F135+69,F135+69, // Eyes-Pupils (+2)
        },
    };

    const auto rat = 5.75f/7.f;
    const std::array<std::vector<float>, NUMBER_MODELS> SIGMA_RATIO{
        std::vector<float>(18, 1.f),                       // 18 (COCO_18, DOME_18)
        std::vector<float>(19, 1.f),                       // 19 (COCO_19, DOME_19)
        std::vector<float>(59, 1.f),                       // 59 (DOME_59), COCO_59_17, MPII_59
        std::vector<float>(19, 1.f),                       // COCO_19b
        std::vector<float>(19, 1.f),                       // COCO_19_V2
        std::vector<float>(25, 1.f),                       // 25 (COCO_25, COCO_25_17)
        std::vector<float>(65, 1.f),                       // 65 (MPII_65_42)
        std::vector<float>(12, 1.f),                       // CAR_12
        std::vector<float>(25, 1.f),                       // 25 (COCO_25E, COCO_25_17E)
        std::vector<float>(23, 1.f),                       // 23 (COCO_23, COCO_23_17)
        std::vector<float>(22, 1.f),                       // CAR_22
        std::vector<float>(19, 1.f),                       // COCO_19E
        std::vector<float>(25, 1.f),                       // 25B (COCO_25B_23, COCO_25B_17, MPII_25B_16)
        std::vector<float>(95, 1.f),                       // 95 (COCO_95_23, COCO_95_17, MPII_95_16, FACE_95_70)
        std::vector<float>{                                // 135 (COCO_135_23, COCO_135_17, MPII_135_16, HAND_135_42, FACE_135_70, DOME_135)
            // Body
            1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,                // COCO
            1.f,1.f,                                                                            // MPII
            1.f,1.f,1.f,1.f,1.f,1.f,                                                            // Foot
            rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,    // Left hand
            rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,    // Right hand
            rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,                // Face contour
            rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,                                            // Eyebrows
            rat,rat,rat,rat,rat,rat,rat,rat,rat,                                                // Nose
            rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,                                    // Eyes
            rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,rat,    // Mouth
            rat,rat                                                                             // Pupils
        },
    };

    const std::array<std::vector<int>, NUMBER_MODELS> TAF_MAP_A{
        std::vector<int>{},
        std::vector<int>{0,6},
    };
    const std::array<std::vector<int>, NUMBER_MODELS> TAF_MAP_B{
        std::vector<int>{},
        std::vector<int>{1,18},
    };

    const std::array<std::map<unsigned int, std::string>, NUMBER_MODELS> MAPPINGS{
        POSE_BODY_18_BODY_PARTS, // 18 (COCO_18, DOME_18)
        POSE_BODY_19_BODY_PARTS, // 19 (COCO_19(b), DOME_19)
        POSE_BODY_59_BODY_PARTS, // 59 (DOME_59), COCO_59_17, MPII_59
        POSE_BODY_19_BODY_PARTS, // COCO_19b
        POSE_BODY_19_BODY_PARTS, // COCO_19_V2
        POSE_BODY_25_BODY_PARTS, // 25 (COCO_25, COCO_25_17)
        POSE_BODY_65_BODY_PARTS, // 65 (MPII_65_42)
        CAR_12_PARTS, // CAR_12
        POSE_BODY_25_BODY_PARTS, // 25E (COCO_25E, COCO_25_17E)
        POSE_BODY_23_BODY_PARTS, // 23 (COCO_23, COCO_23_17)
        CAR_22_PARTS, // CAR_22
        POSE_BODY_19_BODY_PARTS, // COCO_19E
        POSE_BODY_25B_BODY_PARTS, // 25B (COCO_25B_23, COCO_25B_17, MPII_25B_16)
        POSE_BODY_95_BODY_PARTS, // 95 (COCO_95_23, COCO_95_17, MPII_95_16, FACE_95_70)
        POSE_BODY_135_BODY_PARTS, // 135 (COCO_135_23, COCO_135_17, MPII_135_16, HAND_135_70, FACE_135_70)
    };

    const std::array<std::vector<float>, (int)PoseModel::Size> DISTANCE_AVERAGE{
        std::vector<float>{}, // 18 (COCO_18, DOME_18)
        std::vector<float>{0, -2.76364, -1.3345, 0,   -1.95322, 3.95679, -1.20664, 4.76543, // 19 (COCO_19, DOME_19)
                           1.3345, 0, 1.92318, 3.96891,   1.17999, 4.7901, 0, 7.72201,
                           -0.795236, 7.74017, -0.723963,   11.209, -0.651316, 15.6972,
                           0.764623, 7.74869, 0.70755,   11.2307, 0.612832, 15.7281,
                           -0.123134, -3.43515,   0.111775, -3.42761,
                           -0.387066, -3.16603,   0.384038, -3.15951},
        std::vector<float>{}, // 59 (DOME_59), COCO_59_17, MPII_59
        std::vector<float>{}, // COCO_19b
        std::vector<float>{}, // COCO_19_V2
        // std::vector<float>{0, -2.76364, -1.3345, 0,   -1.95322, 3.95679, -1.20664, 4.76543, // 25 (COCO_25, COCO_25_17) // 48 channels
        //                    1.3345, 0, 1.92318, 3.96891,   1.17999, 4.7901, 0, 7.72201,
        //                    -0.795236, 7.74017, -0.723963,   11.209, -0.651316, 15.6972,
        //                    0.764623, 7.74869, 0.70755,   11.2307, 0.612832, 15.7281,
        //                    -0.123134, -3.43515,   0.111775, -3.42761,
        //                    -0.387066, -3.16603,   0.384038, -3.15951,
        //                    0.344764, 12.9666, 0.624157,   12.9057, 0.195454, 12.565,
        //                    -1.06074, 12.9951, -1.2427,   12.9309, -0.800837, 12.5845},
        std::vector<float>{0, -6.55251, // 50 channels
                           0, -4.15062, -1.48818, -4.15506,   -2.22408, -0.312264, -1.42204, 0.588495,
                           1.51044, -4.14629, 2.2113, -0.312283,   1.41081, 0.612377, -0, 3.41112,
                           -0.932306, 3.45504, -0.899812,   6.79837, -0.794223, 11.4972,
                           0.919047, 3.46442, 0.902314,   6.81245, 0.79518, 11.5132,
                           -0.243982, -7.07925,   0.28065, -7.07398,
                           -0.792812, -7.09374,   0.810145, -7.06958,
                           0.582387, 7.46846, 0.889349,   7.40577, 0.465088, 7.03969,
                           -0.96686, 7.46148, -1.20773,   7.38834, -0.762135, 6.99575},
        // std::vector<float>{0.f,0.f,
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f},
        std::vector<float>{}, // 65 (MPII_65_42)
        std::vector<float>{}, // CAR_12
        // std::vector<float>{0, -2.76364, -1.3345, 0,   -1.95322, 3.95679, -1.20664, 4.76543, // 25 (COCO_25E, COCO_25_17E) // 48 channels
        //                    1.3345, 0, 1.92318, 3.96891,   1.17999, 4.7901, 0, 7.72201,
        //                    -0.795236, 7.74017, -0.723963,   11.209, -0.651316, 15.6972,
        //                    0.764623, 7.74869, 0.70755,   11.2307, 0.612832, 15.7281,
        //                    -0.123134, -3.43515,   0.111775, -3.42761,
        //                    -0.387066, -3.16603,   0.384038, -3.15951,
        //                    0.344764, 12.9666, 0.624157,   12.9057, 0.195454, 12.565,
        //                    -1.06074, 12.9951, -1.2427,   12.9309, -0.800837, 12.5845},
        std::vector<float>{0, -6.55251, // 50 channels
                           0, -4.15062, -1.48818, -4.15506,   -2.22408, -0.312264, -1.42204, 0.588495,
                           1.51044, -4.14629, 2.2113, -0.312283,   1.41081, 0.612377, -0, 3.41112,
                           -0.932306, 3.45504, -0.899812,   6.79837, -0.794223, 11.4972,
                           0.919047, 3.46442, 0.902314,   6.81245, 0.79518, 11.5132,
                           -0.243982, -7.07925,   0.28065, -7.07398,
                           -0.792812, -7.09374,   0.810145, -7.06958,
                           0.582387, 7.46846, 0.889349,   7.40577, 0.465088, 7.03969,
                           -0.96686, 7.46148, -1.20773,   7.38834, -0.762135, 6.99575},
        // std::vector<float>{0.f,0.f, // 50 channels
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f},
        std::vector<float>{},
        std::vector<float>{},
        std::vector<float>{}, // CAR_22
        std::vector<float>{}, // BODY_19E
        std::vector<float>{}, // MPII_25B_16
        std::vector<float>{}, // PT_25B_15
        // COCO + MPII + Foot + Face
        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},
        // COCO + MPII + Foot + Hands + Face
        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},
    };

    const std::array<std::vector<float>, (int)PoseModel::Size> DISTANCE_SIGMA{
        std::vector<float>{}, // 18 (COCO_18, DOME_18)
        std::vector<float>{3.39629, 3.15605, 3.16913, 1.8234,   5.82252, 5.05674, 7.09876, 6.64574, // 19 (COCO_19, DOME_19)
                           3.16913, 1.8234, 5.79415, 5.01424,   7.03866, 6.62427, 5.52593, 6.75962,
                           5.91224, 6.87241, 8.66473,   10.1792, 11.5871, 13.6565,
                           5.86653, 6.89568, 8.68067,   10.2127, 11.5954, 13.6722,
                           3.3335, 3.49128,   3.34476, 3.50079,
                           2.93982, 3.11151,   2.95006, 3.11004},
        std::vector<float>{}, // 59 (DOME_59), COCO_59_17, MPII_59
        std::vector<float>{}, // COCO_19b
        std::vector<float>{}, // COCO_19_V2
        // std::vector<float>{3.39629, 3.15605, 3.16913, 1.8234,   5.82252, 5.05674, 7.09876, 6.64574, // 25 (COCO_25, COCO_25_17)
        //                    3.16913, 1.8234, 5.79415, 5.01424,   7.03866, 6.62427, 5.52593, 6.75962,
        //                    5.91224, 6.87241, 8.66473,   10.1792, 11.5871, 13.6565,
        //                    5.86653, 6.89568, 8.68067,   10.2127, 11.5954, 13.6722,
        //                    3.3335, 3.49128,   3.34476, 3.50079,
        //                    2.93982, 3.11151,   2.95006, 3.11004,
        //                    9.69408, 7.58921, 9.71193,   7.44185, 9.19343, 7.11157,
        //                    9.16848, 7.86122, 9.07613,   7.83682, 8.91951, 7.33715},
        std::vector<float>{7.26789, 9.70751, // 50 channels
                           6.29588, 8.93472, 6.97401, 9.13746,   7.49632, 9.44757, 8.06695, 9.97319,
                           6.99726, 9.14608, 7.50529, 9.43568,   8.05888, 9.98207, 6.38929, 9.29314,
                           6.71801, 9.39271, 8.00608,   10.6141, 10.3416, 12.7812,
                           6.69875, 9.41407, 8.01876,   10.637, 10.3475, 12.7849,
                           7.30923, 9.7324,   7.27886, 9.73406,
                           7.35978, 9.7289,   7.28914, 9.67711,
                           7.93153, 8.10845, 7.95577,   8.01729, 7.56865, 7.87314,
                           7.4655, 8.25336, 7.43958,   8.26333, 7.33667, 7.97446},
        // std::vector<float>{1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f},
        std::vector<float>{}, // 65 (MPII_65_42)
        std::vector<float>{}, // CAR_12
        // std::vector<float>{3.39629, 3.15605, 3.16913, 1.8234,   5.82252, 5.05674, 7.09876, 6.64574, // 25 (COCO_25E, COCO_25_17E) // 48 channels
        //                    3.16913, 1.8234, 5.79415, 5.01424,   7.03866, 6.62427, 5.52593, 6.75962,
        //                    5.91224, 6.87241, 8.66473,   10.1792, 11.5871, 13.6565,
        //                    5.86653, 6.89568, 8.68067,   10.2127, 11.5954, 13.6722,
        //                    3.3335, 3.49128,   3.34476, 3.50079,
        //                    2.93982, 3.11151,   2.95006, 3.11004,
        //                    9.69408, 7.58921, 9.71193,   7.44185, 9.19343, 7.11157,
        //                    9.16848, 7.86122, 9.07613,   7.83682, 8.91951, 7.33715},
        std::vector<float>{7.26789, 9.70751, // 50 channels
                           6.29588, 8.93472, 6.97401, 9.13746,   7.49632, 9.44757, 8.06695, 9.97319,
                           6.99726, 9.14608, 7.50529, 9.43568,   8.05888, 9.98207, 6.38929, 9.29314,
                           6.71801, 9.39271, 8.00608,   10.6141, 10.3416, 12.7812,
                           6.69875, 9.41407, 8.01876,   10.637, 10.3475, 12.7849,
                           7.30923, 9.7324,   7.27886, 9.73406,
                           7.35978, 9.7289,   7.28914, 9.67711,
                           7.93153, 8.10845, 7.95577,   8.01729, 7.56865, 7.87314,
                           7.4655, 8.25336, 7.43958,   8.26333, 7.33667, 7.97446},
        // std::vector<float>{1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f},
        std::vector<float>{},
        std::vector<float>{},
        std::vector<float>{}, // CAR_22
        std::vector<float>{}, // BODY_19E
        std::vector<float>{}, // COCO_25B (COCO_25B_23, COCO_25B_17)
        std::vector<float>{}, // MPII_25B_16
        std::vector<float>{}, // PT_25B_15
        // COCO + MPII + Foot + Face
        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},
        // COCO + MPII + Foot + Hands + Face
        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},        std::vector<float>{},
    };





    // Fixed functions
    bool addBkgChannel(const PoseModel poseModel)
    {
        return (getMapping(poseModel).size() != NUMBER_BODY_PARTS[(int)poseModel]);
    }

    int getNumberBodyParts(const PoseModel poseModel)
    {
        return NUMBER_BODY_PARTS.at((int)poseModel);
    }

    int getNumberBodyPartsLmdb(const PoseModel poseModel)
    {
        return NUMBER_PARTS_LMDB.at((int)poseModel);
    }

    int getNumberPafChannels(const PoseModel poseModel)
    {
        return (int)(2*getPafIndexA(poseModel).size());
    }

    int getNumberTafChannels(const int tafTopology)
    {
        return TAF_MAP_A[tafTopology].size()*2;
    }

    int getNumberBodyAndPafChannels(const PoseModel poseModel)
    {
        return NUMBER_BODY_PARTS.at((int)poseModel) + getNumberPafChannels(poseModel);
    }

    int getNumberBodyBkgAndPAF(const PoseModel poseModel)
    {
        return getNumberBodyAndPafChannels(poseModel) + (addBkgChannel(poseModel) ? 1 : 0);
    }

    const std::vector<std::vector<int>>& getLmdbToOpenPoseKeypoints(const PoseModel poseModel)
    {
        return LMDB_TO_OPENPOSE_KEYPOINTS.at((int)poseModel);
    }

    const std::vector<std::vector<int>>& getMaskedChannels(const PoseModel poseModel)
    {
        return CHANNELS_TO_MASK.at((int)poseModel);
    }

    const std::vector<std::array<int,2>>& getSwapLeftRightKeypoints(const PoseModel poseModel)
    {
        return SWAP_LEFT_RIGHT_KEYPOINTS.at(poseModelToIndex(poseModel));
    }

    const std::vector<int>& getPafIndexA(const PoseModel poseModel)
    {
        return LABEL_MAP_A.at(poseModelToIndex(poseModel));
    }

    const std::vector<int>& getPafIndexB(const PoseModel poseModel)
    {
        return LABEL_MAP_B.at(poseModelToIndex(poseModel));
    }

    const std::vector<float>& getSigmaRatio(const PoseModel poseModel)
    {
        return SIGMA_RATIO.at(poseModelToIndex(poseModel));
    }

    const std::vector<int>& getTafIndexA(const int tafTopology)
    {
        return TAF_MAP_A.at(tafTopology);
    }

    const std::vector<int>& getTafIndexB(const int tafTopology)
    {
        return TAF_MAP_B.at(tafTopology);
    }

    const std::map<unsigned int, std::string>& getMapping(const PoseModel poseModel)
    {
        return MAPPINGS.at(poseModelToIndex(poseModel));
    }

    const std::vector<float>& getDistanceAverage(const PoseModel poseModel)
    {
        return DISTANCE_AVERAGE.at(poseModelToIndex(poseModel));
    }

    const std::vector<float>& getDistanceSigma(const PoseModel poseModel)
    {
        return DISTANCE_SIGMA.at(poseModelToIndex(poseModel));
    }

    unsigned int getRootIndex()
    {
        return 1u;
    }

    std::vector<int> getIndexesForParts(const PoseModel poseModel, const std::vector<int>& missingBodyPartsBase,
                                        const std::vector<float>& isVisible, const float minVisibleToBlock, const int tafTopology)
    {
        auto totalTafChannels = getNumberTafChannels(tafTopology);

        auto missingBodyParts = missingBodyPartsBase;
        // If masking also non visible points
        if (isVisible.empty())
            throw std::runtime_error{"Field isVisible cannot be empty" + getLine(__LINE__, __FUNCTION__, __FILE__)};
        // if (!isVisible.empty())
        // {
            for (auto i = 0u ; i < isVisible.size() ; i++)
                if (isVisible[i] >= minVisibleToBlock)
                    missingBodyParts.emplace_back(i);
            std::sort(missingBodyParts.begin(), missingBodyParts.end());
        // }
        // Missing PAF channels
        std::vector<int> missingChannels;
        if (!missingBodyParts.empty())
        {
            // PAF
            const auto& pafIndexA = getPafIndexA(poseModel);
            const auto& pafIndexB = getPafIndexB(poseModel);
            for (auto i = 0u ; i < missingBodyParts.size() ; i++)
            {
                for (auto pafId = 0u ; pafId < pafIndexA.size() ; pafId++)
                {
                    if (pafIndexA[pafId] == missingBodyParts[i] || pafIndexB[pafId] == missingBodyParts[i])
                    {
                        missingChannels.emplace_back(2*pafId);
                        missingChannels.emplace_back(2*pafId+1);
                    }
                }
            }
            // Offset for TAF
            for(auto& m : missingChannels) m+= totalTafChannels;

            // TAF
            if(tafTopology){
                const auto& tafIndexA = getTafIndexA(tafTopology);
                const auto& tafIndexB = getTafIndexB(tafTopology);
                for (auto i = 0u ; i < missingBodyParts.size() ; i++)
                {
                    for (auto tafId = 0u ; tafId < tafIndexA.size() ; tafId++)
                    {
                        if (tafIndexA[tafId] == missingBodyParts[i] || tafIndexB[tafId] == missingBodyParts[i])
                        {
                            missingChannels.emplace_back(2*tafId);
                            missingChannels.emplace_back(2*tafId+1);
                        }
                    }
                }
            }

            // Sort indexes (only disordered in PAFs)
            std::sort(missingChannels.begin(), missingChannels.end());
            // Remove duplicates (only possible in PAFs)
            const auto it = std::unique(missingChannels.begin(), missingChannels.end());
            missingChannels.resize(std::distance(missingChannels.begin(), it));
            // Body parts to channel indexes (add #PAF channels)
            std::transform(missingBodyParts.begin(), missingBodyParts.end(), missingBodyParts.begin(),
                           std::bind2nd(std::plus<int>(), totalTafChannels + getNumberPafChannels(poseModel)));
            missingChannels.insert(missingChannels.end(), missingBodyParts.begin(), missingBodyParts.end());
        }
        // Return result
        return missingChannels;
    }

    std::vector<int> getEmptyChannels(const PoseModel poseModel, const std::vector<float>& isVisible,
                                      const float minVisibleToBlock, const int tafTopology)
    {
        // Missing body parts
        std::vector<int> missingBodyParts;
        // const auto& lmdbToOpenPoseKeypoints = getLmdbToOpenPoseKeypoints(poseModel);
        const auto& lmdbToOpenPoseKeypoints = getMaskedChannels(poseModel);
        for (auto i = 0u ; i < lmdbToOpenPoseKeypoints.size() ; i++)
            if (lmdbToOpenPoseKeypoints[i].empty())
                missingBodyParts.emplace_back(i);
        return getIndexesForParts(poseModel, missingBodyParts, isVisible, minVisibleToBlock, tafTopology);
    }

    std::vector<int> getMinus1Channels(const PoseModel poseModel, const std::vector<float>& isVisible, const int tafTopology)
    {
        // Missing body parts
        std::vector<int> missingBodyParts;
        // const auto& lmdbToOpenPoseKeypoints = getLmdbToOpenPoseKeypoints(poseModel);
        const auto& lmdbToOpenPoseKeypoints = getMaskedChannels(poseModel);
        for (auto i = 0u ; i < lmdbToOpenPoseKeypoints.size() ; i++)
            if (lmdbToOpenPoseKeypoints[i].size() == 1 && lmdbToOpenPoseKeypoints[i][0] == -1)
                missingBodyParts.emplace_back(i);
        return getIndexesForParts(poseModel, missingBodyParts, isVisible, 4.f, tafTopology);
    }
}  // namespace caffe
