// File based in `data_layer.hpp`, extracted from Caffe GitHub on Sep 7th, 2017
// https://github.com/BVLC/caffe/

#ifndef CAFFE_OPENPOSE_OP_DATA_LAYER_HPP
#define CAFFE_OPENPOSE_OP_DATA_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
// OpenPose: added
#include "caffe/openpose/oPDataTransformer.hpp"
// OpenPose: added end

namespace caffe {

template <typename Dtype>
class OPDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit OPDataLayer(const LayerParameter& param);
  virtual ~OPDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "OPData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  // void Next(); // OpenPose: commented for more generic
  // bool Skip(); // OpenPose: commented for more generic
  void Next(const int index = 0); // OpenPose: added
  bool Skip(const int index = 0); // OpenPose: added
  void NextBackground(); // OpenPose: added
  bool SkipBackground(); // OpenPose: added

 public: // OpenPose: added
  virtual void load_batch(Batch<Dtype>* batch);
 protected: // OpenPose: added

  // shared_ptr<db::DB> db_; // OpenPose: commented for more generic mDbs
  // shared_ptr<db::Cursor> cursor_; // OpenPose: commented for more generic mCursors
  // uint64_t offset_; // OpenPose: commented for more generic mOffsets
  // OpenPose: added
  std::vector<shared_ptr<db::DB>> mDbs;
  std::vector<shared_ptr<db::Cursor>> mCursors;
  std::vector<uint64_t> mOffsets;

  // Background lmdb
  uint64_t offsetBackground;
  float onlyBackgroundProbability;
  bool backgroundDb;
  shared_ptr<db::DB> dbBackground;
  shared_ptr<db::Cursor> cursorBackground;
  // New label
  Blob<Dtype> transformed_label_;
  // Data augmentation parameters
  OPTransformationParameter op_transform_param_;
  // Data augmentation class
  std::vector<shared_ptr<OPDataTransformer<Dtype>>> mOPDataTransformers;
  // Timer
  std::vector<unsigned long long> mCounterTimer;
  unsigned long long mCounterTimerBkg;
  int mCounter;
  double mDuration;
  std::vector<long double> mDistanceAverage;
  std::vector<long double> mDistanceSigma;
  std::vector<unsigned long long> mDistanceAverageCounter;
  // Generic for `n` datasets
  std::vector<std::string> mSources;
  std::vector<std::string> mModels;
  std::vector<float> mProbabilities;
  // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_OP_DATA_LAYER_HPP
