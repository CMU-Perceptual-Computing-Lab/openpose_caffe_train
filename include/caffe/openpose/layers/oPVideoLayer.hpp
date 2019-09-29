// File based in `data_layer.hpp`, extracted from Caffe GitHub on Sep 7th, 2017
// https://github.com/BVLC/caffe/

#ifndef CAFFE_OPENPOSE_OP_VIDEO_LAYER_HPP
#define CAFFE_OPENPOSE_OP_VIDEO_LAYER_HPP

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
#include <boost/thread/thread.hpp>

#include <boost/algorithm/string.hpp>
// OpenPose: added end

namespace caffe {

template <typename Dtype>
class OPVideoLayer : public BasePrefetchingDataLayer<Dtype> {
public:
    explicit OPVideoLayer(const LayerParameter& param);
    virtual ~OPVideoLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "OPVideo"; }
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

    virtual void load_batch(Batch<Dtype>* batch);

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
    std::vector<std::string> mInputType;
    std::vector<float> mProbabilities;
    int mFrameSize;
    // OpenPose: added end

    // protected:
    //  void Next();
    //  bool Skip();
    //  virtual void load_batch(Batch<Dtype>* batch);

    //  shared_ptr<db::DB> db_;
    //  shared_ptr<db::Cursor> cursor_;
    //  uint64_t offset_;

    //  // OpenPose: added
    //  bool SkipSecond();
    //  bool SkipThird();
    //  bool SkipA();
    //  bool SkipB();
    //  void NextBackground();
    //  void NextSecond();
    //  void NextThird();
    //  void NextA();
    //  void NextB();
    //  // Secondary lmdb
    //  uint64_t offsetSecond;
    //  bool secondDb;
    //  float secondProbability;
    //  shared_ptr<db::DB> dbSecond;
    //  shared_ptr<db::Cursor> cursorSecond;
    //  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformerSecondary;
    //  // Tertiary lmdb
    //  uint64_t offsetThird;
    //  bool thirdDb;
    //  float thirdProbability;
    //  shared_ptr<db::DB> dbThird;
    //  shared_ptr<db::Cursor> cursorThird;
    //  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformerTertiary;
    //  // A lmdb
    //  uint64_t offsetA;
    //  bool ADb;
    //  float AProbability;
    //  shared_ptr<db::DB> dbA;
    //  shared_ptr<db::Cursor> cursorA;
    //  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformerA;
    //  // B lmdb
    //  uint64_t offsetB;
    //  bool BDb;
    //  float BProbability;
    //  shared_ptr<db::DB> dbB;
    //  shared_ptr<db::Cursor> cursorB;
    //  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformerB;
    //  // Background lmdb
    //  bool backgroundDb;
    //  shared_ptr<db::DB> dbBackground;
    //  shared_ptr<db::Cursor> cursorBackground;
    //  // New label
    //  Blob<Dtype> transformed_label_;
    //  // Data augmentation parameters
    //  OPTransformationParameter op_transform_param_;
    //  // Data augmentation class
    //  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformer;
    //  // Timer
    //  unsigned long long mOnes;
    //  unsigned long long mTwos;
    //  unsigned long long mThrees;
    //  unsigned long long mAs;
    //  unsigned long long mBs;
    //  int mCounter;
    //  int vCounter = 0;
    //  double mDuration;

    //  int frame_size = 6;
    //  void sample_dbs(bool& desiredDbIs1, bool& desiredDbIs2, bool& desiredDbIs3);
    //  void sample_ab(bool& desiredDbA, bool& desiredDbB);
    //  // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_OP_DATA_LAYER_HPP
