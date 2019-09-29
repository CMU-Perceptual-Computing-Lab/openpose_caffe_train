#ifndef CAFFE_CUDNN_CONV_LAYER_HPP_
#define CAFFE_CUDNN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of ConvolutionLayer.
 *        Fallback to ConvolutionLayer for CPU mode.
 *
 * cuDNN accelerates convolution through forward kernels for filtering and bias
 * plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + cuDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
 *
 * The CUDNN engine does not have memory overhead for matrix buffers. For many
 * input and filter regimes the CUDNN engine is faster than the CAFFE engine,
 * but for fully-convolutional models and large inputs the CAFFE engine can be
 * faster as long as it fits in memory.
*/
template <typename Dtype>
class CuDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit CuDNNConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param), handles_setup_(false),
        weight_initialized_{false} {}// Binary added: weight_initialized_
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNConvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t *fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData

  // Binary net added
  bool weight_initialized_;
  std::unique_ptr<Blob<Dtype>> weight_binary_;
  std::unique_ptr<Blob<Dtype>> bottom_binary_;
  std::unique_ptr<Blob<Dtype>> matrix_A_;
  std::unique_ptr<Blob<Dtype>> matrix_one_over_chw;
  std::unique_ptr<Blob<Dtype>> matrix_K_;
  cudnnHandle_t matrix_K_handle_;
  cudaStream_t matrix_K_stream_;
  cudnnTensorDescriptor_t matrix_A_desc_;
  cudnnFilterDescriptor_t matrix_one_filter_desc_;
  cudnnTensorDescriptor_t matrix_K_desc_;
  cudnnConvolutionDescriptor_t matrix_AK_conv_descs_;
  cudnnConvolutionFwdAlgo_t matrix_AK_fwd_algo_;
  size_t matrix_AK_workspace_fwd_sizes_;
  void normalizeWeights();
  void approximateInputGpu(Blob<Dtype>* bottom_binary_, Blob<Dtype>* matrix_A_,
                           Blob<Dtype>* matrix_K_, const Blob<Dtype>* const matrix_one_over_chw,
                           const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
                           const int num, const int binaryOption) const;
  // Binary net end
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_CONV_LAYER_HPP_
