#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 1)
  {
    CUDA_CHECK(cudaStreamCreate(&matrix_K_stream_));
    CUDNN_CHECK(cudnnCreate(&matrix_K_handle_));
    CUDNN_CHECK(cudnnSetStream(matrix_K_handle_, matrix_K_stream_));
  }
  // Binary added end

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 1)
  {
    cudnn::createFilterDesc<Dtype>(&matrix_one_filter_desc_,
        1, 1, kernel_h, kernel_w);
  }
  // Binary added end

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 1)
  {
    cudnn::createTensor4dDesc<Dtype>(&matrix_A_desc_);
    cudnn::createConvolutionDesc<Dtype>(&matrix_AK_conv_descs_);
    cudnn::createTensor4dDesc<Dtype>(&matrix_K_desc_);
  }
  // Binary added end

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_[i]));

    // get workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 1)
  {
    cudnn::setTensor4dDesc<Dtype>(&matrix_A_desc_,
        this->num_,
        1, height, width,
        height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&matrix_K_desc_,
        this->num_,
        1, height_out, width_out,
        this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&matrix_AK_conv_descs_, matrix_A_desc_,
        matrix_one_filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(matrix_K_handle_,
      matrix_A_desc_,
      matrix_one_filter_desc_,
      matrix_AK_conv_descs_,
      matrix_K_desc_,
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &matrix_AK_fwd_algo_));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(matrix_K_handle_,
      matrix_A_desc_,
      matrix_one_filter_desc_,
      matrix_AK_conv_descs_,
      matrix_K_desc_,
      matrix_AK_fwd_algo_,
      &(matrix_AK_workspace_fwd_sizes_)));

    // // choose backward algorithm for filter
    // CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(matrix_K_handle_,
    //       matrix_A_desc_, matrix_K_desc_, matrix_AK_conv_descs_, matrix_one_filter_desc_,
    //       CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
    //       workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // // get workspace for backwards filter algorithm
    // CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(matrix_K_handle_,
    //       matrix_A_desc_, matrix_K_desc_, matrix_AK_conv_descs_, matrix_one_filter_desc_,
    //       bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // // choose backward algo for data
    // CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(matrix_K_handle_,
    //       matrix_one_filter_desc_, matrix_K_desc_, matrix_AK_conv_descs_, matrix_A_desc_,
    //       CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
    //     workspace_limit_bytes, &bwd_data_algo_[i]));

    // // get workspace size
    // CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(matrix_K_handle_,
    //       matrix_one_filter_desc_, matrix_K_desc_, matrix_AK_conv_descs_, matrix_A_desc_,
    //       bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }
  // Binary added end

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 1)
  {
    cudaStreamDestroy(matrix_K_stream_);
    cudnnDestroy(matrix_K_handle_);
  }
  // Binary added end

  cudaFree(workspaceData);
  delete [] workspace;
  delete [] stream_;
  delete [] handle_;
  // Binary added
  // if (this->layer_param_.convolution_param().binary() > 1)
  // {
  //   delete [] matrix_K_stream_; // It is not a ptr
  //   delete [] matrix_K_handle_; // It is not a ptr
  // }
  // Binary added end
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

// Binary added
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::normalizeWeights()
{
  // Data to weightReal
  auto* weightBinaryData = weight_binary_->mutable_cpu_data();
  auto* weightRealData = this->blobs_[0]->mutable_cpu_data();
  // Real to binary data
  // Channel area = volume from axis 2 to final (num, channel, h, w)
  // const auto channelArea = weight_binary_->count(1);
  // const auto imageArea = weight_binary_->count(2);
  // for (auto num = 0 ; num < weight_binary_->shape()[0] ; num++)
  // {
  //   const auto offsetNum = num*channelArea;
  //   for (auto channel = 0 ; channel < weight_binary_->shape()[1] ; channel++)
  //   {
  //     const auto offset = offsetNum + channel * imageArea;

  //     // // XNOR-style
  //     // // L1 norm
  //     // auto l1Norm = Dtype(0);
  //     // for (auto i = 0 ; i < imageArea ; i++)
  //     // {
  //     //   // truncate to +-1
  //     //   weightRealData[offset+i] = std::max(-Dtype(1), std::min(Dtype(1), weightRealData[offset+i]));
  //     //   // l1Norm
  //     //   l1Norm += (weightRealData[offset+i] < 0
  //     //     ? -weightRealData[offset+i] : weightRealData[offset+i]);
  //     // }
  //     // const auto sum = l1Norm / imageArea;
  //     // for (auto i = 0 ; i < imageArea ; i++)
  //     //   weightBinaryData[offset+i] = (weightRealData[offset+i] < 0 ? -sum : sum);

  //     // Old binary net style
  //     // truncate to +-1
  //     for (auto i = 0 ; i < imageArea ; i++)
  //       weightRealData[offset+i] = std::max(-Dtype(1), std::min(Dtype(1), weightRealData[offset+i]));
  //     // Binary approximation
  //     for (auto i = 0 ; i < imageArea ; i++)
  //       weightBinaryData[offset+i] = (weightRealData[offset+i] < 0 ? -Dtype(1) : Dtype(1));
  //   }
  // }
  for (auto i = 0 ; i < this->blobs_[0]->count() ; i++)
  {
    weightRealData[i] = std::max(-Dtype(1), std::min(Dtype(1), weightRealData[i]));
    weightBinaryData[i] = (weightRealData[i] < 0 ? -Dtype(1) : Dtype(1));
  }
}
// Binary added ended

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
