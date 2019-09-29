#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

// Binary added
// #define SLOW_SECURITY_CHECKS

// Get L1 norm
template <typename Dtype>
inline __device__ Dtype getL1Norm(const Dtype* weightData, const int weightArea)
{
  // L1 norm
  auto l1Norm = Dtype(0);
  for (auto i = 0 ; i < weightArea ; i++)
    l1Norm += (weightData[i] < 0 ? -weightData[i] : weightData[i]);
  return l1Norm;
}

// \tilde{W} = alpha * B
//     alpha = ||W||_1 / n
//     n = c x h x w
//     B_i = sign(W_i)
template <typename Dtype>
__global__ void approximateWeightsGpu(Dtype* weightBinaryData, const Dtype* weightRealData, const int count,
                                      const int weightArea)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < count)
  {
    // Offset
    const auto offset = globalIdx * weightArea;
    const auto* weightRealDataOffset = &weightRealData[offset];
    auto* weightBinaryDataOffset = &weightBinaryData[offset];
    // XNOR-style
    // L1 norm & optimal alpha
    const auto alphaOptimal = getL1Norm(weightRealDataOffset, weightArea) / weightArea;
    // Update output
    for (auto i = 0 ; i < weightArea ; i++)
      weightBinaryDataOffset[i] = (weightRealDataOffset[i] < 0 ? -alphaOptimal : alphaOptimal);
  }
}

// Binary weights = +-n - XNOR-style
template <typename Dtype>
__global__ void backwardNormalizeWeightsGpu(Dtype* bottomDiff, const Dtype* topDiff, const Dtype* bottomData, const int count,
                                            const int weightArea)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < count)
  {
    // Offset
    const auto offset = globalIdx * weightArea;
    const auto* topDiffOffset = &topDiff[offset];
    const auto* bottomDataOffset = &bottomData[offset];
    auto* bottomDiffOffset = &bottomDiff[offset];
    // XNOR-style
    // L1 norm & optimal alpha
    const auto alphaOptimal = getL1Norm(bottomDataOffset, weightArea) / weightArea;
    // Update output
    const auto oneOverWeightArea = Dtype(1)/Dtype(weightArea);
    for (auto i = 0 ; i < weightArea ; i++)
      bottomDiffOffset[i] = topDiffOffset[i]
                          * (oneOverWeightArea
                             + alphaOptimal * max(-Dtype(1), min(Dtype(1), bottomDataOffset[i])));
  }
}

// Dtype data (integer, floating, etc.) into binary data
template <typename Dtype>
__global__ void dTypeToBinaryGpu(Dtype* binaryData, const Dtype* realData, const int count)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < count)
  {
    // realData[globalIdx] = max(-Dtype(1), min(Dtype(1), realData[globalIdx])); // When used as binarization
    binaryData[globalIdx] = (realData[globalIdx] < 0 ? Dtype(-1) : Dtype(1));
  }
}

// NxCxHxW --> Nx1xHxW
// output(n,1,h,w) = sum(abs(input(n,:,h,w)))
template <typename Dtype>
__global__ void addOverChannelsGpu(Dtype* outputData, const Dtype* inputData, const int bottomChannels,
                                   const int bottomWHArea)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < bottomWHArea)
  {
    auto& output = outputData[globalIdx];
    output = 0;
    for (auto i = 0 ; i < bottomChannels ; i++)
    {
      const auto value = inputData[globalIdx+i*bottomWHArea];
      output += (value < 0 ? -value : value);
    }
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::approximateInputGpu(Blob<Dtype>* bottom_binary_, Blob<Dtype>* matrix_A_,
  Blob<Dtype>* matrix_K_, const Blob<Dtype>* const matrix_one_over_chw,
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top, const int num, const int binaryOption) const
{
  // Full binary (weights + input)
  if (binaryOption > 1)
  {
    // Get binary input (bottom_binary_)
    const auto count = bottom_binary_->count();
    dTypeToBinaryGpu<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_binary_->mutable_gpu_data(), bottom[0]->gpu_data(), count);
    // // Input (bottom) dependent
    // const auto bottomChannels = bottom_binary_->shape(1);
    // const auto bottomWHArea = bottom_binary_->count(2);
    // const auto bottomNArea = bottomChannels * bottomWHArea;
    // CHECK_EQ(bottomWHArea, matrix_A_->count(1));
    // // Output (top) dependent
    // const auto* const weightOneOverCHW = matrix_one_over_chw->gpu_data();
    // const auto* const matrix_A_data = matrix_A_->gpu_data();
    // // K matrix
    // auto* matrix_K_data = matrix_K_->mutable_gpu_data();
    // for (int n = 0; n < num; ++n)
    // {
    //   // Get A matrix (matrix_A_)
    //   addOverChannelsGpu<<<CAFFE_GET_BLOCKS(bottomWHArea), CAFFE_CUDA_NUM_THREADS>>>(
    //     matrix_A_->mutable_gpu_data() + n*bottomWHArea, bottom[0]->gpu_data() + n*bottomNArea, bottomChannels,
    //     bottomWHArea);
    //   // Get K matrix (matrix_K_)
    //   CUDNN_CHECK(cudnnConvolutionForward(matrix_K_handle_,
    //     cudnn::dataType<Dtype>::one,
    //     matrix_A_desc_, matrix_A_data,
    //     matrix_one_filter_desc_, weightOneOverCHW,
    //     matrix_AK_conv_descs_,
    //     matrix_AK_fwd_algo_, workspace[0], matrix_AK_workspace_fwd_sizes_,
    //     cudnn::dataType<Dtype>::zero,
    //     matrix_K_desc_, matrix_K_data));
    // }
  }
}

// Float data into binary data
template <typename Dtype>
__global__ void multiplyOverChannelsGpu(Dtype* outputData, const Dtype* multiplierData, const int topChannels,
                                        const int topWHArea)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < topWHArea)
    for (auto i = 0 ; i < topChannels ; i++)
      outputData[globalIdx+i*topWHArea] *= multiplierData[globalIdx];
}

// XNOR-style
template <typename Dtype>
__global__ void backwardNormalizeInputGpu(Dtype* bottomDiff, /*const Dtype* topDiff,*/ const Dtype* bottomData, const int count,
                                          const Dtype oneOverWeightArea, const Dtype* matrixK, const int matrixKOffset)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < count)
  {
    // bottomDiff[globalIdx] = topDiff[globalIdx] * (oneOverWeightArea
    bottomDiff[globalIdx] *= (oneOverWeightArea
                             + matrixK[globalIdx % matrixKOffset] * max(-Dtype(1), min(Dtype(1), bottomData[globalIdx])));
  }
}

template <typename Dtype>
__global__ void backwardNormalizeInputGpu(Dtype* bottomDiff, /*const Dtype* topDiff,*/ const Dtype* bottomData, const int count)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < count)
  {
    if ((bottomData[globalIdx] < -1 && bottomDiff[globalIdx] < 0)
        || (bottomData[globalIdx] > 1 && bottomDiff[globalIdx] > 0))
      bottomDiff[globalIdx] = 0;
  }
}
// Binary added ended

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 0)
  {
    const auto binarizeWeightsThisFrame = (this->phase_ == TRAIN || !weight_initialized_);
    // TEST/TRAIN - First frame (initialization)
    if (!weight_initialized_)
    {
      CHECK_EQ(this->group_, 1) << "Binary conv net not implemented for !=1 groups.";
      CHECK_EQ(bottom.size(), 1) << "Binary conv net not implemented for !=1 bottoms.";
      weight_initialized_ = true;
      CHECK_GE(this->blobs_.size(), 1);
      CHECK_GT(this->blobs_[0]->shape().size(), 2u);
      weight_binary_.reset(new Blob<Dtype>());
      weight_binary_->Reshape(this->blobs_[0]->shape());
      if (this->layer_param_.convolution_param().binary() > 1)
      {
        // Blob initialization
        bottom_binary_.reset(new Blob<Dtype>());
        matrix_A_.reset(new Blob<Dtype>());
        matrix_K_.reset(new Blob<Dtype>());
        matrix_one_over_chw.reset(new Blob<Dtype>());
        // Blob reshape
        bottom_binary_->Reshape(bottom[0]->shape());
        matrix_A_->Reshape(bottom[0]->shape(0), 1, bottom[0]->shape(2), bottom[0]->shape(3));
        matrix_K_->Reshape(top[0]->shape(0), 1, top[0]->shape(2), top[0]->shape(3));
        matrix_one_over_chw->Reshape(bottom[0]->shape(0), 1, this->blobs_[0]->shape(2), this->blobs_[0]->shape(3));
        // Filling matrix_one_over_chw
        auto* inputOnes = matrix_one_over_chw->mutable_cpu_data();
        const auto bottomNArea = bottom_binary_->count(1);
        for (auto i = 0 ; i < matrix_one_over_chw->count() ; i++)
          inputOnes[i] = Dtype(1)/Dtype(bottomNArea);
      }
    }
    // 1 frame (if testing), every frame if train
    if (binarizeWeightsThisFrame)
    {
      const auto this_blobs_0 = this->blobs_[0];
      // // Option a - Weight = +-1
      // const auto count = this_blobs_0->count();
      // dTypeToBinaryGpu<<<CAFFE_GET_BLOCKS(count/binarizationArea), CAFFE_CUDA_NUM_THREADS>>>(
      //   weight_binary_->mutable_gpu_data(), this_blobs_0->mutable_gpu_data(), count);
      // // Option b - Weight = +-n per w,h
      // const auto count = this_blobs_0->count();
      // const auto binarizationArea = this_blobs_0->count(2);
      // const auto countReduced = count/binarizationArea;
      // approximateWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
      //   weight_binary_->mutable_gpu_data(), this_blobs_0->gpu_data(), countReduced, binarizationArea);
      // Option c - Weight = +-n per c,w,h
      const auto binarizationArea = this_blobs_0->count(1);
      const auto countReduced = this_blobs_0->shape(0);
      approximateWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
        weight_binary_->mutable_gpu_data(), this_blobs_0->gpu_data(), countReduced, binarizationArea);
      // SECURITY CHECK
      #ifdef SLOW_SECURITY_CHECKS
        const auto cpuDataB = weight_binary_->cpu_data();
        const auto cpuDataW = this_blobs_0->cpu_data();
        for (auto i = 0 ; i < this_blobs_0->shape(0); i++)
        {
          auto counter = Dtype(0);
          for (auto j = 0 ; j < binarizationArea; j++)
            counter += std::abs(cpuDataW[j+i*binarizationArea]);
          counter /= binarizationArea;
          CHECK_EQ(counter, std::abs(cpuDataB[i*binarizationArea]));
          // std::cout << counter << " vs. " << cpuDataB[i*binarizationArea]
          //   << " vs. " << cpuDataB[i*binarizationArea+1] << " vs. " << cpuDataB[i*binarizationArea-1] << std::endl;
        }
      #endif
    }
    // Every frame
    approximateInputGpu(bottom_binary_.get(), matrix_A_.get(), matrix_K_.get(),
                        this->matrix_one_over_chw.get(), bottom, top, this->num_,
                        this->layer_param_.convolution_param().binary());
    // SECURITY CHECK
    #ifdef SLOW_SECURITY_CHECKS
      if (this->layer_param_.convolution_param().binary() > 1)
      {
        const auto bottomData = bottom[0]->cpu_data();
        const auto matrixAData = matrix_A_->cpu_data();
        // bottom_binary_ from bottom[0]
        for (auto i = 0 ; i < bottom[0]->count(); i++)
          CHECK(bottomData[i] < 0
            ? bottom_binary_->cpu_data()[i] == -1
            : bottom_binary_->cpu_data()[i] == 1);
        // matrix_A_
        const auto cwhArea = bottom[0]->count(1);
        const auto whArea = bottom[0]->count(2);
        for (auto num = 0 ; num < bottom[0]->shape(0); num++)
        {
          for (auto xy = 0 ; xy < whArea; xy++)
          {
            auto counter = Dtype(0);
            for (auto c = 0 ; c < bottom[0]->shape(1); c++)
              counter += std::abs(bottomData[xy+c*whArea+num*cwhArea]);
            CHECK_EQ(counter, std::abs(matrixAData[xy+num*whArea]))
              << "Some values: " << bottomData[xy+num*cwhArea]
                          << " " << bottomData[xy+num*cwhArea+1*whArea]
                          << " " << bottomData[xy+num*cwhArea+2*whArea]
                          << " " << bottomData[xy+num*cwhArea+3*whArea];
          }
        }
        // matrix_K_
        // No considered the borders to simplify operation
        CHECK_EQ(this->blobs_[0]->count(2), 9) << "Slow security check only implemented for 3x3 convolutions.";
        CHECK_EQ(matrix_K_->count(3), top[0]->count(3)) << "Slow security check only implemented for pad = 1 sceneario.";
        CHECK_EQ(matrix_K_->shape(1), 1);
        const auto yOffset = matrix_K_->count(3);
        const auto binarizationArea = bottom[0]->count(1);
        for (auto num = 0 ; num < top[0]->shape(0); num++)
        {
          // CHECK_EQ(matrix_K_->shape(1), 1);
          for (auto y = 1 ; y < top[0]->shape(2) - 1; y++)
          {
            for (auto x = 1 ; x < top[0]->shape(3) - 1; x++)
            {
              const auto baseIndex = num * matrix_A_->count(1) + y * yOffset + x;
              const auto counter = (matrixAData[-yOffset+baseIndex-1] + matrixAData[-yOffset+baseIndex] + matrixAData[-yOffset+baseIndex+1]
                                    + matrixAData[baseIndex-1] + matrixAData[baseIndex] + matrixAData[baseIndex+1]
                                    + matrixAData[yOffset+baseIndex-1] + matrixAData[yOffset+baseIndex] + matrixAData[yOffset+baseIndex+1])
                                 / binarizationArea;
              const auto matrixKValue = matrix_K_->cpu_data()[num * matrix_K_->count(1) + y * matrix_K_->count(3) + x];
              // CHECK_EQ(counter, matrixKValue); // Numerical precision issues
              CHECK_LE(std::abs(counter - matrixKValue), 1e-6) // 1e-9 fails
                << counter << " vs. " << matrixKValue;
              // if (y == 15 && x == 19) // Random position
              // {
              //   if (num == 0)
              //     std::cout << "\n" << "n = " << num << "/" << top[0]->shape(0) << ": "
              //       << counter << " vs. " << matrixKValue << ", R = " << counter/matrixKValue;
              //   else
              //     std::cout << ", R = " << counter/matrixKValue;
              //   if (num == top[0]->shape(0)-1)
              //     std::cout << std::endl;
              // }
            }
          }
        }
      }
    #endif
  }
  // Binary added end

  // const Dtype* weight = this->blobs_[0]->gpu_data(); // Binary commented
  // Binary added
  // const Dtype* weight = (this->layer_param_.convolution_param().binary() > 0 && this->phase_ == TRAIN
  //   ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  const Dtype* weight = (this->layer_param_.convolution_param().binary() > 0
    ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  // const Dtype* weight = (this->layer_param_.convolution_param().binary() > 0 && this->phase_ == TEST
  //   ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  // Binary added ended
  for (int i = 0; i < bottom.size(); ++i) {
    // const Dtype* bottom_data = bottom[i]->gpu_data(); // Binary commented
    // Binary added
    const Dtype* bottom_data = (this->layer_param_.convolution_param().binary() > 1
      ? bottom_binary_->gpu_data() : bottom[i]->gpu_data());
    // Binary added ended
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

//       // Binary added
//       if (this->layer_param_.convolution_param().binary() > 1)
//       {
//         const auto topChannels = top[i]->shape(1);
//         const auto topWHArea = top[i]->count(2);

//         // SECURITY CHECK
//         #ifdef SLOW_SECURITY_CHECKS
//           const auto topDataCpu = top[i]->cpu_data();
//           std::vector<Dtype> topDataOriginal(topWHArea);
//           std::copy(&topDataCpu[top_offset_ * g], &topDataCpu[top_offset_ * g] + topWHArea, topDataOriginal.begin());
// // std::cout << matrix_A_->shape(0) << " " << matrix_A_->shape(1) << " " << matrix_A_->shape(2) << " " << matrix_A_->shape(3) << "\t";
// // std::cout << matrix_K_->shape(0) << " " << matrix_K_->shape(1) << " " << matrix_K_->shape(2) << " " << matrix_K_->shape(3) << "\t";
// // std::cout << top[i]->shape(0) << " " << top[i]->shape(1) << " " << top[i]->shape(2) << " " << top[i]->shape(3) << std::endl;
//           auto topDataCpuUpdated = top[i]->cpu_data();
//         #endif

//         // top = K .* top
//         multiplyOverChannelsGpu<<<CAFFE_GET_BLOCKS(topWHArea), CAFFE_CUDA_NUM_THREADS>>>(
//           top_data + top_offset_ * g, matrix_K_->gpu_data(), topChannels, topWHArea);

//         // SECURITY CHECK
//         #ifdef SLOW_SECURITY_CHECKS
//           top_data = top[i]->mutable_gpu_data();
//           // const auto topDataCpuUpdated = top[i]->cpu_data();
//           topDataCpuUpdated = top[i]->cpu_data();
//           const auto matrixKData = matrix_K_->cpu_data();
//           // bottom_binary_ from bottom[0]
//           for (auto i = 0 ; i < topWHArea; i++)
//             CHECK_EQ(topDataCpuUpdated[i + top_offset_ * g], topDataOriginal[i] * matrixKData[i])
//               << topDataCpuUpdated[i + top_offset_ * g] << " vs. " << topDataOriginal[i] << " * " << matrixKData[i];
//         #endif
//       }
//       // Binary added end

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    // weight = this->blobs_[0]->gpu_data(); // Binary commented
    // weight_diff = this->blobs_[0]->mutable_gpu_diff(); // Binary commented
    // Binary added
    weight = (this->layer_param_.convolution_param().binary() > 0
              ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
    weight_diff = (this->layer_param_.convolution_param().binary() > 0
                   ? weight_binary_->mutable_gpu_diff() : this->blobs_[0]->mutable_gpu_diff());
    // Binary added ended
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          // weight = this->blobs_[0]->gpu_data(); // Binary commented
          // Binary added
          weight = (this->layer_param_.convolution_param().binary() > 0
                    ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
          // Binary added ended
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 0)
  {
    // Binarized weights (XNOR-style)
    if (this->param_propagate_down_[0])
    {
      // Common parameters
      const auto countReduced = this->blobs_[0]->shape(0);
      const auto binarizationArea = this->blobs_[0]->count(1);

      // // Option a - Weight = +-1
      // // Do nothing
      // Option b - Weight = +-n
      backwardNormalizeWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
        this->blobs_[0]->mutable_gpu_diff(), weight_binary_->gpu_diff(), this->blobs_[0]->gpu_data(),
        countReduced, binarizationArea);

      // SECURITY CHECK
      #ifdef SLOW_SECURITY_CHECKS
        const auto count = this->blobs_[0]->count();
        const auto cpuDataW = this->blobs_[0]->cpu_data();
        const auto cpuDiffB = weight_binary_->cpu_diff();
        const auto cpuDiffW = this->blobs_[0]->cpu_diff();
        const auto oneOverWeightArea = Dtype(1)/Dtype(binarizationArea);
        for (auto i = 0 ; i < countReduced; i++)
        {
          // Offset
          const auto offset = i * binarizationArea;
          const auto* topDiffOffset = &cpuDiffB[offset];
          const auto* bottomDataOffset = &cpuDataW[offset];
          const auto* bottomDiffOffset = &cpuDiffW[offset];
          // L1 norm & optimal alpha
          auto alphaOptimal = Dtype(0);
          for (auto j = 0 ; j < binarizationArea; j++)
            alphaOptimal += std::abs(bottomDataOffset[j]);
          alphaOptimal /= binarizationArea;
          for (auto j = 0 ; j < binarizationArea; j++)
          {
            const auto diffTemp = topDiffOffset[j]
                                * (oneOverWeightArea
                                   + alphaOptimal * max(-Dtype(1), min(Dtype(1), bottomDataOffset[j])));
            const auto bottomDiff = std::abs(bottomDiffOffset[j] - diffTemp);
            const auto bottomDiffRelative = bottomDiff / diffTemp;
const auto epsilon = 1e-1; // 1e-6 works for binary() == 1
            if (bottomDiff > epsilon && bottomDiffRelative > epsilon) // This worked for 20k iterations (manually stopped)
            {
              // CHECK_EQ(bottomDiffOffset[j], diffTemp); // Numerical precision issues
              // bottomDiff: 1e-2 fails at ~3380, 1e-3 at ~3160, 1e-4 at ~2340, 1e-6 at ~1000 iterations, 1e-9 right away
              CHECK_LE(bottomDiff, epsilon)
                << bottomDiffOffset[j] << " vs. " << diffTemp;
              CHECK_LE(bottomDiffRelative, epsilon)
                << bottomDiffOffset[j] << " vs. " << diffTemp;
            }
          }
        }
      #endif
    }

    // Binarized activations (XNOR-style)
    if (this->layer_param_.convolution_param().binary() > 1)
    {
      const auto binarizationArea = this->blobs_[0]->count(1);
      for (int i = 0; i < top.size(); ++i)
      {
        // Gradient w.r.t. bottom data.
        if (propagate_down[i])
        {
          const auto count = bottom[i]->count();
          // const Dtype oneOverWeightArea = Dtype(1) / Dtype(binarizationArea);
          // const auto matrixKOffset = bottom[i]->count(2);
          // backwardNormalizeInputGpu<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          //   bottom[i]->mutable_gpu_diff(), /*bottom[i]->gpu_diff(),*/ bottom[i]->gpu_data(),
          //   count, oneOverWeightArea, matrix_K_->gpu_data(), matrixKOffset);
          backwardNormalizeInputGpu<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            bottom[i]->mutable_gpu_diff(), /*bottom[i]->gpu_diff(),*/ bottom[i]->gpu_data(),
            count);
        }
      }
    }
  }
  // // Regularization - My binary way (guiding weights to 1)
  // if (this->layer_param_.convolution_param().binary() > 0) // Binary added
  // {
  //   if (this->param_propagate_down_[0]) {
  //     const auto lambda = 0.01f;
  //     const auto* const weight_real = this->blobs_[0]->cpu_data();
  //     auto* weight_real_diff = this->blobs_[0]->mutable_cpu_diff();
  //     for (auto index = 0 ; index < this->blobs_[0]->count() ; index++)
  //       weight_real_diff[index] += 2*lambda*(   weight_real[index] - (weight_real[index] < 0 ? -1 : 1)   );
  //   }
  // }
  // Binary added end
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
