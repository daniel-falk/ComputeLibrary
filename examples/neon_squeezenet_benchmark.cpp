/*
    Squeezenet v1.1 (https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1)

    input 227x227x3

    conv1 3x3, 64 filters, stride 2

    relu

    maxpool 3x3, stride 2

    fire2:
        squeeze - conv 1x1, 16 filters
        relu squeeze
        expand1x1 - conv 1x1, 64 filters
        relu expand1x1
        expand3x3 - conv 3x3, 64 filters, pad=1
        relu expand3x3
        concat expand1x1 expand3x3

    fire3:
        squeeze - conv 1x1, 16 filters
        relu squeeze
        expand1x1 - conv 1x1, 64 filters
        relu expand1x1
        expand3x3 - conv 3x3, 64 filters, pad=1
        relu expand3x3
        concat expand1x1 expand3x3

    maxpool 3x3, stride 2

    fire4:
        squeeze - conv 1x1, 32 filters
        relu squeeze
        expand1x1 - conv 1x1, 128 filters
        relu expand1x1
        expand3x3 - conv 3x3, 128 filters, pad=1
        relu expand3x3
        concat  expand1x1 expand3x3

    fire5:
        squeeze - conv 1x1, 32 filters
        relu squeeze
        expand1x1 - conv 1x1, 128 filters
        relu expand1x1
        expand3x3 - conv 3x3, 128 filters, pad=1
        relu expand3x3
        concat  expand1x1 expand3x3

    maxpool 3x3, stride 2

    fire6:
        squeeze - conv 1x1, 48 filters
        relu squeeze
        expand1x1 - conv 1x1, 192 filters
        relu expand1x1
        expand3x3 - conv 3x3, 192 filters, pad=1
        relu expand3x3
        concat  expand1x1 expand3x3

    fire7:
        squeeze - conv 1x1, 48 filters
        relu squeeze
        expand1x1 - conv 1x1, 192 filters
        relu expand1x1
        expand3x3 - conv 3x3, 192 filters, pad=1
        relu expand3x3
        concat  expand1x1 expand3x3

    fire8:
        squeeze - conv 1x1, 64 filters
        relu squeeze
        expand1x1 - conv 1x1, 256 filters
        relu expand1x1
        expand3x3 - conv 3x3, 256 filters, pad=1
        relu expand3x3
        concat  expand1x1 expand3x3

    fire9:
        squeeze - conv 1x1, 64 filters
        relu squeeze
        expand1x1 - conv 1x1, 256 filters
        relu expand1x1
        expand3x3 - conv 3x3, 256 filters, pad=1
        relu expand3x3
        concat  expand1x1 expand3x3

    dropout 0.5

    conv 1x1, 1000 filters

    global average pooling

    softmax
*/

#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"
#include "utils/GraphUtils.h"

#include <time.h>
#include <stdlib.h>

using namespace arm_compute;
using namespace utils;

template <typename F>
void clock_fn(F f) {
    clock_t t_begin = std::clock();
    f();
    clock_t t_end = std::clock();
    double t = double(t_end - t_begin) * 1000 / CLOCKS_PER_SEC;
    std::cout << t << " ms" << std::endl;
}

void load_npy(Tensor &tensor, std::string prefix, std::string name) {
    graph_utils::NumPyBinLoader ldr(prefix + name + ".npy");
    ldr.access_tensor(tensor);
}


/*
 * Time each op individually, elsewise same as neon_squeezenet.cpp
 */
void main_cnn(int argc, const char **argv)
{
    ARM_COMPUTE_UNUSED(argc);
    ARM_COMPUTE_UNUSED(argv);

    // Create NEON allocator
    Allocator allocator;

    // The src tensor should contain the input image
    Tensor src;

    // The weights and biases tensors should be initialized with the values inferred with the training

    // Conv1
    Tensor conv1_weights;
    Tensor conv1_bias;
    Tensor conv1_out;
    Tensor conv1_act_out;

    // Maxpool1
    Tensor max_pool1_out;

    // Fire 2
    Tensor fire2_weights_conv_squeeze;
    Tensor fire2_bias_conv_squeeze;
    Tensor fire2_conv_squeeze_out;
    Tensor fire2_squeeze_act_out;

    Tensor fire2_weights_conv_expand1x1;
    Tensor fire2_bias_conv_expand1x1;
    Tensor fire2_conv_expand1x1_out;
    Tensor fire2_expand1x1_act_out;

    Tensor fire2_weights_conv_expand3x3;
    Tensor fire2_bias_conv_expand3x3;
    Tensor fire2_conv_expand3x3_out;
    Tensor fire2_expand3x3_act_out;

    Tensor fire2_concat_out;

    // Fire 3
    Tensor fire3_weights_conv_squeeze;
    Tensor fire3_bias_conv_squeeze;
    Tensor fire3_conv_squeeze_out;
    Tensor fire3_squeeze_act_out;

    Tensor fire3_weights_conv_expand1x1;
    Tensor fire3_bias_conv_expand1x1;
    Tensor fire3_conv_expand1x1_out;
    Tensor fire3_expand1x1_act_out;

    Tensor fire3_weights_conv_expand3x3;
    Tensor fire3_bias_conv_expand3x3;
    Tensor fire3_conv_expand3x3_out;
    Tensor fire3_expand3x3_act_out;

    Tensor fire3_concat_out;

    // Maxpool2
    Tensor max_pool2_out;

    // Fire 4
    Tensor fire4_weights_conv_squeeze;
    Tensor fire4_bias_conv_squeeze;
    Tensor fire4_conv_squeeze_out;
    Tensor fire4_squeeze_act_out;

    Tensor fire4_weights_conv_expand1x1;
    Tensor fire4_bias_conv_expand1x1;
    Tensor fire4_conv_expand1x1_out;
    Tensor fire4_expand1x1_act_out;

    Tensor fire4_weights_conv_expand3x3;
    Tensor fire4_bias_conv_expand3x3;
    Tensor fire4_conv_expand3x3_out;
    Tensor fire4_expand3x3_act_out;

    Tensor fire4_concat_out;

    // Fire 5
    Tensor fire5_weights_conv_squeeze;
    Tensor fire5_bias_conv_squeeze;
    Tensor fire5_conv_squeeze_out;
    Tensor fire5_squeeze_act_out;

    Tensor fire5_weights_conv_expand1x1;
    Tensor fire5_bias_conv_expand1x1;
    Tensor fire5_conv_expand1x1_out;
    Tensor fire5_expand1x1_act_out;

    Tensor fire5_weights_conv_expand3x3;
    Tensor fire5_bias_conv_expand3x3;
    Tensor fire5_conv_expand3x3_out;
    Tensor fire5_expand3x3_act_out;

    Tensor fire5_concat_out;

    // Maxpool3
    Tensor max_pool3_out;

    // Fire 6
    Tensor fire6_weights_conv_squeeze;
    Tensor fire6_bias_conv_squeeze;
    Tensor fire6_conv_squeeze_out;
    Tensor fire6_squeeze_act_out;

    Tensor fire6_weights_conv_expand1x1;
    Tensor fire6_bias_conv_expand1x1;
    Tensor fire6_conv_expand1x1_out;
    Tensor fire6_expand1x1_act_out;

    Tensor fire6_weights_conv_expand3x3;
    Tensor fire6_bias_conv_expand3x3;
    Tensor fire6_conv_expand3x3_out;
    Tensor fire6_expand3x3_act_out;

    Tensor fire6_concat_out;

    // Fire 7
    Tensor fire7_weights_conv_squeeze;
    Tensor fire7_bias_conv_squeeze;
    Tensor fire7_conv_squeeze_out;
    Tensor fire7_squeeze_act_out;

    Tensor fire7_weights_conv_expand1x1;
    Tensor fire7_bias_conv_expand1x1;
    Tensor fire7_conv_expand1x1_out;
    Tensor fire7_expand1x1_act_out;

    Tensor fire7_weights_conv_expand3x3;
    Tensor fire7_bias_conv_expand3x3;
    Tensor fire7_conv_expand3x3_out;
    Tensor fire7_expand3x3_act_out;

    Tensor fire7_concat_out;

    // Fire 8
    Tensor fire8_weights_conv_squeeze;
    Tensor fire8_bias_conv_squeeze;
    Tensor fire8_conv_squeeze_out;
    Tensor fire8_squeeze_act_out;

    Tensor fire8_weights_conv_expand1x1;
    Tensor fire8_bias_conv_expand1x1;
    Tensor fire8_conv_expand1x1_out;
    Tensor fire8_expand1x1_act_out;

    Tensor fire8_weights_conv_expand3x3;
    Tensor fire8_bias_conv_expand3x3;
    Tensor fire8_conv_expand3x3_out;
    Tensor fire8_expand3x3_act_out;

    Tensor fire8_concat_out;

    // Fire 9
    Tensor fire9_weights_conv_squeeze;
    Tensor fire9_bias_conv_squeeze;
    Tensor fire9_conv_squeeze_out;
    Tensor fire9_squeeze_act_out;

    Tensor fire9_weights_conv_expand1x1;
    Tensor fire9_bias_conv_expand1x1;
    Tensor fire9_conv_expand1x1_out;
    Tensor fire9_expand1x1_act_out;

    Tensor fire9_weights_conv_expand3x3;
    Tensor fire9_bias_conv_expand3x3;
    Tensor fire9_conv_expand3x3_out;
    Tensor fire9_expand3x3_act_out;

    Tensor fire9_concat_out;

    // Conv 10
    Tensor conv10_weights;
    Tensor conv10_bias;
    Tensor conv10_out;
    Tensor conv10_act_out;

    // global avg pool
    Tensor global_avg_pool_out;

    // Flatten
    Tensor flatten_out;

    // Softmax
    Tensor softmax_out;

    // Create layers

    // conv1
    NEConvolutionLayer          conv1;
    NEActivationLayer           conv1_act;

    // maxpool1
    NEPoolingLayer              max_pool1;

    // fire2
    NEConvolutionLayer          fire2_conv_squeeze;
    NEActivationLayer           fire2_act_squeeze;
    NEConvolutionLayer          fire2_conv_expand1x1;
    NEActivationLayer           fire2_act_expand1x1;
    NEConvolutionLayer          fire2_conv_expand3x3;
    NEActivationLayer           fire2_act_expand3x3;
    NEDepthConcatenateLayer     fire2_concat;

    // fire3
    NEConvolutionLayer          fire3_conv_squeeze;
    NEActivationLayer           fire3_act_squeeze;
    NEConvolutionLayer          fire3_conv_expand1x1;
    NEActivationLayer           fire3_act_expand1x1;
    NEConvolutionLayer          fire3_conv_expand3x3;
    NEActivationLayer           fire3_act_expand3x3;
    NEDepthConcatenateLayer     fire3_concat;

    // maxpool2
    NEPoolingLayer              max_pool2;

    // fire4
    NEConvolutionLayer          fire4_conv_squeeze;
    NEActivationLayer           fire4_act_squeeze;
    NEConvolutionLayer          fire4_conv_expand1x1;
    NEActivationLayer           fire4_act_expand1x1;
    NEConvolutionLayer          fire4_conv_expand3x3;
    NEActivationLayer           fire4_act_expand3x3;
    NEDepthConcatenateLayer     fire4_concat;

    // fire5
    NEConvolutionLayer          fire5_conv_squeeze;
    NEActivationLayer           fire5_act_squeeze;
    NEConvolutionLayer          fire5_conv_expand1x1;
    NEActivationLayer           fire5_act_expand1x1;
    NEConvolutionLayer          fire5_conv_expand3x3;
    NEActivationLayer           fire5_act_expand3x3;
    NEDepthConcatenateLayer     fire5_concat;

    // maxpool3
    NEPoolingLayer              max_pool3;

    // fire6
    NEConvolutionLayer          fire6_conv_squeeze;
    NEActivationLayer           fire6_act_squeeze;
    NEConvolutionLayer          fire6_conv_expand1x1;
    NEActivationLayer           fire6_act_expand1x1;
    NEConvolutionLayer          fire6_conv_expand3x3;
    NEActivationLayer           fire6_act_expand3x3;
    NEDepthConcatenateLayer     fire6_concat;

    // fire7
    NEConvolutionLayer          fire7_conv_squeeze;
    NEActivationLayer           fire7_act_squeeze;
    NEConvolutionLayer          fire7_conv_expand1x1;
    NEActivationLayer           fire7_act_expand1x1;
    NEConvolutionLayer          fire7_conv_expand3x3;
    NEActivationLayer           fire7_act_expand3x3;
    NEDepthConcatenateLayer     fire7_concat;

    // fire8

    NEConvolutionLayer          fire8_conv_squeeze;
    NEActivationLayer           fire8_act_squeeze;
    NEConvolutionLayer          fire8_conv_expand1x1;
    NEActivationLayer           fire8_act_expand1x1;
    NEConvolutionLayer          fire8_conv_expand3x3;
    NEActivationLayer           fire8_act_expand3x3;
    NEDepthConcatenateLayer     fire8_concat;

    // fire9
    NEConvolutionLayer          fire9_conv_squeeze;
    NEActivationLayer           fire9_act_squeeze;
    NEConvolutionLayer          fire9_conv_expand1x1;
    NEActivationLayer           fire9_act_expand1x1;
    NEConvolutionLayer          fire9_conv_expand3x3;
    NEActivationLayer           fire9_act_expand3x3;
    NEDepthConcatenateLayer     fire9_concat;

    // conv10
    NEConvolutionLayer          conv10;
    NEActivationLayer           conv10_act;

    // global avg pool
    NEPoolingLayer              global_avg_pool;

    // Flatten layer
    NEFlattenLayer              flatten;
    // Softmax
    NESoftmaxLayer              softmax;


    /* ------------------------------ [Initialize tensors] ------------------------------- */

    // Initialize src tensor
    constexpr unsigned int width_src_image  = 227;
    constexpr unsigned int height_src_image = 227;
    constexpr unsigned int ifm_src_img      = 3;

    const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
    src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

    // Initialize tensors of conv1
    constexpr unsigned int conv1_kernel_x = 3;
    constexpr unsigned int conv1_kernel_y = 3;
    constexpr unsigned int conv1_ofm      = 64; //OFM = OUTPUT FEATURE MAPS = NUM FILTERS

    const TensorShape conv1_weights_shape(conv1_kernel_x, conv1_kernel_y, src_shape.z(), conv1_ofm);
    const TensorShape conv1_bias_shape(conv1_weights_shape[3]);
    const TensorShape conv1_out_shape(src_shape.x(), src_shape.y(), conv1_weights_shape[3]);

    conv1_weights.allocator()->init(TensorInfo(conv1_weights_shape, 1, DataType::F32));
    conv1_bias.allocator()->init(TensorInfo(conv1_bias_shape, 1, DataType::F32));
    conv1_out.allocator()->init(TensorInfo(conv1_out_shape, 1, DataType::F32));
    conv1_act_out.allocator()->init(TensorInfo(conv1_out_shape, 1, DataType::F32));

    // Initialize tensor of maxpool1
    TensorShape max_pool1_shape = conv1_out_shape;
    max_pool1_shape.set(0, (max_pool1_shape.x() - 3 + 2) / 2 /* + 1 */); // O = (W - K + 2P) / S + 1 !??!?!?!?!?!?
    max_pool1_shape.set(1, (max_pool1_shape.y() - 3 + 2) / 2 /* + 1 */);
    max_pool1_out.allocator()->init(TensorInfo(max_pool1_shape, 1, DataType::F32));

    // Initialize tensors of fire2
    constexpr unsigned int fire2_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire2_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire2_squeeze_conv_ofm      = 16;

    const TensorShape fire2_squeeze_conv_weights_shape(fire2_squeeze_conv_kernel_x, fire2_squeeze_conv_kernel_y, max_pool1_shape.z(), fire2_squeeze_conv_ofm);
    const TensorShape fire2_squeeze_conv_bias_shape(fire2_squeeze_conv_weights_shape[3]);
    const TensorShape fire2_squeeze_conv_out_shape(max_pool1_shape.x(), max_pool1_shape.y(), fire2_squeeze_conv_weights_shape[3]);

    fire2_weights_conv_squeeze.allocator()->init(TensorInfo(fire2_squeeze_conv_weights_shape, 1, DataType::F32));
    fire2_bias_conv_squeeze.allocator()->init(TensorInfo(fire2_squeeze_conv_bias_shape, 1, DataType::F32));
    fire2_conv_squeeze_out.allocator()->init(TensorInfo(fire2_squeeze_conv_out_shape, 1, DataType::F32));
    fire2_squeeze_act_out.allocator()->init(TensorInfo(fire2_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire2_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire2_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire2_expand1x1_conv_ofm      = 64;

    const TensorShape fire2_expand1x1_conv_weights_shape(fire2_expand1x1_conv_kernel_x, fire2_expand1x1_conv_kernel_y, fire2_squeeze_conv_out_shape.z(), fire2_expand1x1_conv_ofm);
    const TensorShape fire2_expand1x1_conv_bias_shape(fire2_expand1x1_conv_weights_shape[3]);
    const TensorShape fire2_expand1x1_conv_out_shape(fire2_squeeze_conv_out_shape.x(), fire2_squeeze_conv_out_shape.y(), fire2_expand1x1_conv_weights_shape[3]);

    fire2_weights_conv_expand1x1.allocator()->init(TensorInfo(fire2_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire2_bias_conv_expand1x1.allocator()->init(TensorInfo(fire2_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire2_conv_expand1x1_out.allocator()->init(TensorInfo(fire2_expand1x1_conv_out_shape, 1, DataType::F32));
    fire2_expand1x1_act_out.allocator()->init(TensorInfo(fire2_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire2_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire2_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire2_expand3x3_conv_ofm      = 64;

    const TensorShape fire2_expand3x3_conv_weights_shape(fire2_expand3x3_conv_kernel_x, fire2_expand3x3_conv_kernel_y, fire2_squeeze_conv_out_shape.z(), fire2_expand3x3_conv_ofm);
    const TensorShape fire2_expand3x3_conv_bias_shape(fire2_expand3x3_conv_weights_shape[3]);
    const TensorShape fire2_expand3x3_conv_out_shape(fire2_squeeze_conv_out_shape.x(), fire2_squeeze_conv_out_shape.y(), fire2_expand3x3_conv_weights_shape[3]);

    fire2_weights_conv_expand3x3.allocator()->init(TensorInfo(fire2_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire2_bias_conv_expand3x3.allocator()->init(TensorInfo(fire2_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire2_conv_expand3x3_out.allocator()->init(TensorInfo(fire2_expand3x3_conv_out_shape, 1, DataType::F32));
    fire2_expand3x3_act_out.allocator()->init(TensorInfo(fire2_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire2_concat_out_shape(fire2_expand3x3_conv_out_shape.x(), fire2_expand3x3_conv_out_shape.y(), fire2_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire2_concat_out.allocator()->init(TensorInfo(fire2_concat_out_shape, 1, DataType::F32));


    // Initialize tensors of fire3
    constexpr unsigned int fire3_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire3_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire3_squeeze_conv_ofm      = 16;

    const TensorShape fire3_squeeze_conv_weights_shape(fire3_squeeze_conv_kernel_x, fire3_squeeze_conv_kernel_y, fire2_concat_out_shape.z(), fire3_squeeze_conv_ofm);
    const TensorShape fire3_squeeze_conv_bias_shape(fire3_squeeze_conv_weights_shape[3]);
    const TensorShape fire3_squeeze_conv_out_shape(fire2_concat_out_shape.x(), fire2_concat_out_shape.y(), fire3_squeeze_conv_weights_shape[3]);

    fire3_weights_conv_squeeze.allocator()->init(TensorInfo(fire3_squeeze_conv_weights_shape, 1, DataType::F32));
    fire3_bias_conv_squeeze.allocator()->init(TensorInfo(fire3_squeeze_conv_bias_shape, 1, DataType::F32));
    fire3_conv_squeeze_out.allocator()->init(TensorInfo(fire3_squeeze_conv_out_shape, 1, DataType::F32));
    fire3_squeeze_act_out.allocator()->init(TensorInfo(fire3_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire3_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire3_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire3_expand1x1_conv_ofm      = 64;

    const TensorShape fire3_expand1x1_conv_weights_shape(fire3_expand1x1_conv_kernel_x, fire3_expand1x1_conv_kernel_y, fire3_squeeze_conv_out_shape.z(), fire3_expand1x1_conv_ofm);
    const TensorShape fire3_expand1x1_conv_bias_shape(fire3_expand1x1_conv_weights_shape[3]);
    const TensorShape fire3_expand1x1_conv_out_shape(fire3_squeeze_conv_out_shape.x(), fire3_squeeze_conv_out_shape.y(), fire3_expand1x1_conv_weights_shape[3]);

    fire3_weights_conv_expand1x1.allocator()->init(TensorInfo(fire3_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire3_bias_conv_expand1x1.allocator()->init(TensorInfo(fire3_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire3_conv_expand1x1_out.allocator()->init(TensorInfo(fire3_expand1x1_conv_out_shape, 1, DataType::F32));
    fire3_expand1x1_act_out.allocator()->init(TensorInfo(fire3_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire3_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire3_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire3_expand3x3_conv_ofm      = 64;

    const TensorShape fire3_expand3x3_conv_weights_shape(fire3_expand3x3_conv_kernel_x, fire3_expand3x3_conv_kernel_y, fire3_squeeze_conv_out_shape.z(), fire3_expand3x3_conv_ofm);
    const TensorShape fire3_expand3x3_conv_bias_shape(fire3_expand3x3_conv_weights_shape[3]);
    const TensorShape fire3_expand3x3_conv_out_shape(fire3_squeeze_conv_out_shape.x(), fire3_squeeze_conv_out_shape.y(), fire3_expand3x3_conv_weights_shape[3]);

    fire3_weights_conv_expand3x3.allocator()->init(TensorInfo(fire3_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire3_bias_conv_expand3x3.allocator()->init(TensorInfo(fire3_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire3_conv_expand3x3_out.allocator()->init(TensorInfo(fire3_expand3x3_conv_out_shape, 1, DataType::F32));
    fire3_expand3x3_act_out.allocator()->init(TensorInfo(fire3_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire3_concat_out_shape(fire3_expand3x3_conv_out_shape.x(), fire3_expand3x3_conv_out_shape.y(), fire3_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire3_concat_out.allocator()->init(TensorInfo(fire3_concat_out_shape, 1, DataType::F32));

    // Initialize tensor of maxpool2
    TensorShape max_pool2_shape = fire3_concat_out_shape;
    max_pool2_shape.set(0, (max_pool2_shape.x() - 3) / 2 + 1);
    max_pool2_shape.set(1, (max_pool2_shape.y() - 3) / 2 + 1);
    max_pool2_out.allocator()->init(TensorInfo(max_pool2_shape, 1, DataType::F32));

    // Initialize tensors of fire4
    constexpr unsigned int fire4_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire4_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire4_squeeze_conv_ofm      = 32;

    const TensorShape fire4_squeeze_conv_weights_shape(fire4_squeeze_conv_kernel_x, fire4_squeeze_conv_kernel_y, max_pool2_shape.z(), fire4_squeeze_conv_ofm);
    const TensorShape fire4_squeeze_conv_bias_shape(fire4_squeeze_conv_weights_shape[3]);
    const TensorShape fire4_squeeze_conv_out_shape(max_pool2_shape.x(), max_pool2_shape.y(), fire4_squeeze_conv_weights_shape[3]);

    fire4_weights_conv_squeeze.allocator()->init(TensorInfo(fire4_squeeze_conv_weights_shape, 1, DataType::F32));
    fire4_bias_conv_squeeze.allocator()->init(TensorInfo(fire4_squeeze_conv_bias_shape, 1, DataType::F32));
    fire4_conv_squeeze_out.allocator()->init(TensorInfo(fire4_squeeze_conv_out_shape, 1, DataType::F32));
    fire4_squeeze_act_out.allocator()->init(TensorInfo(fire4_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire4_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire4_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire4_expand1x1_conv_ofm      = 128;

    const TensorShape fire4_expand1x1_conv_weights_shape(fire4_expand1x1_conv_kernel_x, fire4_expand1x1_conv_kernel_y, fire4_squeeze_conv_out_shape.z(), fire4_expand1x1_conv_ofm);
    const TensorShape fire4_expand1x1_conv_bias_shape(fire4_expand1x1_conv_weights_shape[3]);
    const TensorShape fire4_expand1x1_conv_out_shape(fire4_squeeze_conv_out_shape.x(), fire4_squeeze_conv_out_shape.y(), fire4_expand1x1_conv_weights_shape[3]);

    fire4_weights_conv_expand1x1.allocator()->init(TensorInfo(fire4_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire4_bias_conv_expand1x1.allocator()->init(TensorInfo(fire4_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire4_conv_expand1x1_out.allocator()->init(TensorInfo(fire4_expand1x1_conv_out_shape, 1, DataType::F32));
    fire4_expand1x1_act_out.allocator()->init(TensorInfo(fire4_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire4_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire4_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire4_expand3x3_conv_ofm      = 128;

    const TensorShape fire4_expand3x3_conv_weights_shape(fire4_expand3x3_conv_kernel_x, fire4_expand3x3_conv_kernel_y, fire4_squeeze_conv_out_shape.z(), fire4_expand3x3_conv_ofm);
    const TensorShape fire4_expand3x3_conv_bias_shape(fire4_expand3x3_conv_weights_shape[3]);
    const TensorShape fire4_expand3x3_conv_out_shape(fire4_squeeze_conv_out_shape.x(), fire4_squeeze_conv_out_shape.y(), fire4_expand3x3_conv_weights_shape[3]);

    fire4_weights_conv_expand3x3.allocator()->init(TensorInfo(fire4_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire4_bias_conv_expand3x3.allocator()->init(TensorInfo(fire4_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire4_conv_expand3x3_out.allocator()->init(TensorInfo(fire4_expand3x3_conv_out_shape, 1, DataType::F32));
    fire4_expand3x3_act_out.allocator()->init(TensorInfo(fire4_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire4_concat_out_shape(fire4_expand3x3_conv_out_shape.x(), fire4_expand3x3_conv_out_shape.y(), fire4_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire4_concat_out.allocator()->init(TensorInfo(fire4_concat_out_shape, 1, DataType::F32));

    // Initialize tensors of fire5
    constexpr unsigned int fire5_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire5_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire5_squeeze_conv_ofm      = 32;

    const TensorShape fire5_squeeze_conv_weights_shape(fire5_squeeze_conv_kernel_x, fire5_squeeze_conv_kernel_y, fire4_concat_out_shape.z(), fire5_squeeze_conv_ofm);
    const TensorShape fire5_squeeze_conv_bias_shape(fire5_squeeze_conv_weights_shape[3]);
    const TensorShape fire5_squeeze_conv_out_shape(fire4_concat_out_shape.x(), fire4_concat_out_shape.y(), fire5_squeeze_conv_weights_shape[3]);

    fire5_weights_conv_squeeze.allocator()->init(TensorInfo(fire5_squeeze_conv_weights_shape, 1, DataType::F32));
    fire5_bias_conv_squeeze.allocator()->init(TensorInfo(fire5_squeeze_conv_bias_shape, 1, DataType::F32));
    fire5_conv_squeeze_out.allocator()->init(TensorInfo(fire5_squeeze_conv_out_shape, 1, DataType::F32));
    fire5_squeeze_act_out.allocator()->init(TensorInfo(fire5_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire5_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire5_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire5_expand1x1_conv_ofm      = 128;

    const TensorShape fire5_expand1x1_conv_weights_shape(fire5_expand1x1_conv_kernel_x, fire5_expand1x1_conv_kernel_y, fire5_squeeze_conv_out_shape.z(), fire5_expand1x1_conv_ofm);
    const TensorShape fire5_expand1x1_conv_bias_shape(fire5_expand1x1_conv_weights_shape[3]);
    const TensorShape fire5_expand1x1_conv_out_shape(fire5_squeeze_conv_out_shape.x(), fire5_squeeze_conv_out_shape.y(), fire5_expand1x1_conv_weights_shape[3]);

    fire5_weights_conv_expand1x1.allocator()->init(TensorInfo(fire5_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire5_bias_conv_expand1x1.allocator()->init(TensorInfo(fire5_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire5_conv_expand1x1_out.allocator()->init(TensorInfo(fire5_expand1x1_conv_out_shape, 1, DataType::F32));
    fire5_expand1x1_act_out.allocator()->init(TensorInfo(fire5_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire5_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire5_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire5_expand3x3_conv_ofm      = 128;

    const TensorShape fire5_expand3x3_conv_weights_shape(fire5_expand3x3_conv_kernel_x, fire5_expand3x3_conv_kernel_y, fire5_squeeze_conv_out_shape.z(), fire5_expand3x3_conv_ofm);
    const TensorShape fire5_expand3x3_conv_bias_shape(fire5_expand3x3_conv_weights_shape[3]);
    const TensorShape fire5_expand3x3_conv_out_shape(fire5_squeeze_conv_out_shape.x(), fire5_squeeze_conv_out_shape.y(), fire5_expand3x3_conv_weights_shape[3]);

    fire5_weights_conv_expand3x3.allocator()->init(TensorInfo(fire5_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire5_bias_conv_expand3x3.allocator()->init(TensorInfo(fire5_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire5_conv_expand3x3_out.allocator()->init(TensorInfo(fire5_expand3x3_conv_out_shape, 1, DataType::F32));
    fire5_expand3x3_act_out.allocator()->init(TensorInfo(fire5_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire5_concat_out_shape(fire5_expand3x3_conv_out_shape.x(), fire5_expand3x3_conv_out_shape.y(), fire5_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire5_concat_out.allocator()->init(TensorInfo(fire5_concat_out_shape, 1, DataType::F32));

    // Initialize tensor of maxpool3
    TensorShape max_pool3_shape = fire5_concat_out_shape;
    max_pool3_shape.set(0, (max_pool2_shape.x() - 3) / 2 + 1);
    max_pool3_shape.set(1, (max_pool2_shape.y() - 3) / 2 + 1);
    max_pool3_out.allocator()->init(TensorInfo(max_pool3_shape, 1, DataType::F32));

    // Initialize tensors of fire6
    constexpr unsigned int fire6_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire6_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire6_squeeze_conv_ofm      = 48;

    const TensorShape fire6_squeeze_conv_weights_shape(fire6_squeeze_conv_kernel_x, fire6_squeeze_conv_kernel_y, max_pool3_shape.z(), fire6_squeeze_conv_ofm);
    const TensorShape fire6_squeeze_conv_bias_shape(fire6_squeeze_conv_weights_shape[3]);
    const TensorShape fire6_squeeze_conv_out_shape(max_pool3_shape.x(), max_pool3_shape.y(), fire6_squeeze_conv_weights_shape[3]);

    fire6_weights_conv_squeeze.allocator()->init(TensorInfo(fire6_squeeze_conv_weights_shape, 1, DataType::F32));
    fire6_bias_conv_squeeze.allocator()->init(TensorInfo(fire6_squeeze_conv_bias_shape, 1, DataType::F32));
    fire6_conv_squeeze_out.allocator()->init(TensorInfo(fire6_squeeze_conv_out_shape, 1, DataType::F32));
    fire6_squeeze_act_out.allocator()->init(TensorInfo(fire6_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire6_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire6_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire6_expand1x1_conv_ofm      = 192;

    const TensorShape fire6_expand1x1_conv_weights_shape(fire6_expand1x1_conv_kernel_x, fire6_expand1x1_conv_kernel_y, fire6_squeeze_conv_out_shape.z(), fire6_expand1x1_conv_ofm);
    const TensorShape fire6_expand1x1_conv_bias_shape(fire6_expand1x1_conv_weights_shape[3]);
    const TensorShape fire6_expand1x1_conv_out_shape(fire6_squeeze_conv_out_shape.x(), fire6_squeeze_conv_out_shape.y(), fire6_expand1x1_conv_weights_shape[3]);

    fire6_weights_conv_expand1x1.allocator()->init(TensorInfo(fire6_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire6_bias_conv_expand1x1.allocator()->init(TensorInfo(fire6_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire6_conv_expand1x1_out.allocator()->init(TensorInfo(fire6_expand1x1_conv_out_shape, 1, DataType::F32));
    fire6_expand1x1_act_out.allocator()->init(TensorInfo(fire6_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire6_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire6_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire6_expand3x3_conv_ofm      = 192;

    const TensorShape fire6_expand3x3_conv_weights_shape(fire6_expand3x3_conv_kernel_x, fire6_expand3x3_conv_kernel_y, fire6_squeeze_conv_out_shape.z(), fire6_expand3x3_conv_ofm);
    const TensorShape fire6_expand3x3_conv_bias_shape(fire6_expand3x3_conv_weights_shape[3]);
    const TensorShape fire6_expand3x3_conv_out_shape(fire6_squeeze_conv_out_shape.x(), fire6_squeeze_conv_out_shape.y(), fire6_expand3x3_conv_weights_shape[3]);

    fire6_weights_conv_expand3x3.allocator()->init(TensorInfo(fire6_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire6_bias_conv_expand3x3.allocator()->init(TensorInfo(fire6_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire6_conv_expand3x3_out.allocator()->init(TensorInfo(fire6_expand3x3_conv_out_shape, 1, DataType::F32));
    fire6_expand3x3_act_out.allocator()->init(TensorInfo(fire6_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire6_concat_out_shape(fire6_expand3x3_conv_out_shape.x(), fire6_expand3x3_conv_out_shape.y(), fire6_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire6_concat_out.allocator()->init(TensorInfo(fire6_concat_out_shape, 1, DataType::F32));

    // Initialize tensors of fire7
    constexpr unsigned int fire7_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire7_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire7_squeeze_conv_ofm      = 48;

    const TensorShape fire7_squeeze_conv_weights_shape(fire7_squeeze_conv_kernel_x, fire7_squeeze_conv_kernel_y, fire6_concat_out_shape.z(), fire7_squeeze_conv_ofm);
    const TensorShape fire7_squeeze_conv_bias_shape(fire7_squeeze_conv_weights_shape[3]);
    const TensorShape fire7_squeeze_conv_out_shape(fire6_concat_out_shape.x(), fire6_concat_out_shape.y(), fire7_squeeze_conv_weights_shape[3]);

    fire7_weights_conv_squeeze.allocator()->init(TensorInfo(fire7_squeeze_conv_weights_shape, 1, DataType::F32));
    fire7_bias_conv_squeeze.allocator()->init(TensorInfo(fire7_squeeze_conv_bias_shape, 1, DataType::F32));
    fire7_conv_squeeze_out.allocator()->init(TensorInfo(fire7_squeeze_conv_out_shape, 1, DataType::F32));
    fire7_squeeze_act_out.allocator()->init(TensorInfo(fire7_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire7_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire7_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire7_expand1x1_conv_ofm      = 192;

    const TensorShape fire7_expand1x1_conv_weights_shape(fire7_expand1x1_conv_kernel_x, fire7_expand1x1_conv_kernel_y, fire7_squeeze_conv_out_shape.z(), fire7_expand1x1_conv_ofm);
    const TensorShape fire7_expand1x1_conv_bias_shape(fire7_expand1x1_conv_weights_shape[3]);
    const TensorShape fire7_expand1x1_conv_out_shape(fire7_squeeze_conv_out_shape.x(), fire7_squeeze_conv_out_shape.y(), fire7_expand1x1_conv_weights_shape[3]);

    fire7_weights_conv_expand1x1.allocator()->init(TensorInfo(fire7_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire7_bias_conv_expand1x1.allocator()->init(TensorInfo(fire7_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire7_conv_expand1x1_out.allocator()->init(TensorInfo(fire7_expand1x1_conv_out_shape, 1, DataType::F32));
    fire7_expand1x1_act_out.allocator()->init(TensorInfo(fire7_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire7_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire7_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire7_expand3x3_conv_ofm      = 192;

    const TensorShape fire7_expand3x3_conv_weights_shape(fire7_expand3x3_conv_kernel_x, fire7_expand3x3_conv_kernel_y, fire7_squeeze_conv_out_shape.z(), fire7_expand3x3_conv_ofm);
    const TensorShape fire7_expand3x3_conv_bias_shape(fire7_expand3x3_conv_weights_shape[3]);
    const TensorShape fire7_expand3x3_conv_out_shape(fire7_squeeze_conv_out_shape.x(), fire7_squeeze_conv_out_shape.y(), fire7_expand3x3_conv_weights_shape[3]);

    fire7_weights_conv_expand3x3.allocator()->init(TensorInfo(fire7_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire7_bias_conv_expand3x3.allocator()->init(TensorInfo(fire7_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire7_conv_expand3x3_out.allocator()->init(TensorInfo(fire7_expand3x3_conv_out_shape, 1, DataType::F32));
    fire7_expand3x3_act_out.allocator()->init(TensorInfo(fire7_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire7_concat_out_shape(fire7_expand3x3_conv_out_shape.x(), fire7_expand3x3_conv_out_shape.y(), fire7_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire7_concat_out.allocator()->init(TensorInfo(fire7_concat_out_shape, 1, DataType::F32));

    // Initialize tensors of fire8
    constexpr unsigned int fire8_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire8_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire8_squeeze_conv_ofm      = 64;

    const TensorShape fire8_squeeze_conv_weights_shape(fire8_squeeze_conv_kernel_x, fire8_squeeze_conv_kernel_y, fire7_concat_out_shape.z(), fire8_squeeze_conv_ofm);
    const TensorShape fire8_squeeze_conv_bias_shape(fire8_squeeze_conv_weights_shape[3]);
    const TensorShape fire8_squeeze_conv_out_shape(fire7_concat_out_shape.x(), fire7_concat_out_shape.y(), fire8_squeeze_conv_weights_shape[3]);

    fire8_weights_conv_squeeze.allocator()->init(TensorInfo(fire8_squeeze_conv_weights_shape, 1, DataType::F32));
    fire8_bias_conv_squeeze.allocator()->init(TensorInfo(fire8_squeeze_conv_bias_shape, 1, DataType::F32));
    fire8_conv_squeeze_out.allocator()->init(TensorInfo(fire8_squeeze_conv_out_shape, 1, DataType::F32));
    fire8_squeeze_act_out.allocator()->init(TensorInfo(fire8_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire8_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire8_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire8_expand1x1_conv_ofm      = 256;

    const TensorShape fire8_expand1x1_conv_weights_shape(fire8_expand1x1_conv_kernel_x, fire8_expand1x1_conv_kernel_y, fire8_squeeze_conv_out_shape.z(), fire8_expand1x1_conv_ofm);
    const TensorShape fire8_expand1x1_conv_bias_shape(fire8_expand1x1_conv_weights_shape[3]);
    const TensorShape fire8_expand1x1_conv_out_shape(fire8_squeeze_conv_out_shape.x(), fire8_squeeze_conv_out_shape.y(), fire8_expand1x1_conv_weights_shape[3]);

    fire8_weights_conv_expand1x1.allocator()->init(TensorInfo(fire8_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire8_bias_conv_expand1x1.allocator()->init(TensorInfo(fire8_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire8_conv_expand1x1_out.allocator()->init(TensorInfo(fire8_expand1x1_conv_out_shape, 1, DataType::F32));
    fire8_expand1x1_act_out.allocator()->init(TensorInfo(fire8_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire8_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire8_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire8_expand3x3_conv_ofm      = 256;

    const TensorShape fire8_expand3x3_conv_weights_shape(fire8_expand3x3_conv_kernel_x, fire8_expand3x3_conv_kernel_y, fire8_squeeze_conv_out_shape.z(), fire8_expand3x3_conv_ofm);
    const TensorShape fire8_expand3x3_conv_bias_shape(fire8_expand3x3_conv_weights_shape[3]);
    const TensorShape fire8_expand3x3_conv_out_shape(fire8_squeeze_conv_out_shape.x(), fire8_squeeze_conv_out_shape.y(), fire8_expand3x3_conv_weights_shape[3]);

    fire8_weights_conv_expand3x3.allocator()->init(TensorInfo(fire8_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire8_bias_conv_expand3x3.allocator()->init(TensorInfo(fire8_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire8_conv_expand3x3_out.allocator()->init(TensorInfo(fire8_expand3x3_conv_out_shape, 1, DataType::F32));
    fire8_expand3x3_act_out.allocator()->init(TensorInfo(fire8_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire8_concat_out_shape(fire8_expand3x3_conv_out_shape.x(), fire8_expand3x3_conv_out_shape.y(), fire8_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire8_concat_out.allocator()->init(TensorInfo(fire8_concat_out_shape, 1, DataType::F32));

    // Initialize tensors of fire9
    constexpr unsigned int fire9_squeeze_conv_kernel_x = 1;
    constexpr unsigned int fire9_squeeze_conv_kernel_y = 1;
    constexpr unsigned int fire9_squeeze_conv_ofm      = 64;

    const TensorShape fire9_squeeze_conv_weights_shape(fire9_squeeze_conv_kernel_x, fire9_squeeze_conv_kernel_y, fire8_concat_out_shape.z(), fire9_squeeze_conv_ofm);
    const TensorShape fire9_squeeze_conv_bias_shape(fire9_squeeze_conv_weights_shape[3]);
    const TensorShape fire9_squeeze_conv_out_shape(fire8_concat_out_shape.x(), fire8_concat_out_shape.y(), fire9_squeeze_conv_weights_shape[3]);

    fire9_weights_conv_squeeze.allocator()->init(TensorInfo(fire9_squeeze_conv_weights_shape, 1, DataType::F32));
    fire9_bias_conv_squeeze.allocator()->init(TensorInfo(fire9_squeeze_conv_bias_shape, 1, DataType::F32));
    fire9_conv_squeeze_out.allocator()->init(TensorInfo(fire9_squeeze_conv_out_shape, 1, DataType::F32));
    fire9_squeeze_act_out.allocator()->init(TensorInfo(fire9_squeeze_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire9_expand1x1_conv_kernel_x = 1;
    constexpr unsigned int fire9_expand1x1_conv_kernel_y = 1;
    constexpr unsigned int fire9_expand1x1_conv_ofm      = 256;

    const TensorShape fire9_expand1x1_conv_weights_shape(fire9_expand1x1_conv_kernel_x, fire9_expand1x1_conv_kernel_y, fire9_squeeze_conv_out_shape.z(), fire9_expand1x1_conv_ofm);
    const TensorShape fire9_expand1x1_conv_bias_shape(fire9_expand1x1_conv_weights_shape[3]);
    const TensorShape fire9_expand1x1_conv_out_shape(fire9_squeeze_conv_out_shape.x(), fire9_squeeze_conv_out_shape.y(), fire9_expand1x1_conv_weights_shape[3]);

    fire9_weights_conv_expand1x1.allocator()->init(TensorInfo(fire9_expand1x1_conv_weights_shape, 1, DataType::F32));
    fire9_bias_conv_expand1x1.allocator()->init(TensorInfo(fire9_expand1x1_conv_bias_shape, 1, DataType::F32));
    fire9_conv_expand1x1_out.allocator()->init(TensorInfo(fire9_expand1x1_conv_out_shape, 1, DataType::F32));
    fire9_expand1x1_act_out.allocator()->init(TensorInfo(fire9_expand1x1_conv_out_shape, 1, DataType::F32));

    constexpr unsigned int fire9_expand3x3_conv_kernel_x = 3;
    constexpr unsigned int fire9_expand3x3_conv_kernel_y = 3;
    constexpr unsigned int fire9_expand3x3_conv_ofm      = 256;

    const TensorShape fire9_expand3x3_conv_weights_shape(fire9_expand3x3_conv_kernel_x, fire9_expand3x3_conv_kernel_y, fire9_squeeze_conv_out_shape.z(), fire9_expand3x3_conv_ofm);
    const TensorShape fire9_expand3x3_conv_bias_shape(fire9_expand3x3_conv_weights_shape[3]);
    const TensorShape fire9_expand3x3_conv_out_shape(fire9_squeeze_conv_out_shape.x(), fire9_squeeze_conv_out_shape.y(), fire9_expand3x3_conv_weights_shape[3]);

    fire9_weights_conv_expand3x3.allocator()->init(TensorInfo(fire9_expand3x3_conv_weights_shape, 1, DataType::F32));
    fire9_bias_conv_expand3x3.allocator()->init(TensorInfo(fire9_expand3x3_conv_bias_shape, 1, DataType::F32));
    fire9_conv_expand3x3_out.allocator()->init(TensorInfo(fire9_expand3x3_conv_out_shape, 1, DataType::F32));
    fire9_expand3x3_act_out.allocator()->init(TensorInfo(fire9_expand3x3_conv_out_shape, 1, DataType::F32));

    const TensorShape fire9_concat_out_shape(fire9_expand3x3_conv_out_shape.x(), fire9_expand3x3_conv_out_shape.y(), fire9_expand3x3_conv_out_shape.z()*2); //double feature maps because of depthwise concatenation (expand1x1+expand3x3)

    fire9_concat_out.allocator()->init(TensorInfo(fire9_concat_out_shape, 1, DataType::F32));

    // Initialize tensors of conv10
    constexpr unsigned int conv10_kernel_x = 1;
    constexpr unsigned int conv10_kernel_y = 1;
    constexpr unsigned int conv10_ofm      = 1000; //OFM = OUTPUT FEATURE MAPS = NUM FILTERS

    const TensorShape conv10_weights_shape(conv10_kernel_x, conv10_kernel_y, fire8_concat_out_shape.z(), conv10_ofm);
    const TensorShape conv10_bias_shape(conv10_weights_shape[3]);
    const TensorShape conv10_out_shape(fire8_concat_out_shape.x(), fire8_concat_out_shape.y(), conv10_weights_shape[3]);

    conv10_weights.allocator()->init(TensorInfo(conv10_weights_shape, 1, DataType::F32));
    conv10_bias.allocator()->init(TensorInfo(conv10_bias_shape, 1, DataType::F32));
    conv10_out.allocator()->init(TensorInfo(conv10_out_shape, 1, DataType::F32));
    conv10_act_out.allocator()->init(TensorInfo(conv10_out_shape, 1, DataType::F32));

    // Initialize tensors of global_avg_pool
    const TensorShape global_avg_pool_shape(conv10_kernel_x, conv10_kernel_y, conv10_out_shape.z()); //global avg pool = vector of avg values, one per featuremap
    global_avg_pool_out.allocator()->init(TensorInfo(global_avg_pool_shape, 1, DataType::F32));

    // Flatten layer for 1x1x1000 -> 1000
    constexpr unsigned int num_labels = 1000;
    const TensorShape flatten_shape(num_labels);
    flatten_out.allocator()->init(TensorInfo(flatten_shape, 1, DataType::F32));

    // Initialize tensor of softmax
    const TensorShape softmax_shape(num_labels);
    softmax_out.allocator()->init(TensorInfo(softmax_shape, 1, DataType::F32));

    /* ------------------------------ [Configure functions] ----------------------------------- */

    //conv1 - in: 227x227x3: 3x3 convolution, 64 output feature maps stride 2
    conv1.configure(&src, &conv1_weights, &conv1_bias, &conv1_out, PadStrideInfo(1, 1, 1, 1)); //STRIDE_X, STRIDE_Y, PAD_X, PAD_Y
    conv1_act.configure(&conv1_out, &conv1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //maxpool1 - in: 3x3, stride 2
    max_pool1.configure(&conv1_act_out, &max_pool1_out, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2,2)));

    //fire2 - squeeze conv 1x1 16 filters, expand 1x1 64, expand 3x3 64 (PAD=1)
    fire2_conv_squeeze.configure(&max_pool1_out, &fire2_weights_conv_squeeze, &fire2_bias_conv_squeeze, &fire2_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire2_act_squeeze.configure(&fire2_conv_squeeze_out, &fire2_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire2_conv_expand1x1.configure(&fire2_squeeze_act_out, &fire2_weights_conv_expand1x1, &fire2_bias_conv_expand1x1, &fire2_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire2_act_expand1x1.configure(&fire2_conv_expand1x1_out, &fire2_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire2_conv_expand3x3.configure(&fire2_squeeze_act_out, &fire2_weights_conv_expand3x3, &fire2_bias_conv_expand3x3, &fire2_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire2_act_expand3x3.configure(&fire2_conv_expand3x3_out, &fire2_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire2_concat2;
    fire2_concat2.push_back(&fire2_expand1x1_act_out);
    fire2_concat2.push_back(&fire2_expand3x3_act_out);

    fire2_concat.configure(fire2_concat2, &fire2_concat_out);

    //fire3 - squeeze conv 1x1 16 filters, expand 1x1 64, expand 3x3 64 (PAD=1)
    fire3_conv_squeeze.configure(&fire2_concat_out, &fire3_weights_conv_squeeze, &fire3_bias_conv_squeeze, &fire3_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire3_act_squeeze.configure(&fire3_conv_squeeze_out, &fire3_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire3_conv_expand1x1.configure(&fire3_squeeze_act_out, &fire3_weights_conv_expand1x1, &fire3_bias_conv_expand1x1, &fire3_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire3_act_expand1x1.configure(&fire3_conv_expand1x1_out, &fire3_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire3_conv_expand3x3.configure(&fire3_squeeze_act_out, &fire3_weights_conv_expand3x3, &fire3_bias_conv_expand3x3, &fire3_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire3_act_expand3x3.configure(&fire3_conv_expand3x3_out, &fire3_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire3_concat2;
    fire3_concat2.push_back(&fire3_expand1x1_act_out);
    fire3_concat2.push_back(&fire3_expand3x3_act_out);

    fire3_concat.configure(fire3_concat2, &fire3_concat_out);


    //maxpool2 - in: 3x3, stride 2
    max_pool2.configure(&fire3_concat_out, &max_pool2_out, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2,2)));

    //fire4

    fire4_conv_squeeze.configure(&max_pool2_out, &fire4_weights_conv_squeeze, &fire4_bias_conv_squeeze, &fire4_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire4_act_squeeze.configure(&fire4_conv_squeeze_out, &fire4_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire4_conv_expand1x1.configure(&fire4_squeeze_act_out, &fire4_weights_conv_expand1x1, &fire4_bias_conv_expand1x1, &fire4_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire4_act_expand1x1.configure(&fire4_conv_expand1x1_out, &fire4_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire4_conv_expand3x3.configure(&fire4_squeeze_act_out, &fire4_weights_conv_expand3x3, &fire4_bias_conv_expand3x3, &fire4_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire4_act_expand3x3.configure(&fire4_conv_expand3x3_out, &fire4_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire4_concat2;
    fire4_concat2.push_back(&fire4_expand1x1_act_out);
    fire4_concat2.push_back(&fire4_expand3x3_act_out);

    fire4_concat.configure(fire4_concat2, &fire4_concat_out);

    //fire5

    fire5_conv_squeeze.configure(&fire4_concat_out, &fire5_weights_conv_squeeze, &fire5_bias_conv_squeeze, &fire5_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire5_act_squeeze.configure(&fire5_conv_squeeze_out, &fire5_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire5_conv_expand1x1.configure(&fire5_squeeze_act_out, &fire5_weights_conv_expand1x1, &fire5_bias_conv_expand1x1, &fire5_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire5_act_expand1x1.configure(&fire5_conv_expand1x1_out, &fire5_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire5_conv_expand3x3.configure(&fire5_squeeze_act_out, &fire5_weights_conv_expand3x3, &fire5_bias_conv_expand3x3, &fire5_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire5_act_expand3x3.configure(&fire5_conv_expand3x3_out, &fire5_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire5_concat2;
    fire5_concat2.push_back(&fire5_expand1x1_act_out);
    fire5_concat2.push_back(&fire5_expand3x3_act_out);

    fire5_concat.configure(fire5_concat2, &fire5_concat_out);

    //maxpool3 - in: 3x3, stride 2
    max_pool3.configure(&fire5_concat_out, &max_pool3_out, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2,2)));

    //fire6

    fire6_conv_squeeze.configure(&max_pool3_out, &fire6_weights_conv_squeeze, &fire6_bias_conv_squeeze, &fire6_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire6_act_squeeze.configure(&fire6_conv_squeeze_out, &fire6_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire6_conv_expand1x1.configure(&fire6_squeeze_act_out, &fire6_weights_conv_expand1x1, &fire6_bias_conv_expand1x1, &fire6_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire6_act_expand1x1.configure(&fire6_conv_expand1x1_out, &fire6_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire6_conv_expand3x3.configure(&fire6_squeeze_act_out, &fire6_weights_conv_expand3x3, &fire6_bias_conv_expand3x3, &fire6_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire6_act_expand3x3.configure(&fire6_conv_expand3x3_out, &fire6_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire6_concat2;
    fire6_concat2.push_back(&fire6_expand1x1_act_out);
    fire6_concat2.push_back(&fire6_expand3x3_act_out);

    fire6_concat.configure(fire6_concat2, &fire6_concat_out);

    //fire7

    fire7_conv_squeeze.configure(&fire6_concat_out, &fire7_weights_conv_squeeze, &fire7_bias_conv_squeeze, &fire7_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire7_act_squeeze.configure(&fire7_conv_squeeze_out, &fire7_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire7_conv_expand1x1.configure(&fire7_squeeze_act_out, &fire7_weights_conv_expand1x1, &fire7_bias_conv_expand1x1, &fire7_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire7_act_expand1x1.configure(&fire7_conv_expand1x1_out, &fire7_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire7_conv_expand3x3.configure(&fire7_squeeze_act_out, &fire7_weights_conv_expand3x3, &fire7_bias_conv_expand3x3, &fire7_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire7_act_expand3x3.configure(&fire7_conv_expand3x3_out, &fire7_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire7_concat2;
    fire7_concat2.push_back(&fire7_expand1x1_act_out);
    fire7_concat2.push_back(&fire7_expand3x3_act_out);

    fire7_concat.configure(fire7_concat2, &fire7_concat_out);

    //fire8

    fire8_conv_squeeze.configure(&fire7_concat_out, &fire8_weights_conv_squeeze, &fire8_bias_conv_squeeze, &fire8_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire8_act_squeeze.configure(&fire8_conv_squeeze_out, &fire8_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire8_conv_expand1x1.configure(&fire8_squeeze_act_out, &fire8_weights_conv_expand1x1, &fire8_bias_conv_expand1x1, &fire8_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire8_act_expand1x1.configure(&fire8_conv_expand1x1_out, &fire8_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire8_conv_expand3x3.configure(&fire8_squeeze_act_out, &fire8_weights_conv_expand3x3, &fire8_bias_conv_expand3x3, &fire8_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire8_act_expand3x3.configure(&fire8_conv_expand3x3_out, &fire8_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire8_concat2;
    fire8_concat2.push_back(&fire8_expand1x1_act_out);
    fire8_concat2.push_back(&fire8_expand3x3_act_out);

    fire8_concat.configure(fire8_concat2, &fire8_concat_out);

    //fire9

    fire9_conv_squeeze.configure(&fire8_concat_out, &fire9_weights_conv_squeeze, &fire9_bias_conv_squeeze, &fire9_conv_squeeze_out, PadStrideInfo(1,1,0,0));
    fire9_act_squeeze.configure(&fire9_conv_squeeze_out, &fire9_squeeze_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire9_conv_expand1x1.configure(&fire9_squeeze_act_out, &fire9_weights_conv_expand1x1, &fire9_bias_conv_expand1x1, &fire9_conv_expand1x1_out, PadStrideInfo(1,1,0,0));
    fire9_act_expand1x1.configure(&fire9_conv_expand1x1_out, &fire9_expand1x1_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    fire9_conv_expand3x3.configure(&fire9_squeeze_act_out, &fire9_weights_conv_expand3x3, &fire9_bias_conv_expand3x3, &fire9_conv_expand3x3_out, PadStrideInfo(1,1,1,1)); //PAD=1
    fire9_act_expand3x3.configure(&fire9_conv_expand3x3_out, &fire9_expand3x3_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //input 2 tensors, output 1 tensor
    std::vector<arm_compute::ITensor*> fire9_concat2;
    fire9_concat2.push_back(&fire9_expand1x1_act_out);
    fire9_concat2.push_back(&fire9_expand3x3_act_out);

    fire9_concat.configure(fire9_concat2, &fire9_concat_out);

    //conv10
    conv10.configure(&fire9_concat_out, &conv10_weights, &conv10_bias, &conv10_out, PadStrideInfo(1,1,0,0));
    conv10_act.configure(&conv10_out, &conv10_act_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //global avg pool
    global_avg_pool.configure(&conv10_out, &global_avg_pool_out, PoolingLayerInfo(PoolingType::AVG));

    //Flatten
    flatten.configure(&global_avg_pool_out, &flatten_out);
    //Softmax
    softmax.configure(&flatten_out, &softmax_out);


    /* ----------------------- [ Allocate Tensor memory ] ---------------------------- */

    src.allocator() -> allocate();

    conv1_weights.allocator() -> allocate();
    conv1_bias.allocator() -> allocate();
    conv1_out.allocator() -> allocate();
    conv1_act_out.allocator() -> allocate();

    max_pool1_out.allocator() -> allocate();

    fire2_weights_conv_squeeze.allocator() -> allocate();
    fire2_bias_conv_squeeze.allocator() -> allocate();
    fire2_conv_squeeze_out.allocator() -> allocate();
    fire2_squeeze_act_out.allocator() -> allocate();

    fire2_weights_conv_expand1x1.allocator() -> allocate();
    fire2_bias_conv_expand1x1.allocator() -> allocate();
    fire2_conv_expand1x1_out.allocator() -> allocate();
    fire2_expand1x1_act_out.allocator() -> allocate();

    fire2_weights_conv_expand3x3.allocator() -> allocate();
    fire2_bias_conv_expand3x3.allocator() -> allocate();
    fire2_conv_expand3x3_out.allocator() -> allocate();
    fire2_expand3x3_act_out.allocator() -> allocate();

    fire2_concat_out.allocator() -> allocate();

    fire3_weights_conv_squeeze.allocator() -> allocate();
    fire3_bias_conv_squeeze.allocator() -> allocate();
    fire3_conv_squeeze_out.allocator() -> allocate();
    fire3_squeeze_act_out.allocator() -> allocate();

    fire3_weights_conv_expand1x1.allocator() -> allocate();
    fire3_bias_conv_expand1x1.allocator() -> allocate();
    fire3_conv_expand1x1_out.allocator() -> allocate();
    fire3_expand1x1_act_out.allocator() -> allocate();

    fire3_weights_conv_expand3x3.allocator() -> allocate();
    fire3_bias_conv_expand3x3.allocator() -> allocate();
    fire3_conv_expand3x3_out.allocator() -> allocate();
    fire3_expand3x3_act_out.allocator() -> allocate();

    fire3_concat_out.allocator() -> allocate();

    max_pool2_out.allocator() -> allocate();

    fire4_weights_conv_squeeze.allocator() -> allocate();
    fire4_bias_conv_squeeze.allocator() -> allocate();
    fire4_conv_squeeze_out.allocator() -> allocate();
    fire4_squeeze_act_out.allocator() -> allocate();

    fire4_weights_conv_expand1x1.allocator() -> allocate();
    fire4_bias_conv_expand1x1.allocator() -> allocate();
    fire4_conv_expand1x1_out.allocator() -> allocate();
    fire4_expand1x1_act_out.allocator() -> allocate();

    fire4_weights_conv_expand3x3.allocator() -> allocate();
    fire4_bias_conv_expand3x3.allocator() -> allocate();
    fire4_conv_expand3x3_out.allocator() -> allocate();
    fire4_expand3x3_act_out.allocator() -> allocate();

    fire4_concat_out.allocator() -> allocate();

    fire5_weights_conv_squeeze.allocator() -> allocate();
    fire5_bias_conv_squeeze.allocator() -> allocate();
    fire5_conv_squeeze_out.allocator() -> allocate();
    fire5_squeeze_act_out.allocator() -> allocate();

    fire5_weights_conv_expand1x1.allocator() -> allocate();
    fire5_bias_conv_expand1x1.allocator() -> allocate();
    fire5_conv_expand1x1_out.allocator() -> allocate();
    fire5_expand1x1_act_out.allocator() -> allocate();

    fire5_weights_conv_expand3x3.allocator() -> allocate();
    fire5_bias_conv_expand3x3.allocator() -> allocate();
    fire5_conv_expand3x3_out.allocator() -> allocate();
    fire5_expand3x3_act_out.allocator() -> allocate();

    fire5_concat_out.allocator() -> allocate();

    max_pool3_out.allocator() -> allocate();

    fire6_weights_conv_squeeze.allocator() -> allocate();
    fire6_bias_conv_squeeze.allocator() -> allocate();
    fire6_conv_squeeze_out.allocator() -> allocate();
    fire6_squeeze_act_out.allocator() -> allocate();

    fire6_weights_conv_expand1x1.allocator() -> allocate();
    fire6_bias_conv_expand1x1.allocator() -> allocate();
    fire6_conv_expand1x1_out.allocator() -> allocate();
    fire6_expand1x1_act_out.allocator() -> allocate();

    fire6_weights_conv_expand3x3.allocator() -> allocate();
    fire6_bias_conv_expand3x3.allocator() -> allocate();
    fire6_conv_expand3x3_out.allocator() -> allocate();
    fire6_expand3x3_act_out.allocator() -> allocate();

    fire6_concat_out.allocator() -> allocate();

    fire7_weights_conv_squeeze.allocator() -> allocate();
    fire7_bias_conv_squeeze.allocator() -> allocate();
    fire7_conv_squeeze_out.allocator() -> allocate();
    fire7_squeeze_act_out.allocator() -> allocate();

    fire7_weights_conv_expand1x1.allocator() -> allocate();
    fire7_bias_conv_expand1x1.allocator() -> allocate();
    fire7_conv_expand1x1_out.allocator() -> allocate();
    fire7_expand1x1_act_out.allocator() -> allocate();

    fire7_weights_conv_expand3x3.allocator() -> allocate();
    fire7_bias_conv_expand3x3.allocator() -> allocate();
    fire7_conv_expand3x3_out.allocator() -> allocate();
    fire7_expand3x3_act_out.allocator() -> allocate();

    fire7_concat_out.allocator() -> allocate();

    fire8_weights_conv_squeeze.allocator() -> allocate();
    fire8_bias_conv_squeeze.allocator() -> allocate();
    fire8_conv_squeeze_out.allocator() -> allocate();
    fire8_squeeze_act_out.allocator() -> allocate();

    fire8_weights_conv_expand1x1.allocator() -> allocate();
    fire8_bias_conv_expand1x1.allocator() -> allocate();
    fire8_conv_expand1x1_out.allocator() -> allocate();
    fire8_expand1x1_act_out.allocator() -> allocate();

    fire8_weights_conv_expand3x3.allocator() -> allocate();
    fire8_bias_conv_expand3x3.allocator() -> allocate();
    fire8_conv_expand3x3_out.allocator() -> allocate();
    fire8_expand3x3_act_out.allocator() -> allocate();

    fire8_concat_out.allocator() -> allocate();

    fire9_weights_conv_squeeze.allocator() -> allocate();
    fire9_bias_conv_squeeze.allocator() -> allocate();
    fire9_conv_squeeze_out.allocator() -> allocate();
    fire9_squeeze_act_out.allocator() -> allocate();

    fire9_weights_conv_expand1x1.allocator() -> allocate();
    fire9_bias_conv_expand1x1.allocator() -> allocate();
    fire9_conv_expand1x1_out.allocator() -> allocate();
    fire9_expand1x1_act_out.allocator() -> allocate();

    fire9_weights_conv_expand3x3.allocator() -> allocate();
    fire9_bias_conv_expand3x3.allocator() -> allocate();
    fire9_conv_expand3x3_out.allocator() -> allocate();
    fire9_expand3x3_act_out.allocator() -> allocate();

    fire9_concat_out.allocator() -> allocate();

    conv10_weights.allocator() -> allocate();
    conv10_bias.allocator() -> allocate();
    conv10_out.allocator() -> allocate();
    conv10_act_out.allocator() -> allocate();

    global_avg_pool_out.allocator() -> allocate();

    flatten_out.allocator() -> allocate();

    softmax_out.allocator() -> allocate();

    /* ----------------------- [Initialize weights and biases tensors] --------------------------------- */

    //src image
    const std::string src_path = "images/cat.ppm";
    graph_utils::PPMAccessor src_ldr(src_path, true, 104., 117., 123.);
    src_ldr.access_tensor(src);

    // Load the weigths for all layers and biases from numpy-files

    const std::string wb_path = "weights/";

    // Conv1
    load_npy(conv1_weights, wb_path, "conv1_w");
    load_npy(conv1_bias, wb_path, "conv1_b");

    // Fire 2
    load_npy(fire2_weights_conv_squeeze, wb_path, "fire2_squeeze1x1_w");
    load_npy(fire2_bias_conv_squeeze, wb_path, "fire2_squeeze1x1_b");
    load_npy(fire2_weights_conv_expand1x1, wb_path, "fire2_expand1x1_w");
    load_npy(fire2_bias_conv_expand1x1, wb_path, "fire2_expand1x1_b");
    load_npy(fire2_weights_conv_expand3x3, wb_path, "fire2_expand3x3_w");
    load_npy(fire2_bias_conv_expand3x3, wb_path, "fire2_expand3x3_b");

    // Fire 3
    load_npy(fire3_weights_conv_squeeze, wb_path, "fire3_squeeze1x1_w");
    load_npy(fire3_bias_conv_squeeze, wb_path, "fire3_squeeze1x1_b");
    load_npy(fire3_weights_conv_expand1x1, wb_path, "fire3_expand1x1_w");
    load_npy(fire3_bias_conv_expand1x1, wb_path, "fire3_expand1x1_b");
    load_npy(fire3_weights_conv_expand3x3, wb_path, "fire3_expand3x3_w");
    load_npy(fire3_bias_conv_expand3x3, wb_path, "fire3_expand3x3_b");

    // Fire 4
    load_npy(fire4_weights_conv_squeeze, wb_path, "fire4_squeeze1x1_w");
    load_npy(fire4_bias_conv_squeeze, wb_path, "fire4_squeeze1x1_b");
    load_npy(fire4_weights_conv_expand1x1, wb_path, "fire4_expand1x1_w");
    load_npy(fire4_bias_conv_expand1x1, wb_path, "fire4_expand1x1_b");
    load_npy(fire4_weights_conv_expand3x3, wb_path, "fire4_expand3x3_w");
    load_npy(fire4_bias_conv_expand3x3, wb_path, "fire4_expand3x3_b");

    // Fire 5
    load_npy(fire5_weights_conv_squeeze, wb_path, "fire5_squeeze1x1_w");
    load_npy(fire5_bias_conv_squeeze, wb_path, "fire5_squeeze1x1_b");
    load_npy(fire5_weights_conv_expand1x1, wb_path, "fire5_expand1x1_w");
    load_npy(fire5_bias_conv_expand1x1, wb_path, "fire5_expand1x1_b");
    load_npy(fire5_weights_conv_expand3x3, wb_path, "fire5_expand3x3_w");
    load_npy(fire5_bias_conv_expand3x3, wb_path, "fire5_expand3x3_b");

    // Fire 6
    load_npy(fire6_weights_conv_squeeze, wb_path, "fire6_squeeze1x1_w");
    load_npy(fire6_bias_conv_squeeze, wb_path, "fire6_squeeze1x1_b");
    load_npy(fire6_weights_conv_expand1x1, wb_path, "fire6_expand1x1_w");
    load_npy(fire6_bias_conv_expand1x1, wb_path, "fire6_expand1x1_b");
    load_npy(fire6_weights_conv_expand3x3, wb_path, "fire6_expand3x3_w");
    load_npy(fire6_bias_conv_expand3x3, wb_path, "fire6_expand3x3_b");

    // Fire 7
    load_npy(fire7_weights_conv_squeeze, wb_path, "fire7_squeeze1x1_w");
    load_npy(fire7_bias_conv_squeeze, wb_path, "fire7_squeeze1x1_b");
    load_npy(fire7_weights_conv_expand1x1, wb_path, "fire7_expand1x1_w");
    load_npy(fire7_bias_conv_expand1x1, wb_path, "fire7_expand1x1_b");
    load_npy(fire7_weights_conv_expand3x3, wb_path, "fire7_expand3x3_w");
    load_npy(fire7_bias_conv_expand3x3, wb_path, "fire7_expand3x3_b");

    // Fire 8
    load_npy(fire8_weights_conv_squeeze, wb_path, "fire8_squeeze1x1_w");
    load_npy(fire8_bias_conv_squeeze, wb_path, "fire8_squeeze1x1_b");
    load_npy(fire8_weights_conv_expand1x1, wb_path, "fire8_expand1x1_w");
    load_npy(fire8_bias_conv_expand1x1, wb_path, "fire8_expand1x1_b");
    load_npy(fire8_weights_conv_expand3x3, wb_path, "fire8_expand3x3_w");
    load_npy(fire8_bias_conv_expand3x3, wb_path, "fire8_expand3x3_b");

    // Fire 9
    load_npy(fire9_weights_conv_squeeze, wb_path, "fire9_squeeze1x1_w");
    load_npy(fire9_bias_conv_squeeze, wb_path, "fire9_squeeze1x1_b");
    load_npy(fire9_weights_conv_expand1x1, wb_path, "fire9_expand1x1_w");
    load_npy(fire9_bias_conv_expand1x1, wb_path, "fire9_expand1x1_b");
    load_npy(fire9_weights_conv_expand3x3, wb_path, "fire9_expand3x3_w");
    load_npy(fire9_bias_conv_expand3x3, wb_path, "fire9_expand3x3_b");

    // Conv 10
    load_npy(conv10_weights, wb_path, "conv10_w");
    load_npy(conv10_bias, wb_path, "conv10_b");

    /* -------------------------------- [Execute the functions] -------------------------------- */

    std::cout << "conv1:" << std::endl;
    clock_fn([&](){conv1.run();});
    clock_fn([&](){conv1_act.run();});


    std::cout << "\nmaxpool1:" << std::endl;
    clock_fn([&](){max_pool1.run();});


    std::cout << "\nfire2:" << std::endl;
    clock_fn([&](){fire2_conv_squeeze.run();});
    clock_fn([&](){fire2_act_squeeze.run();});
    clock_fn([&](){fire2_conv_expand1x1.run();});
    clock_fn([&](){fire2_act_expand1x1.run();});
    clock_fn([&](){fire2_conv_expand3x3.run();});
    clock_fn([&](){fire2_act_expand3x3.run();});

    clock_fn([&](){fire2_concat.run();});

    std::cout << "\nfire3:" << std::endl;
    clock_fn([&](){fire3_conv_squeeze.run();});
    clock_fn([&](){fire3_act_squeeze.run();});
    clock_fn([&](){fire3_conv_expand1x1.run();});
    clock_fn([&](){fire3_act_expand1x1.run();});
    clock_fn([&](){fire3_conv_expand3x3.run();});
    clock_fn([&](){fire3_act_expand3x3.run();});

    clock_fn([&](){fire3_concat.run();});

    std::cout << "\nmaxpool2:" << std::endl;
    clock_fn([&](){max_pool2.run();});


    std::cout << "\nfire4:" << std::endl;
    clock_fn([&](){fire4_conv_squeeze.run();});
    clock_fn([&](){fire4_act_squeeze.run();});
    clock_fn([&](){fire4_conv_expand1x1.run();});
    clock_fn([&](){fire4_act_expand1x1.run();});
    clock_fn([&](){fire4_conv_expand3x3.run();});
    clock_fn([&](){fire4_act_expand3x3.run();});

    clock_fn([&](){fire4_concat.run();});

    std::cout << "\nfire5:" << std::endl;
    clock_fn([&](){fire5_conv_squeeze.run();});
    clock_fn([&](){fire5_act_squeeze.run();});
    clock_fn([&](){fire5_conv_expand1x1.run();});
    clock_fn([&](){fire5_act_expand1x1.run();});
    clock_fn([&](){fire5_conv_expand3x3.run();});
    clock_fn([&](){fire5_act_expand3x3.run();});

    clock_fn([&](){fire5_concat.run();});

    std::cout << "\nmaxpool3:" << std::endl;
    clock_fn([&](){max_pool3.run();});

    std::cout << "\nfire6:" << std::endl;
    clock_fn([&](){fire6_conv_squeeze.run();});
    clock_fn([&](){fire6_act_squeeze.run();});
    clock_fn([&](){fire6_conv_expand1x1.run();});
    clock_fn([&](){fire6_act_expand1x1.run();});
    clock_fn([&](){fire6_conv_expand3x3.run();});
    clock_fn([&](){fire6_act_expand3x3.run();});

    clock_fn([&](){fire6_concat.run();});


    std::cout << "\nfire7:" << std::endl;
    clock_fn([&](){fire7_conv_squeeze.run();});
    clock_fn([&](){fire7_act_squeeze.run();});
    clock_fn([&](){fire7_conv_expand1x1.run();});
    clock_fn([&](){fire7_act_expand1x1.run();});
    clock_fn([&](){fire7_conv_expand3x3.run();});
    clock_fn([&](){fire7_act_expand3x3.run();});

    clock_fn([&](){fire7_concat.run();});



    std::cout << "\nfire8:" << std::endl;
    clock_fn([&](){fire8_conv_squeeze.run();});
    clock_fn([&](){fire8_act_squeeze.run();});
    clock_fn([&](){fire8_conv_expand1x1.run();});
    clock_fn([&](){fire8_act_expand1x1.run();});
    clock_fn([&](){fire8_conv_expand3x3.run();});
    clock_fn([&](){fire8_act_expand3x3.run();});

    clock_fn([&](){fire8_concat.run();});

    std::cout << "\nfire9:" << std::endl;
    clock_fn([&](){fire9_conv_squeeze.run();});
    clock_fn([&](){fire9_act_squeeze.run();});
    clock_fn([&](){fire9_conv_expand1x1.run();});
    clock_fn([&](){fire9_act_expand1x1.run();});
    clock_fn([&](){fire9_conv_expand3x3.run();});
    clock_fn([&](){fire9_act_expand3x3.run();});

    clock_fn([&](){fire9_concat.run();});

    std::cout << "\nconv10:" << std::endl;
    clock_fn([&](){conv10.run();});
    clock_fn([&](){conv10_act.run();});

    std::cout << "\nglobal_avg_pool:" << std::endl;
    clock_fn([&](){global_avg_pool.run();});

    std::cout << "\nflatten:" << std::endl;
    clock_fn([&](){flatten.run();});

    std::cout << "\nsoftmax:" << std::endl;
    clock_fn([&](){softmax.run();});

    // Release memory?
}

int main(int argc, const char **argv)
{
    return utils::run_example(argc, argv, main_cnn);
}
