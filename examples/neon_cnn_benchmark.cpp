/*
 * Notice: Code has been modified from neon_cnn.cpp
 *
 * Copyright (c) 2016, 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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
#include <limits.h>
#include <stdlib.h>
#include <assert.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

using namespace arm_compute;
using namespace utils;


static double t_msec(int clock_type) {
    struct timespec t;

    clock_gettime(clock_type, &t);
    return 1e3 * t.tv_sec + t.tv_nsec / 1e6;
}


void main_cnn(int argc, const char **argv)
{
    ARM_COMPUTE_UNUSED(argc);
    ARM_COMPUTE_UNUSED(argv);

    // The src tensor should contain the input image
    Tensor src;

    // The weights and biases tensors should be initialized with the values inferred with the training
    Tensor weights0;
    Tensor weights1;
    Tensor weights2;
    Tensor biases0;
    Tensor biases1;
    Tensor biases2;

    Tensor out_conv0;
    Tensor out_conv1;
    Tensor out_act0;
    Tensor out_act1;
    Tensor out_act2;
    Tensor out_pool0;
    Tensor out_pool1;
    Tensor out_fc0;
    Tensor out_softmax;

    // Create layers and set memory manager where allowed to manage internal memory requirements
    NEConvolutionLayer    conv0;
    NEConvolutionLayer    conv1;
    NEPoolingLayer        pool0;
    NEPoolingLayer        pool1;
    NEFullyConnectedLayer fc0;
    NEActivationLayer     act0;
    NEActivationLayer     act1;
    NEActivationLayer     act2;
    NESoftmaxLayer        softmax;

    /* [Initialize tensors] */

    // Initialize src tensor
    constexpr unsigned int width_src_image  = 128;
    constexpr unsigned int height_src_image = 128;
    constexpr unsigned int ifm_src_img      = 3;

    const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
    src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

    // Initialize tensors of conv0
    constexpr unsigned int kernel_x_conv0 = 5;
    constexpr unsigned int kernel_y_conv0 = 5;
    constexpr unsigned int ofm_conv0      = 8;

    const TensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);
    const TensorShape biases_shape_conv0(weights_shape_conv0[3]);
    const TensorShape out_shape_conv0(src_shape.x(), src_shape.y(), weights_shape_conv0[3]);

    weights0.allocator()->init(TensorInfo(weights_shape_conv0, 1, DataType::F32));
    biases0.allocator()->init(TensorInfo(biases_shape_conv0, 1, DataType::F32));
    out_conv0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

    // Initialize tensor of act0
    out_act0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

    // Initialize tensor of pool0
    TensorShape out_shape_pool0 = out_shape_conv0;
    out_shape_pool0.set(0, out_shape_pool0.x() / 2);
    out_shape_pool0.set(1, out_shape_pool0.y() / 2);
    out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType::F32));

    // Initialize tensors of conv1
    constexpr unsigned int kernel_x_conv1 = 3;
    constexpr unsigned int kernel_y_conv1 = 3;
    constexpr unsigned int ofm_conv1      = 16;

    const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, out_shape_pool0.z(), ofm_conv1);

    const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
    const TensorShape out_shape_conv1(out_shape_pool0.x(), out_shape_pool0.y(), weights_shape_conv1[3]);

    weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
    biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
    out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

    // Initialize tensor of act1
    out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

    // Initialize tensor of pool1
    TensorShape out_shape_pool1 = out_shape_conv1;
    out_shape_pool1.set(0, out_shape_pool1.x() / 2);
    out_shape_pool1.set(1, out_shape_pool1.y() / 2);
    out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));

    // Initialize tensor of fc0
    constexpr unsigned int num_labels = 128;

    const TensorShape weights_shape_fc0(out_shape_pool1.x() * out_shape_pool1.y() * out_shape_pool1.z(), num_labels);
    const TensorShape biases_shape_fc0(num_labels);
    const TensorShape out_shape_fc0(num_labels);

    weights2.allocator()->init(TensorInfo(weights_shape_fc0, 1, DataType::F32));
    biases2.allocator()->init(TensorInfo(biases_shape_fc0, 1, DataType::F32));
    out_fc0.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));

    // Initialize tensor of act2
    out_act2.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));

    // Initialize tensor of softmax
    const TensorShape out_shape_softmax(out_shape_fc0.x());
    out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));

    /* -----------------------End: [Initialize tensors] */

    /* [Configure functions] */

    // in:32x32x1: 5x5 convolution, 8 output features maps (OFM)
    conv0.configure(&src, &weights0, &biases0, &out_conv0, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 2 /* pad_x */, 2 /* pad_y */));

    // in:32x32x8, out:32x32x8, Activation function: relu
    act0.configure(&out_conv0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    // in:32x32x8, out:16x16x8 (2x2 pooling), Pool type function: Max
    pool0.configure(&out_act0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

    // in:16x16x8: 3x3 convolution, 16 output features maps (OFM)
    conv1.configure(&out_pool0, &weights1, &biases1, &out_conv1, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1 /* pad_x */, 1 /* pad_y */));

    // in:16x16x16, out:16x16x16, Activation function: relu
    act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    // in:16x16x16, out:8x8x16 (2x2 pooling), Pool type function: Average
    pool1.configure(&out_act1, &out_pool1, PoolingLayerInfo(PoolingType::AVG, 2, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

    // in:8x8x16, out:128
    fc0.configure(&out_pool1, &weights2, &biases2, &out_fc0);

    // in:128, out:128, Activation function: relu
    act2.configure(&out_fc0, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    // in:128, out:128
    softmax.configure(&out_act2, &out_softmax);

    /* -----------------------[Allocate tensors]------------------ */
    // Now that the padding requirements are known we can allocate all tensors

    out_conv0.allocator()->allocate();
    out_act0.allocator()->allocate();
    out_pool0.allocator()->allocate();
    out_conv1.allocator()->allocate();
    out_act1.allocator()->allocate();
    out_pool1.allocator()->allocate();
    out_fc0.allocator()->allocate();
    out_act2.allocator()->allocate();
    out_softmax.allocator()->allocate();

    src.allocator()->allocate();
    weights0.allocator()->allocate();
    weights1.allocator()->allocate();
    weights2.allocator()->allocate();
    biases0.allocator()->allocate();
    biases1.allocator()->allocate();
    biases2.allocator()->allocate();

    /* -----------------------[Initialize weights and biases tensors] */

    const std::string src_path = "/tmp/images/128x128.ppm";
    graph_utils::PPMAccessor src_ldr(src_path, true, 104., 117., 123.);
    src_ldr.access_tensor(src);

    // Load the weigths for convolutionary and fully connected layers
    graph_utils::NumPyBinLoader w0_ldr("/tmp/weights/conv0_w.npy");
    w0_ldr.access_tensor(weights0);

    graph_utils::NumPyBinLoader w1_ldr("/tmp/weights/conv1_w.npy");
    w1_ldr.access_tensor(weights1);

    graph_utils::NumPyBinLoader w2_ldr("/tmp/weights/fc2_w.npy");
    w2_ldr.access_tensor(weights2);

    // Load the biases for conv layers
    graph_utils::NumPyBinLoader b0_ldr("/tmp/weights/conv0_b.npy");
    b0_ldr.access_tensor(biases0);

    graph_utils::NumPyBinLoader b1_ldr("/tmp/weights/conv1_b.npy");
    b1_ldr.access_tensor(biases1);

    graph_utils::NumPyBinLoader b2_ldr("/tmp/weights/fc2_b.npy");
    b2_ldr.access_tensor(biases2);

    /* [Execute the functions] */

    std::cout << "Starting timed test..." << std::endl;

    double t = t_msec(CLOCK_MONOTONIC);

    int i;
    for (i = 0; i < 100; i++) {
        conv0.run();
        act0.run();
        pool0.run();
        conv1.run();
        act1.run();
        pool1.run();
        fc0.run();
        act2.run();
        softmax.run();
    }

    t = t_msec(CLOCK_MONOTONIC) - t;

    std::cout << "Timed test at: " << t << " mS for " \
        << i << " iterations" << std::endl \
        << "Avg: " << t / i << " mS per iteration." << std::endl << std::endl;

    std::cout << "First layer's ten first activations:" << std::endl;
    float *dptr = reinterpret_cast<float *>(out_conv0.buffer() + out_conv0.info()->offset_element_in_bytes(Coordinates(0, 0, 0, 0)));
    for (int i = 0; i < 10; i ++, dptr ++) {
        std::cout << " " << *dptr << " ";
    }
    std::cout << std::endl;

    /* -----------------------End: [Execute the functions] */
}

/** Main program for cnn test
 *
 * The example implements the following CNN architecture:
 *
 * Input -> conv0:5x5 -> act0:relu -> pool:2x2 -> conv1:3x3 -> act1:relu -> pool:2x2 -> fc0 -> act2:relu -> softmax
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, const char **argv)
{
    return utils::run_example(argc, argv, main_cnn);
}

#pragma GCC diagnostic pop
