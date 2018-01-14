#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "utils/Utils.h"

#include <string.h>
#include <ctime>

using namespace arm_compute;
using namespace utils;

// 4x4 kernel
const uint8_t kernel4x4[] = {
    54, 114, 154, 56,
    65, 15,  2,   26,
    13, 244, 133, 55,
    98, 12,  98,  199
};

// ****** Unsigned 8 (fixpoint) conv ********

NEConvolutionLayer conv_qs8;
Tensor             w_qs8;
Tensor             b_qs8;

void acl_init_qs8(Tensor *src, Tensor *dst) {
    const TensorShape w_shape(static_cast<unsigned int>(4), static_cast<unsigned int>(4));
    const TensorShape b_shape(static_cast<unsigned int>(1));
    w_qs8.allocator()->init(TensorInfo(w_shape, 1, DataType::QS8, 5));
    b_qs8.allocator()->init(TensorInfo(b_shape, 1, DataType::QS8, 5));
    w_qs8.allocator()->allocate();
    b_qs8.allocator()->allocate();
    memcpy((void *)(w_qs8.buffer()), (void *)(kernel4x4), 16);

    conv_qs8.configure(src, &w_qs8, &b_qs8, dst, PadStrideInfo(1,1,0,0));
}

void acl_qs8() {
    conv_qs8.run();
}

// ****** Floating Point 32 conv ******* 

NEConvolutionLayer conv_f32;
Tensor             w_f32;
Tensor             b_f32;

void acl_init_f32(Tensor *src, Tensor *dst) {
    const TensorShape w_shape(static_cast<unsigned int>(4), static_cast<unsigned int>(4));
    const TensorShape b_shape(static_cast<unsigned int>(1));
    w_f32.allocator()->init(TensorInfo(w_shape, 1, DataType::F32));
    b_f32.allocator()->init(TensorInfo(b_shape, 1, DataType::F32));
    w_f32.allocator()->allocate();
    b_f32.allocator()->allocate();

    float *ptr = reinterpret_cast<float *>(w_f32.buffer());
    for (size_t i = 0; i < 16; i++) {
        *(ptr++) = static_cast<float>(kernel4x4[i]);
    }

    *(reinterpret_cast<float *>(b_f32.buffer())) = 4.;

    conv_f32.configure(src, &w_f32, &b_f32, dst, PadStrideInfo(1,1,0,0));
}

void acl_f32() {
    conv_f32.run();
}

// *************************************

int main(int argc, const char **argv) {
    ARM_COMPUTE_UNUSED(argc);
    ARM_COMPUTE_UNUSED(argv);

    // Setup source images
    Tensor src_qs8;
    Tensor dst_qs8;
    Tensor src_f32;
    Tensor dst_f32;

    const TensorShape src_shape(static_cast<unsigned int>(640), static_cast<unsigned int>(480));
    const TensorShape dst_shape(static_cast<unsigned int>(637), static_cast<unsigned int>(477));

    src_qs8.allocator()->init(TensorInfo(src_shape, 1, DataType::QS8, 5));
    dst_qs8.allocator()->init(TensorInfo(dst_shape, 1, DataType::QS8, 5));
    src_f32.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));
    dst_f32.allocator()->init(TensorInfo(dst_shape, 1, DataType::F32));

    src_qs8.info()->auto_padding();
    src_f32.info()->auto_padding();

    src_qs8.allocator()->allocate();
    dst_qs8.allocator()->allocate();
    src_f32.allocator()->allocate();
    dst_f32.allocator()->allocate();

    acl_init_qs8(&src_qs8, &dst_qs8);
    acl_init_f32(&src_f32, &dst_f32);

    /// Fill input vector with data
    {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(src_qs8.buffer());
        for (size_t i = 0; i < src_qs8.info()->total_size(); i++) {
            *(ptr++) = static_cast<uint8_t>(i % 256);
        }
    }
    {
        float *ptr = reinterpret_cast<float *>(src_f32.buffer());
        for (size_t i = 0; i < src_f32.info()->total_size() / sizeof(float); i++) {
            *(ptr++) = static_cast<float>(i % 256);
        }
    }

    // Save F32 SRC to numpy file
    const std::string npy1 = "f32_src.npy";
    save_to_npy(src_f32, npy1, false);

    // Time qs8
    clock_t begin = clock();
    for (int i = 0; i < 10; i++)
        acl_qs8();
    clock_t end = clock();
    double t = double(end - begin) / CLOCKS_PER_SEC;
    printf("QS8: Elapsed time %.2f S, avg: %.2f mS\n", t, t * 100);

    // Time f32
    begin = clock();
    for (int i = 0; i < 10; i++)
        acl_f32();
    end = clock();
    t = double(end - begin) / CLOCKS_PER_SEC;
    printf("F32: Elapsed time %.2f S, avg: %.2f mS\n", t, t * 100);

    // Save F32 DST to numpy file
    const std::string npy2 = "f32_dst.npy";
    save_to_npy(dst_f32, npy2, false);

    return 1;
}
