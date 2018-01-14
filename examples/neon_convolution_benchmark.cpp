#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "utils/Utils.h"

#include <string.h>
#include <ctime>

using namespace arm_compute;
using namespace utils;

// 4x4 kernel
const int16_t kernel3x3[] = {
    0, 0, 1,
    0, 1, 0,
    0, 0, 1
};
const uint8_t bias = 98;

// ****** Unsigned 8 conv (kernel-pass only) ********

NEConvolution3x3 conv_u8;

void acl_init_u8(Tensor *src, Tensor *dst) {
    conv_u8.configure(src, dst, kernel3x3, 1, BorderMode::UNDEFINED);
}

void acl_u8(Tensor *dst) {
    conv_u8.run();
    uint8_t *ptr = dst->buffer();
    for (size_t i = 0; i < dst->info()->total_size(); i++) {
        ptr[i] = bias > 0xFF - ptr[i] ? 0xFF : ptr[i] + bias;
    }
}
// ****** Unsigned 8 (fixpoint) conv ********

NEConvolutionLayer conv_qs8;
Tensor             w_qs8;
Tensor             b_qs8;

void acl_init_qs8(Tensor *src, Tensor *dst) {
    const TensorShape w_shape(static_cast<unsigned int>(3), static_cast<unsigned int>(3));
    const TensorShape b_shape(static_cast<unsigned int>(1));
    w_qs8.allocator()->init(TensorInfo(w_shape, 1, DataType::QS8, 5));
    b_qs8.allocator()->init(TensorInfo(b_shape, 1, DataType::QS8, 5));
    w_qs8.allocator()->allocate();
    b_qs8.allocator()->allocate();

    uint8_t *ptr = w_qs8.buffer();
    for (size_t i = 0; i < 9; i++) {
        *(ptr++) = static_cast<uint8_t>(kernel3x3[i]);
    }

    *(b_qs8.buffer()) = bias;

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
    const TensorShape w_shape(static_cast<unsigned int>(3), static_cast<unsigned int>(3));
    const TensorShape b_shape(static_cast<unsigned int>(1));
    w_f32.allocator()->init(TensorInfo(w_shape, 1, DataType::F32));
    b_f32.allocator()->init(TensorInfo(b_shape, 1, DataType::F32));
    w_f32.allocator()->allocate();
    b_f32.allocator()->allocate();

    float *ptr = reinterpret_cast<float *>(w_f32.buffer());
    for (size_t i = 0; i < 9; i++) {
        *(ptr++) = static_cast<float>(kernel3x3[i]);
    }

    *(reinterpret_cast<float *>(b_f32.buffer())) = static_cast<float>(bias);

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
    Tensor src_u8;
    Tensor dst_u8;
    Tensor src_qs8;
    Tensor dst_qs8;
    Tensor src_f32;
    Tensor dst_f32;

    const TensorShape src_shape(static_cast<unsigned int>(640), static_cast<unsigned int>(480));
    const TensorShape dst_shape(static_cast<unsigned int>(638), static_cast<unsigned int>(478));

    src_u8.allocator()->init(TensorInfo(src_shape.x(), src_shape.y(), Format::U8));
    dst_u8.allocator()->init(TensorInfo(src_shape.x(), src_shape.y(), Format::U8));
    src_qs8.allocator()->init(TensorInfo(src_shape, 1, DataType::QS8, 0));
    dst_qs8.allocator()->init(TensorInfo(dst_shape, 1, DataType::QS8, 0));
    src_f32.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));
    dst_f32.allocator()->init(TensorInfo(dst_shape, 1, DataType::F32));

    src_u8.info()->auto_padding();
    src_qs8.info()->auto_padding();
    src_f32.info()->auto_padding();

    acl_init_u8(&src_u8, &dst_u8);
    acl_init_qs8(&src_qs8, &dst_qs8);
    acl_init_f32(&src_f32, &dst_f32);

    src_u8.allocator()->allocate();
    dst_u8.allocator()->allocate();
    src_qs8.allocator()->allocate();
    dst_qs8.allocator()->allocate();
    src_f32.allocator()->allocate();
    dst_f32.allocator()->allocate();

    /// Fill input vector with data
    {
        uint8_t *ptr = src_u8.buffer();
        int x, y;
        for (x = y = 0; true; x++) {
            if (static_cast<unsigned int>(x) == src_shape.x()) {
                y++;
                x = 0;
                if (src_shape.y() == static_cast<unsigned int>(y)) {
                    break;
                }
            }
            ptr[src_u8.info()->offset_element_in_bytes(Coordinates(x, y))] = static_cast<uint8_t>((y * src_shape.x() + x) % 256);
        }
    }
    {
        uint8_t *ptr = src_qs8.buffer();
        int x, y;
        for (x = y = 0; true; x++) {
            if (static_cast<unsigned int>(x) == src_shape.x()) {
                y++;
                x = 0;
                if (src_shape.y() == static_cast<unsigned int>(y)) {
                    break;
                }
            }
            ptr[src_qs8.info()->offset_element_in_bytes(Coordinates(x, y))] = static_cast<uint8_t>((y * src_shape.x() + x) % 256);
        }
    }
    {
        uint8_t *ptr = src_f32.buffer();
        int x, y;
        for (x = y = 0; true; x++) {
            if (static_cast<unsigned int>(x) == src_shape.x()) {
                y++;
                x = 0;
                if (src_shape.y() == static_cast<unsigned int>(y)) {
                    break;
                }
            }
            *(reinterpret_cast<float *>(ptr + src_f32.info()->offset_element_in_bytes(Coordinates(x, y)))) = static_cast<float>((y * src_shape.x() + x) % 256);
        }
    }

    // Save src to numpy file
    const std::string npy2 = "f32_src.npy";
    save_to_npy(src_f32, npy2, false);

    clock_t begin, end;
    double  t;

    // Time u8
    begin = clock();
    for (int i = 0; i < 10; i++)
        acl_u8(&dst_u8);
    end = clock();
    t = double(end - begin) / CLOCKS_PER_SEC;
    printf("U8: Elapsed time %.2f S, avg: %.2f mS\n", t, t * 100);

    // Time qs8
    begin = clock();
    for (int i = 0; i < 10; i++)
        acl_qs8();
    end = clock();
    t = double(end - begin) / CLOCKS_PER_SEC;
    printf("QS8: Elapsed time %.2f S, avg: %.2f mS\n", t, t * 100);

    // Time f32
    begin = clock();
    for (int i = 0; i < 10; i++)
        acl_f32();
    end = clock();
    t = double(end - begin) / CLOCKS_PER_SEC;
    printf("F32: Elapsed time %.2f S, avg: %.2f mS\n", t, t * 100);

    {
        uint8_t *buff = dst_u8.buffer();
        printf("\nU8  out:");
        for (int i = 0; i < 10; i ++) {
            printf("\t%d", buff[dst_u8.info()->offset_element_in_bytes(Coordinates(i + 1, 1))]); // Skip border
        }
    }
    {

        uint8_t *buff = dst_f32.buffer();
        printf("\nF32 out:");
        for (int i = 0; i < 10; i ++) {
            printf("\t%.2f", *(reinterpret_cast<float *>(buff + dst_f32.info()->offset_element_in_bytes(Coordinates(i, 0)))));
        }
    }
    {
        uint8_t *buff = dst_qs8.buffer();
        printf("\nQS8 out:");
        for (int i = 0; i < 10; i ++) {
            printf("\t%d", buff[dst_qs8.info()->offset_element_in_bytes(Coordinates(i, 0))]);
        }
    }

    printf("\n");

    // Save dst to numpy file
    const std::string npy4 = "f32_dst.npy";
    save_to_npy(dst_f32, npy4, false);

    return 1;
}
