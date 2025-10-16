/* CNN MNIST Inference Application for Kria KV260 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>

#include "CGRA.h"
#include "FPGA_Driver.c" // cung cấp Xil_Out32 / Xil_In32
#include "weights_array.h"
#include "test_images.c" // image_array, test_labels

// ===== Fixed-point Q16.16 (host side)
typedef int32_t fxp;
#define FRAC_BITS 16
static inline fxp float_to_fxp(float x) { return (fxp)llroundf(x * (1 << FRAC_BITS)); }
static inline float fxp_to_float(fxp x) { return (float)x / (1 << FRAC_BITS); }

// ===== Base addresses (ĐỊA CHỈ BYTE). KHÔNG shift >>2
#include "xparameters.h"

#define CNN_CTRL_BASEADDR XPAR_CNN_TOP_0_S_AXI_CONTROL_BASEADDR     // 0xA0000000
#define INMODEL_BRAM_BASEADDR XPAR_INMODEL_BRAM_CTRL_S_AXI_BASEADDR // 0xA0020000
#define WEIGHTS_BRAM_BASEADDR XPAR_WEIGHTS_BRAM_CTRL_S_AXI_BASEADDR // 0xA0028000

#define CNN_CTRL_AP_CTRL 0x00
#define CNN_CTRL_OUTMODEL0 0x10
#define INPUT_SIZE 784
#define WEIGHTS_SIZE 5738
#define NUM_TEST_IMAGES 10

// ---- Ghi mảng 32-bit vào BRAM: NHÂN 4 (địa chỉ byte)
static void load_weights_to_bram(const fxp *weights)
{
    for (int i = 0; i < WEIGHTS_SIZE; i++)
    {
        Xil_Out32(WEIGHTS_BRAM_BASEADDR + i, (uint32_t)weights[i]);
    }
}
static void load_image_to_bram(const fxp *image)
{
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        Xil_Out32(INMODEL_BRAM_BASEADDR + i, (uint32_t)image[i]);
    }
}

// ---- Điều khiển IP HLS qua ap_ctrl
static void start_cnn(void)
{
    // ap_start = 1 (bit0). Thông thường không cần clear lại 0 ngay.
    Xil_Out32(CNN_CTRL_BASEADDR + CNN_CTRL_AP_CTRL, 0x1u);
}
static void wait_cnn_done(void)
{
    // ap_done = bit1 (RO; clear-on-read hoặc khi ap_start hạ)
    while ((Xil_In32(CNN_CTRL_BASEADDR + CNN_CTRL_AP_CTRL) & 0x2u) == 0u)
    {
        // spin
    }
}

static float read_cnn_output(void)
{
    uint32_t raw = Xil_In32(CNN_CTRL_BASEADDR + CNN_CTRL_OUTMODEL0);
    fxp y = (fxp)raw; // giữ nguyên bit pattern (signed)
    return fxp_to_float(y);
}

// ---- Chạy 1 ảnh
static int run_inference(const fxp *image_fixed)
{
    load_image_to_bram(image_fixed);
    start_cnn();
    wait_cnn_done();
    float output = read_cnn_output(); // tùy IP: nếu ra 1 giá trị lớp thì ok
    int predicted_class = (int)(output + 0.5f);
    return predicted_class;
}

int main(void)
{
    // Giả định weights_array và image_array chứa sẵn Q16.16 (int32_t)
    const fxp *Weights = (const fxp *)weights_array;

    unsigned char *membase;

    if (cgra_open() == 0)
    {
        fprintf(stderr, "cgra_open() failed\n");
        return 1;
    }

    cgra.dma_ctrl = CGRA_info.dma_mmap;
    membase = (unsigned char *)CGRA_info.ddr_mmap;

    printf("membase: 0x%016" PRIxPTR "\n", (uintptr_t)membase);

    printf("\n========================================\n");
    printf("  CNN MNIST Inference on KV260\n");
    printf("========================================\n\n");

    // ---- Load weights 1 lần
    printf("Loading weights into BRAM...\n");
    load_weights_to_bram(Weights);
    printf("Load weights done!\n");

    // ---- Chạy N ảnh test
    printf("Running inference on %d test images...\n\n", NUM_TEST_IMAGES);

    int correct = 0;

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        printf("Image %d: ", i);

        int predicted = run_inference(test_images_fixed[i]);
        int expected = test_labels[i];

        if (predicted == expected)
        {
            printf("PASS - Predicted: %d, Expected: %d ✓\n", predicted, expected);
            correct++;
        }
        else
        {
            printf("FAIL - Predicted: %d, Expected: %d ✗\n", predicted, expected);
        }
    }

    // Print summary
    printf("\n========================================\n");
    printf("  RESULTS\n");
    printf("========================================\n");
    printf("Total images: %d\n", NUM_TEST_IMAGES);
    printf("Correct:      %d\n", correct);
    printf("Accuracy:     %.1f%%\n", (float)correct / NUM_TEST_IMAGES * 100.0f);
    printf("========================================\n\n");

    if (correct == NUM_TEST_IMAGES)
    {
        printf("✓✓✓ ALL TESTS PASSED! ✓✓✓\n");
    }
    else
    {
        printf("Some tests failed.\n");
    }

    return 0;
}
