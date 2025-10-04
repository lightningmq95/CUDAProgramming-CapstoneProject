#include <iostream>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cufft.h>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// For brevity in filesystem paths
namespace fs = std::filesystem;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Custom atomicMax for floats using atomicCAS
__device__ static float atomicMax_float(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// All GPU kernels remain the same as before
__global__ void sepia_kernel(const unsigned char* src, unsigned char* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float r_in = src[idx + 0], g_in = src[idx + 1], b_in = src[idx + 2];
        unsigned char r_out = (unsigned char)fminf(255.0f, r_in * 0.393f + g_in * 0.769f + b_in * 0.189f);
        unsigned char g_out = (unsigned char)fminf(255.0f, r_in * 0.349f + g_in * 0.686f + b_in * 0.168f);
        unsigned char b_out = (unsigned char)fminf(255.0f, r_in * 0.272f + g_in * 0.534f + b_in * 0.131f);
        dst[idx + 0] = r_out;
        dst[idx + 1] = g_out;
        dst[idx + 2] = b_out;
    }
}
__global__ void rgb_to_grayscale_complex_kernel(const unsigned char* rgb, cufftComplex* gray_complex, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int rgb_idx = (y * width + x) * 3;
        int gray_idx = y * width + x;
        float gray_val = 0.299f * rgb[rgb_idx] + 0.587f * rgb[rgb_idx + 1] + 0.114f * rgb[rgb_idx + 2];
        gray_complex[gray_idx].x = gray_val;
        gray_complex[gray_idx].y = 0.0f;
    }
}
__global__ void calculate_log_magnitude_kernel(const cufftComplex* fft_result, float* log_magnitude, float* d_max_val, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        cufftComplex c = fft_result[idx];
        float mag = sqrtf(c.x * c.x + c.y * c.y);
        float log_mag = logf(1.0f + mag);
        log_magnitude[idx] = log_mag;
        atomicMax_float(d_max_val, log_mag);
    }
}
__global__ void shift_and_normalize_kernel(const float* log_magnitude, unsigned char* out_img, const float* d_max_val, int width, int height) {
    float max_val = *d_max_val;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int sx = (x + width / 2) % width;
        int sy = (y + height / 2) % height;
        int src_idx = sy * width + sx;
        int dst_idx = y * width + x;
        float normalized_val = 0.0f;
        if (max_val > 0.0f) {
            normalized_val = (log_magnitude[src_idx] / max_val) * 255.0f;
        }
        out_img[dst_idx] = (unsigned char)fminf(255.0f, normalized_val);
    }
}

int main() {
    // === 1. Setup Input/Output Directories ===
    const std::string input_dir = "input_images";
    const std::string output_dir = "output_images";

    if (!fs::exists(input_dir)) {
        std::cerr << "Error: Input directory '" << input_dir << "' not found!" << std::endl;
        return 1;
    }
    fs::create_directory(output_dir); // Create output directory if it doesn't exist

    // === 2. Create CUDA Stream and Events (reused for all images) ===
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Starting batch image processing..." << std::endl;
    int processed_count = 0;

    // === 3. Loop Through Each File in the Input Directory ===
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        const auto& input_path = entry.path();
        const std::string extension = input_path.extension().string();

        // Simple check for common image extensions
        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png" && extension != ".bmp") {
            continue;
        }

        std::cout << "\nProcessing: " << input_path.filename().string() << std::endl;

        // --- A. Load Image ---
        int width, height, channels;
        unsigned char* h_input_img = stbi_load(input_path.string().c_str(), &width, &height, &channels, 3); 
        if (!h_input_img) {
            std::cerr << "Warning: Could not load image " << input_path.filename().string() << ". Skipping." << std::endl;
            continue;
        }
        
        // --- B. Allocate Memory for Current Image ---
        // Memory is allocated and freed inside the loop to handle varying image sizes
        size_t img_size_rgb = width * height * 3 * sizeof(unsigned char);
        size_t img_size_gray = width * height * sizeof(unsigned char);

        unsigned char *h_blurred_img, *h_sepia_img, *h_fft_img;
        unsigned char *d_input_img, *d_blurred_img, *d_sepia_img, *d_fft_img;
        
        CUDA_CHECK(cudaMallocHost(&h_blurred_img, img_size_rgb));
        CUDA_CHECK(cudaMallocHost(&h_sepia_img, img_size_rgb));
        CUDA_CHECK(cudaMallocHost(&h_fft_img, img_size_gray));

        CUDA_CHECK(cudaMalloc(&d_input_img, img_size_rgb));
        CUDA_CHECK(cudaMalloc(&d_blurred_img, img_size_rgb));
        CUDA_CHECK(cudaMalloc(&d_sepia_img, img_size_rgb));
        CUDA_CHECK(cudaMalloc(&d_fft_img, img_size_gray));

        // --- C. Run GPU Processing Pipeline ---
        float milliseconds = 0;
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUDA_CHECK(cudaMemcpyAsync(d_input_img, h_input_img, img_size_rgb, cudaMemcpyHostToDevice, stream));

        NppiSize oSizeROI = { width, height };
        nppiFilterGauss_8u_C3R(d_input_img, width * 3, d_blurred_img, width * 3, oSizeROI, NPP_MASK_SIZE_5_X_5);
        CUDA_CHECK(cudaMemcpyAsync(h_blurred_img, d_blurred_img, img_size_rgb, cudaMemcpyDeviceToHost, stream));

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        sepia_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_blurred_img, d_sepia_img, width, height);
        CUDA_CHECK(cudaMemcpyAsync(h_sepia_img, d_sepia_img, img_size_rgb, cudaMemcpyDeviceToHost, stream));

        cufftComplex* d_complex_img;
        float* d_log_magnitude;
        float* d_max_val;
        
        CUDA_CHECK(cudaMalloc(&d_complex_img, width * height * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_log_magnitude, width * height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_max_val, sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_max_val, 0, sizeof(float), stream));

        rgb_to_grayscale_complex_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_sepia_img, d_complex_img, width, height);

        cufftHandle fft_plan;
        cufftPlan2d(&fft_plan, height, width, CUFFT_C2C);
        cufftExecC2C(fft_plan, d_complex_img, d_complex_img, CUFFT_FORWARD);

        calculate_log_magnitude_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_complex_img, d_log_magnitude, d_max_val, width, height);
        shift_and_normalize_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_log_magnitude, d_fft_img, d_max_val, width, height);

        CUDA_CHECK(cudaMemcpyAsync(h_fft_img, d_fft_img, img_size_gray, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); 
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "GPU pipeline time: " << milliseconds << " ms" << std::endl;

        // --- D. Save Output Images ---
        std::string stem = input_path.stem().string();
        std::string blurred_path = (fs::path(output_dir) / (stem + "_blurred.png")).string();
        std::string sepia_path = (fs::path(output_dir) / (stem + "_sepia.png")).string();
        std::string fft_path = (fs::path(output_dir) / (stem + "_fft.png")).string();

        stbi_write_png(blurred_path.c_str(), width, height, 3, h_blurred_img, width * 3);
        stbi_write_png(sepia_path.c_str(), width, height, 3, h_sepia_img, width * 3);
        stbi_write_png(fft_path.c_str(), width, height, 1, h_fft_img, width);
        std::cout << "Outputs saved to '" << output_dir << "' folder." << std::endl;

        // --- E. Cleanup Memory for Current Image ---
        stbi_image_free(h_input_img);
        CUDA_CHECK(cudaFreeHost(h_blurred_img));
        CUDA_CHECK(cudaFreeHost(h_sepia_img));
        CUDA_CHECK(cudaFreeHost(h_fft_img));
        CUDA_CHECK(cudaFree(d_input_img));
        CUDA_CHECK(cudaFree(d_blurred_img));
        CUDA_CHECK(cudaFree(d_sepia_img));
        CUDA_CHECK(cudaFree(d_complex_img));
        CUDA_CHECK(cudaFree(d_log_magnitude));
        CUDA_CHECK(cudaFree(d_max_val));
        CUDA_CHECK(cudaFree(d_fft_img));
        cufftDestroy(fft_plan);

        processed_count++;
    }

    // === 4. Final Cleanup ===
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "\nBatch processing complete. " << processed_count << " images processed." << std::endl;
    return 0;
}