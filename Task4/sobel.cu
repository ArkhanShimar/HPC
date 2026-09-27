#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
extern "C" {
#include "lodepng.h"
}

__device__ int grey(const unsigned char *image, long long x, long long y, unsigned width, unsigned height) {
    if (x < 0 || y < 0 || x >= width || y >= height) return 0; // Zero padding.
    size_t p = ((size_t)y * width + (size_t)x)*4;
    return (77*(int)image[p] + 150*(int)image[p+1] + 29*(int)image[p+2]) >> 8;
}
__global__ void sobel_edges(const unsigned char *input, unsigned char *output, unsigned width, unsigned height) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)width*height) return;
    long long x = i % width, y = i / width;
    const int gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    const int gy[9] = {-1,-2,-1,0,0,0,1,2,1};
    int sx = 0, sy = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int value = grey(input, x+dx, y+dy, width, height);
            int k = (dy+1)*3+dx+1; sx += value*gx[k]; sy += value*gy[k];
        }
    }
    int magnitude = (int)(sqrtf((float)(sx*sx+sy*sy)) + 0.5f);
    unsigned char edge = (unsigned char)(magnitude > 255 ? 255 : magnitude);
    output[i*4] = output[i*4+1] = output[i*4+2] = edge;
    output[i*4+3] = 255; // Opaque edge map, including originally transparent pixels.
}
static int checked(cudaError_t code, const char *action) {
    if (code == cudaSuccess) return 1;
    fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(code)); return 0;
}
static int process_image(const char *filename) {
    unsigned char *input = NULL, *output = NULL, *device_in = NULL, *device_out = NULL;
    unsigned width = 0, height = 0, error;
    size_t pixels = 0, bytes = 0, blocks = 0;
    cudaEvent_t start = NULL, stop = NULL; cudaDeviceProp properties;
    int status = 1; float ms = 0;
    const char *base = strrchr(filename, '/'); base = base ? base+1 : filename;
    size_t name_length = strlen(base); char *out_name = NULL;
    if (name_length < 5 || strcmp(base+name_length-4, ".png")) { fprintf(stderr, "Expected a .png filename: %s\n", filename); return 1; }
    out_name = (char *)malloc(name_length+7);
    if (!out_name) return 1;
    memcpy(out_name, base, name_length-4); strcpy(out_name+name_length-4, "_edges.png");
    error = lodepng_decode32_file(&input, &width, &height, filename);
    if (error) { fprintf(stderr, "%s: %s\n", filename, lodepng_error_text(error)); goto cleanup; }
    if (!width || !height || (size_t)width > SIZE_MAX/(size_t)height/4) { fprintf(stderr, "Invalid image dimensions.\n"); goto cleanup; }
    pixels = (size_t)width*height; bytes = pixels*4; blocks = (pixels+255)/256;
    output = (unsigned char *)malloc(bytes);
    if (!output) goto cleanup;
    if (!checked(cudaGetDeviceProperties(&properties, 0), "GPU properties")) goto cleanup;
    if (blocks > (size_t)properties.maxGridSize[0]) { fprintf(stderr, "Image exceeds grid limit.\n"); goto cleanup; }
    if (!checked(cudaMalloc((void **)&device_in, bytes), "Allocate input image") ||
        !checked(cudaMalloc((void **)&device_out, bytes), "Allocate output image") ||
        !checked(cudaMemcpy(device_in, input, bytes, cudaMemcpyHostToDevice), "Copy image") ||
        !checked(cudaEventCreate(&start), "Create start event") ||
        !checked(cudaEventCreate(&stop), "Create stop event") ||
        !checked(cudaEventRecord(start), "Record start")) goto cleanup;
    sobel_edges<<<(unsigned)blocks, 256>>>(device_in, device_out, width, height);
    if (!checked(cudaGetLastError(), "Launch Sobel kernel") ||
        !checked(cudaEventRecord(stop), "Record stop") ||
        !checked(cudaEventSynchronize(stop), "Wait for Sobel kernel") ||
        !checked(cudaEventElapsedTime(&ms, start, stop), "Measure kernel") ||
        !checked(cudaMemcpy(output, device_out, bytes, cudaMemcpyDeviceToHost), "Copy edge map")) goto cleanup;
    error = lodepng_encode32_file(out_name, output, width, height);
    if (error) { fprintf(stderr, "%s: %s\n", out_name, lodepng_error_text(error)); goto cleanup; }
    printf("%s -> %s | %ux%u | %zu blocks x 256 | Kernel %.3f ms\n", filename, out_name, width, height, blocks, ms);
    status = 0;
cleanup:
    if (start && !checked(cudaEventDestroy(start), "Destroy start event")) status = 1;
    if (stop && !checked(cudaEventDestroy(stop), "Destroy stop event")) status = 1;
    if (device_in && !checked(cudaFree(device_in), "Free device input")) status = 1;
    if (device_out && !checked(cudaFree(device_out), "Free device output")) status = 1;
    free(input); free(output); free(out_name); return status;
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s image1.png [image2.png ...]\n", argv[0]); return 1; }
    // Output names use input basenames. Reject collisions before processing.
    for (int i = 1; i < argc; ++i) for (int j = 1; j < i; ++j) {
        const char *a = strrchr(argv[i], '/'), *b = strrchr(argv[j], '/');
        a = a ? a+1 : argv[i]; b = b ? b+1 : argv[j];
        if (!strcmp(a,b)) { fprintf(stderr, "Duplicate image basename: %s\n", a); return 1; }
    }
    int failed = 0;
    for (int i = 1; i < argc; ++i) failed += process_image(argv[i]) != 0;
    printf("Images completed: %d | Failed: %d\n", argc-1-failed, failed);
    return failed ? 1 : 0;
}
