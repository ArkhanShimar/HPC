#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// The arithmetic and boundary rules are preserved from CryptForCuda.c.
// A caller-owned array replaces the original static buffer for thread safety.
__device__ void cudaCrypt(const char *raw, char *encrypted) {
    encrypted[0] = raw[0]+2; encrypted[1] = raw[0]-2; encrypted[2] = raw[0]+1;
    encrypted[3] = raw[1]+3; encrypted[4] = raw[1]-3; encrypted[5] = raw[1]-1;
    encrypted[6] = raw[2]+2; encrypted[7] = raw[2]-2;
    encrypted[8] = raw[3]+4; encrypted[9] = raw[3]-4; encrypted[10] = '\0';
    for (int i = 0; i < 10; ++i) {
        int low = i < 6 ? 97 : 48, high = i < 6 ? 122 : 57;
        if (encrypted[i] > high) encrypted[i] = (encrypted[i]-high)+low;
        else if (encrypted[i] < low) encrypted[i] = (low-encrypted[i])+low;
    }
}
__global__ void recover_passwords(const char *encrypted, char *plain, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    char raw[5], candidate[11]; raw[4] = '\0';
    // One password per thread. The grid grows with the number of input lines.
    for (int code = 0; code < 26*26*100; ++code) {
        raw[0] = 'a' + code / 2600;
        raw[1] = 'a' + (code / 100) % 26;
        raw[2] = '0' + (code / 10) % 10; raw[3] = '0' + code % 10;
        cudaCrypt(raw, candidate);
        int same = 1;
        for (int j = 0; j < 10; ++j) if (candidate[j] != encrypted[i*11+j]) { same = 0; break; }
        if (same) { for (int j = 0; j < 5; ++j) plain[i*5+j] = raw[j]; return; }
    }
}
static int checked(cudaError_t code, const char *action) {
    if (code == cudaSuccess) return 1;
    fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(code)); return 0;
}
static int run_gpu(const char *encrypted, char *plain, size_t count) {
    char *device_input = NULL, *device_output = NULL;
    cudaDeviceProp properties;
    int status = 1, threads = 128;
    size_t blocks = (count + threads - 1) / threads;
    if (!checked(cudaGetDeviceProperties(&properties, 0), "GPU properties")) goto cleanup;
    if (blocks > (size_t)properties.maxGridSize[0]) { fprintf(stderr, "Input exceeds the GPU grid limit.\n"); goto cleanup; }
    if (!checked(cudaMalloc((void **)&device_input, count*11), "Allocate encrypted passwords") ||
        !checked(cudaMalloc((void **)&device_output, count*5), "Allocate recovered passwords") ||
        !checked(cudaMemcpy(device_input, encrypted, count*11, cudaMemcpyHostToDevice), "Copy input") ||
        !checked(cudaMemset(device_output, 0, count*5), "Clear output")) goto cleanup;
    recover_passwords<<<(unsigned)blocks, threads>>>(device_input, device_output, count);
    if (!checked(cudaGetLastError(), "Launch password kernel") ||
        !checked(cudaDeviceSynchronize(), "Wait for kernel") ||
        !checked(cudaMemcpy(plain, device_output, count*5, cudaMemcpyDeviceToHost), "Copy results")) goto cleanup;
    printf("GPU: %s | Passwords: %zu | Blocks: %zu | Threads/block: %d\n", properties.name, count, blocks, threads);
    status = 0;
cleanup:
    if (device_input && !checked(cudaFree(device_input), "Free input")) status = 1;
    if (device_output && !checked(cudaFree(device_output), "Free output")) status = 1;
    return status;
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "Usage: %s passwords.txt\n", argv[0]); return 1; }
    FILE *input = fopen(argv[1], "r");
    if (!input) { perror(argv[1]); return 1; }
    size_t count = 0, capacity = 256, missing = 0;
    char *encrypted = (char *)malloc(capacity*11), *plain = NULL;
    char line[32]; FILE *output = NULL; int status = 1;
    if (!encrypted) goto cleanup;
    while (fgets(line, sizeof(line), input)) {
        size_t length = strcspn(line, "\r\n");
        if (length != 10 || (line[length] == '\r' && line[length+1] && line[length+1] != '\n')) {
            fprintf(stderr, "Line %zu: expected 10 encrypted characters.\n", count+1); goto cleanup;
        }
        for (int j = 0; j < 10; ++j) {
            if ((j < 6 && (line[j] < 'a' || line[j] > 'z')) || (j >= 6 && (line[j] < '0' || line[j] > '9'))) {
                fprintf(stderr, "Line %zu: invalid encrypted character.\n", count+1); goto cleanup;
            }
        }
        if (count == capacity) {
            if (capacity > SIZE_MAX / 22) goto cleanup;
            char *larger = (char *)realloc(encrypted, capacity*22);
            if (!larger) goto cleanup;
            encrypted = larger; capacity *= 2;
        }
        memcpy(encrypted+count*11, line, 10); encrypted[count*11+10] = '\0'; count++;
    }
    if (ferror(input) || !count) { fprintf(stderr, "Cannot read passwords or file is empty.\n"); goto cleanup; }
    plain = (char *)calloc(count, 5);
    if (!plain || run_gpu(encrypted, plain, count)) goto cleanup;
    output = fopen("decrypted.txt", "w");
    if (!output) { perror("decrypted.txt"); goto cleanup; }
    for (size_t i = 0; i < count; ++i) {
        if (plain[i*5]) fprintf(output, "%s\n", plain+i*5);
        else { fprintf(output, "NOT_FOUND\n"); missing++; }
    }
    if (ferror(output)) goto cleanup;
    printf("Recovered: %zu | Not found: %zu | Output: decrypted.txt\n", count-missing, missing);
    status = missing ? 2 : 0;
cleanup:
    if (fclose(input)) status = 1;
    if (output && fclose(output)) status = 1;
    free(encrypted); free(plain); return status;
}
