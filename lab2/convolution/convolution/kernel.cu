#include <stdlib.h>
#include <stdio.h>

// cuda runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// png processing lib
#include "lodepng.h"

// Weight Matrix
#include "wm.h"
#include <math.h>

#define wm_size (sizeof(w) / sizeof(w[0]))

__global__ void convolution(unsigned char* image, unsigned char* conv_image, int height, int width, float wm[wm_size][wm_size], int thread_count)
{
	float rows = (float) height / (float) thread_count;
	float start = threadIdx.x * rows;
	float end = (threadIdx.x+1) * rows;

	if (threadIdx.x == 0) {
		start = wm_size/2;
		printf("%d\n", width);
	}
	if (threadIdx.x == thread_count-1) {
		end -= wm_size/2;
	}

	printf("conv %d | %f %f\n", threadIdx.x, start, end);
	int index = 0;
	for (int i = (int) start; i < (int) end; i++) {
		for (int j = 1; j < width-1; j++) {

			// For each colour
			for (int color = 0; color < 4; color++) {

				// For elements in weighted matrix
				
				float sum_float = 0;
				unsigned char sum;
				if (color != 3) {
					for (int ii = 0; ii < wm_size; ii++) {
						for (int jj = 0; jj < wm_size; jj++) {

							sum_float += image[(i + ii - 1) * width * 4 + (j * 4 + (jj - 1) * 4) + color] * wm[ii][jj];
						}
					}
					if (sum_float > 255) {
						sum_float = 255;
					}
					if (sum_float < 0) {
						sum_float = 0;
					}
				}
				else {
					sum_float = image[i * width * 4 + (j * 4) + color];
				}

				sum = (unsigned char)sum_float;
				conv_image[(i - 1) * (width-2) * 4 + (j - 1)* 4 + color] = sum;
			}
		}
	}
}

int main(int argc, char* argv[])
{
	char* input_filename = "H:\\ecse420\\convolution\\convolution\\test.png";
	char* output_filename = "H:\\ecse420\\convolution\\convolution\\test_convolve.png";
	int thread_count = 100; //atoi(argv[3]);
	int png_length, png_length_conv;
	unsigned error;
	unsigned char* image, * new_image, * convolution_image;
	unsigned width, height;

	float wm_vals[wm_size][wm_size];
	float* wm;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			wm_vals[i][j] = w[i][j];
		}
	}

	// Load png file
	error = lodepng_decode32_file(&image, &width, &height, input_filename);

	// Check if there was an error
	if (error) {
		exit(error);
	}

	// Calculate length of the loaded png
	png_length = width * height * 4 * sizeof(unsigned char);
	png_length_conv = (width - 1 - wm_size/2) * (height - 1 - wm_size/2) * 4;

	// Allocate space in Unified Memory for image
	cudaMallocManaged((void**)& new_image, png_length * sizeof(unsigned char));
	cudaMallocManaged((void**)& convolution_image, png_length_conv * sizeof(unsigned char));
	cudaMallocManaged((void**)& wm, wm_size * wm_size * sizeof(float));

	std::copy(&wm_vals[0][0], &wm_vals[0][0] + wm_size * wm_size, wm);

	// Initialize png data array to be sent to GPU
	for (int i = 0; i < png_length; i++) {
		new_image[i] = image[i];
	}

	// Launch pool() kernel on GPU with thread_count threads
	printf("%d %d\n", height, width);
	convolution << <1, thread_count >> > (new_image, convolution_image, height, width, reinterpret_cast<float(*)[wm_size]>(wm), thread_count);

	// Wait for GPU threads to complete
	cudaDeviceSynchronize();

	// Write the raw image data to file
	lodepng_encode32_file(output_filename, convolution_image, width - 1 - wm_size/2, height - 1 - wm_size/2);

	// Cleanup
	cudaFree(new_image);
	cudaFree(convolution_image);
	free(image);

    return 0;
}
