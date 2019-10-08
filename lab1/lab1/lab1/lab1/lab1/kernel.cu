#include <stdlib.h>
#include <stdio.h>

// cuda runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// png processing lib
#include "lodepng.h"

__global__ void rectify(unsigned char* image, int png_length, int n) {
	int thread_index = threadIdx.x;

	for (int i = thread_index * (png_length / n); i < (thread_index + 1) * (png_length / n); i++) {
		if (image[i] < 127) {
			image[i] = (unsigned char)127;
		}
	}
}

int main(int argc, char* argv[]) {

	char input_filename[] = "C:\\Users\\dhavel\\ecse420\\test.png"; //argv[1];
	char output_filename[] = "C:\\Users\\dhavel\\ecse420\\test_rectify.png"; //argv[2];
	int thread_count = 1024; //atoi(argv[3]);
	int png_length;
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;

	// Load png file
	error = lodepng_decode32_file(&image, &width, &height, input_filename);

	// Check if there was an error
	if (error) {
		exit(error);
	}

	// Calculate length of the loaded png
	png_length = width * height * 4 * sizeof(unsigned char);

	// Allocate space in Unified Memory for image
	cudaMallocManaged((void**)& new_image, png_length * sizeof(unsigned char));

	// Initialize png data array to be sent to GPU
	for (int i = 0; i < png_length; i++) {
		new_image[i] = image[i];
	}

	// Launch rectify() kernel on GPU with thread_count threads
	rectify << <1, thread_count >> > (new_image, png_length, thread_count);

	// Wait for GPU threads to complete
	cudaDeviceSynchronize();

	// Write the raw image data to file
	lodepng_encode32_file(output_filename, new_image, width, height);

	// Cleanup
	cudaFree(new_image);
	free(image);

	return 0;
}