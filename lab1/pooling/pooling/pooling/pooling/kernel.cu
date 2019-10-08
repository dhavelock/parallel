#include <stdlib.h>
#include <stdio.h>

// cuda runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// png processing lib
#include "lodepng.h"

/* Divides image into sections based off the number of threads */
void getDimensions(int thread_count, int* w, int* h) {
	*w = 1;
	*h = 1;
	bool side = true;
	while (thread_count > 1) {
		if (side) {
			*w *= 2;
		}
		else {
			*h *= 2;
		}
		side = !side;
		thread_count /= 2;
	}
}

/* GPU function */
__global__ void pool(unsigned char* image, unsigned char* pool_image, int pixelWidth, int pixelHeight, int w, int h) {
	int thread_index = threadIdx.x;

	// index of the section of pixels assigned to the thread
	int col = thread_index % w;
	int row = thread_index / w;
	
	// width and height of the section of pixels assigned to the thread
	int sector_width = pixelWidth * 4 / w;
	int sector_height = pixelHeight / h;

	sector_width -= sector_width % 2; // must be even sized sectors
	sector_height -= sector_height % 2;

	// stores values and indices of 2x2 grid
	int max_val, t_left, t_right, b_left, b_right;
	int t_left_index, t_right_index, b_left_index, b_right_index;

	// index of pixel in the context of the full matrix of pixels
	int scaled_col, scaled_row;

	// index of pool_image array to write to
	int pool_index, offset;

	for (int i = 0; i <= sector_height; i += 2) {

		pool_index = sector_width * col / 2 + (i/2 + sector_height / 2 * row) * pixelWidth * 2;
		offset = pool_index % 4;
		pool_index = pool_index - offset;

		for (int j = 0 - offset; j <= sector_width + 4 - offset; j += 8) {

			// Calculate pixel location index of image[]
			scaled_row = i + sector_height * row;
			
			scaled_col = j + sector_width * col;

			// Align indices such that
			scaled_col = scaled_col - scaled_col % 4;

			// iterate through each colour
			for (int colour = 0; colour < 4; colour++) {
				t_left = 0;
				t_right = 0;
				b_left = 0;
				b_right = 0;

				// Get index of each corner of 2x2 region
				t_left_index = scaled_col + colour + scaled_row * pixelWidth * 4;
				t_right_index = scaled_col + colour + 4 + scaled_row * pixelWidth * 4;
				b_left_index = scaled_col + colour + (scaled_row + 1) * pixelWidth * 4;
				b_right_index = scaled_col + colour + 4 + (scaled_row + 1) * pixelWidth * 4;

				// Get value of each corner of 2x2 region, checking for edges
				if (t_left < pixelWidth * pixelHeight * 4) {
					t_left = image[t_left_index];
				}
				

				if (scaled_col + colour + 4 < pixelWidth * 4) {
					t_right = image[t_right_index];
				}

				if (b_left_index < pixelWidth * pixelHeight * 4) {
					b_left = image[b_left_index];
				}

				if (scaled_col + colour + 4 < pixelWidth * 4 && b_right_index < pixelWidth * pixelHeight * 4) {
					b_right = image[b_right_index];
				}

				// Calculate max
				max_val = t_left;
				if (t_right > max_val) {
					max_val = t_right;
				}
				if (b_left > max_val) {
					max_val = b_left;
				}
				if (b_right > max_val) {
					max_val = b_right;
				}

				if (pool_index < pixelWidth * pixelHeight) {
					pool_image[pool_index++] = max_val;
				}
				else {
					break;
				}
			}
		}
	}
}

int main(int argc, char* argv[]) {

	char input_filename[] = "C:\\Users\\dhavel\\ecse420\\test.png"; //argv[1];
	char output_filename[] = "C:\\Users\\dhavel\\ecse420\\test_pooling.png"; //argv[2];
	int thread_count = 1024; //atoi(argv[3]);
	int png_length;
	unsigned error;
	unsigned char* image, * new_image, *pool_image;
	unsigned width, height;

	int numSectorsX, numSectorsY;
	getDimensions(thread_count, &numSectorsX, &numSectorsY);

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
	cudaMallocManaged((void**)& pool_image, png_length / 4 * sizeof(unsigned char));

	// Initialize png data array to be sent to GPU
	for (int i = 0; i < png_length; i++) {
		new_image[i] = image[i];
	}

	// Launch pool() kernel on GPU with thread_count threads
	pool << <1, thread_count >> > (new_image, pool_image, width, height, numSectorsX, numSectorsY);

	// Wait for GPU threads to complete
	cudaDeviceSynchronize();

	// Write the raw image data to file
	lodepng_encode32_file(output_filename, pool_image, width/2, height/2);

	// Cleanup
	cudaFree(new_image);
	cudaFree(pool_image);
	free(image);

	return 0;
}