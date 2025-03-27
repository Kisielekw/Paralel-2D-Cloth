/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <iostream>
#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>

#include "Cloth.h"

typedef unsigned int uint;
typedef unsigned char uchar;


cudaArray* d_imageArray = 0;


__global__ void d_render(uchar4* d_output, uint width, uint height, Spring* springs, const int num_springs) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    
    if ((x < width) && (y < height)) 
    {
        d_output[i] = make_uchar4(255, 255, 255, 255);

        float u = x / (float)width;
        float v = y / (float)height;

        v = 1 - v;

        int local_x = static_cast<int>(u * static_cast<float>(width));
        int local_y = static_cast<int>(v * static_cast<float>(height));

        for (int j = 0; j < num_springs; j++)
		{
			Spring* s = &springs[j];
			float2 p1 = make_float2(s->node1->position[0] * 10 + 150, s->node1->position[1] * 10 + 150);
			float2 p2 = make_float2(s->node2->position[0] * 10 + 150, s->node2->position[1] * 10 + 150);

            float2 spring_vec = p2 - p1;
            float2 point_pixel = make_float2(local_x, local_y) - p1;

            float dot = spring_vec.x * point_pixel.x + spring_vec.y * point_pixel.y;
            float projection = dot / sqrt(spring_vec.x * spring_vec.x + spring_vec.y * spring_vec.y);

            if(projection < 0 || projection > 1)
				continue;

            float2 closest = p1 + spring_vec * projection;
			float2 diff = closest - make_float2(local_x, local_y);

			float mag = sqrt(diff.x * diff.x + diff.y * diff.y);

            if(mag < 10)
				d_output[i] = make_uchar4(0, 0, 0, 255);
		}

    }
}

__global__ void spring_kernel(Spring* springs, const int num_springs, const float spring_coe, const float rest_length)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= num_springs)
	{
		return;
	}

	float dx = springs[index].node2->position[0] - springs[index].node1->position[0];
	float dy = springs[index].node2->position[1] - springs[index].node1->position[1];

	float distance = sqrt(dx * dx + dy * dy);

	float force = spring_coe * (distance - rest_length);

	float fx = force * dx / distance;
	float fy = force * dy / distance;

	springs[index].force_x = fx;
	springs[index].force_y = fy;
}

__global__ void node_kernel(Node* nodes, const int num_nodes, const int width, const float dampening, const float mass, const bool gravity, const float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= num_nodes / width)
	{
		return;
	}

	int index = x + y * width;

	Node* node = &nodes[index];

	float acceleration_x = 0;
	float acceleration_y = 0;

	if (gravity)
	{
		acceleration_y += -9.8f;
	}

	if (!node->is_fixed)
	{
		for (int i = 0; i < node->num_springs; i++)
		{
			float fx;
			float fy;
			if (node->springs[i]->node1 == node)
			{
				fx = node->springs[i]->force_x;
				fy = node->springs[i]->force_y;
			}
			else
			{
				fx = -node->springs[i]->force_x;
				fy = -node->springs[i]->force_y;
			}

			acceleration_x += fx / mass;
			acceleration_y += fy / mass;
		}

		float fx = -node->velocity[0] * dampening;
		float fy = -node->velocity[1] * dampening;

		acceleration_x += fx / mass;
		acceleration_y += fy / mass;

		float new_x = node->position[0] + (node->velocity[0] * dt) + (0.5 * acceleration_x * dt * dt);
		float new_y = node->position[1] + (node->velocity[1] * dt) + (0.5 * acceleration_y * dt * dt);

		if (dt > 0.0f)
		{
			node->velocity[0] = (new_x - node->position[0]) / dt;
			node->velocity[1] = (new_y - node->position[1]) / dt;
		}

		node->position[0] = new_x;
		node->position[1] = new_y;
	}
}

extern "C" void freeTexture() {

    checkCudaErrors(cudaFreeArray(d_imageArray));
}

// render image using CUDA
extern "C" void render(int width, int height,  dim3 blockSize, dim3 gridSize,
     uchar4 * output, Cloth* cloth) {

	dim3 spring_block = dim3(256);
	dim3 spring_grid = dim3(ceil(static_cast<float>(cloth->num_springs) / 256.0f));

	dim3 node_block = dim3(16, 16);
	dim3 node_grid = dim3(ceil(static_cast<float>(cloth->width) / 16.0f), ceil(static_cast<float>(cloth->height) / 16.0f));

	auto start = std::chrono::high_resolution_clock::now();

	spring_kernel << < spring_grid, spring_block >> > (cloth->springs, cloth->num_springs, cloth->spring_coe, cloth->rest_length);
	cudaDeviceSynchronize();

	node_kernel << < node_grid, node_block >> > (cloth->nodes, cloth->width * cloth->height, cloth->width, cloth->damping_coe, cloth->mass_per_node, cloth->enabled_gravity, 0.01f);
	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

            d_render << <gridSize, blockSize >> > (output, width, height, cloth->springs, cloth->num_springs);


    getLastCudaError("kernel failed");
}

#endif