#include "Cloth.h"
#include <helper_math.h>


Cloth::Cloth(const int width, const int height, const bool gravity)
{
	cudaMallocManaged((void**)&this->nodes, width * height * sizeof(Node));
	cudaMallocManaged((void**)&this->springs, ((height * (width - 1)) + (width * (height - 1))) * sizeof(Spring));

	this->enabled_gravity = gravity;
	this->damping_coe = 0.03f;
	this->mass_per_node = 0.01f;
	this->spring_coe = 10.0f;
	this->rest_length = 1.0f;

	this->height = height;
	this->width = width;

	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
		{
			if((x == 0 || x == width - 1) && y == height - 1)
			{
				this->nodes[x + y * width] = Node(x, y, true);
				continue;
			}

			this->nodes[x + y * width] = Node(x, y, false);
		}
	}

	int index = 0;
	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
		{
			if(x < width - 1)
			{
				this->springs[index] = Spring(&this->nodes[y * width + x], &this->nodes[y * width + x + 1]);
				nodes[y * width + x].add_spring(&this->springs[index]);
				nodes[y * width + x + 1].add_spring(&this->springs[index]);
				index++;
			}
			if(y < height - 1)
			{
				this->springs[index] = Spring(&this->nodes[y * width + x], &this->nodes[(y + 1) * width + x]);
				nodes[y * width + x].add_spring(&this->springs[index]);
				nodes[(y + 1) * width + x].add_spring(&this->springs[index]);
				index++;
			}
		}
	}

	num_springs = index;
}

void Cloth::toggle_gravity()
{
	this->enabled_gravity = !this->enabled_gravity;
}

Cloth::~Cloth()
{
	cudaFree(this->nodes);
	cudaFree(this->springs);
}
