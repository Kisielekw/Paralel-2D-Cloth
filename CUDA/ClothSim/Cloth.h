#pragma once
#include <vector>
#include "Node.h"
#include "Spring.h"

class Cloth
{
private:
	

public:
	Node* nodes;

	bool enabled_gravity;

	float spring_coe;
	float damping_coe;
	float mass_per_node;
	float rest_length;

	int width;
	int height;
	Spring* springs;

	int num_springs;

	Cloth(const int width, const int height, const bool gravity);
	~Cloth();

	void toggle_gravity();
};

