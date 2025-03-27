#pragma once
#include <vector>

class Cloth;
class Spring;

class Node
{
private:
	float mass;
public:
	int num_springs;

	float position[2];
	float velocity[2];

	bool is_fixed;

	Spring* springs[4];

	Node(float x, float y, bool is_fixed);
	~Node();

	void add_spring(Spring* spring);

	friend Spring;
	friend Cloth;
};

