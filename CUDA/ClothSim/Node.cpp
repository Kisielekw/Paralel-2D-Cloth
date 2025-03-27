#include "Node.h"
#include "Spring.h"

#include <helper_math.h>

Node::Node(const float x, const float y, const bool is_fixed)
{
	this->mass = 0.01f;

	this->num_springs = 0;

	this->position[0] = x;
	this->position[1] = y;

	this->velocity[0] = 0;
	this->velocity[1] = 0;

	this->is_fixed = is_fixed;
}

void Node::add_spring(Spring* spring)
{
	this->springs[num_springs] = spring;
	this->num_springs++;
}

Node::~Node()
{
	
}
