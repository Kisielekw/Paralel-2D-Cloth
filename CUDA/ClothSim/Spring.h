#pragma once
#include "Node.h"

class Spring
{
public:
	Node* node1;
	Node* node2;

	float force_x;
	float force_y;

	Spring(Node* node1, Node* node2);
	~Spring();
};
