#include "Spring.h"


#include <helper_math.h>


Spring::Spring(Node* node1, Node* node2)
{
	this->node1 = node1;
	this->node2 = node2;

	this->force_x = 0;
	this->force_y = 0;
}

Spring::~Spring()
{
	
}

