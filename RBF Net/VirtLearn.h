#pragma once
#include <vector>
#include "RBFNet.h"

using namespace std;

template <class I, class O, class K>
class VirtLearn
{
public:
	VirtLearn(void) {;}
	~VirtLearn(void) {;}

	virtual void learning(vector<I*>*, vector<O*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*) = 0;
};

