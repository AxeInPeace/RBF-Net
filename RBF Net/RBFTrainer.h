#pragma once

#include "RBFNet.h"
#include "GeneticLearn.h"

using namespace std;

template <class I, class O, class K>
class RBFTrainer
{
public:
	RBFTrainer();
	~RBFTrainer();
	void initLearnAgent(GeneticLearn<I, O, K>*);
	void initNet(RBFNet<I, O, K>*);

	RBFNet<I, O, K>* train(vector<I*> inputValues, vector<O*> outputValues, vector<I*> testInValues, vector<O*> testOutValues);
private:
	GeneticLearn<I, O, K>* _learnAgent;
	RBFNet<I, O, K>* _net;
};

