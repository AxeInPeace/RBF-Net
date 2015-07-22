#pragma once

#include "RBFNet.h"
#include "VirtLearn.h"

using namespace std;

template <class I, class O, class K>
class RBFTrainer
{
public:
	RBFTrainer();
	~RBFTrainer();
	void initLearnAgent(VirtLearn<I, O, K>*);
	void initNet(RBFNet<I, O, K>*);

	RBFNet<I, O, K>* train(vector<I*> inputValues, vector<O*> outputValues, vector<I*> testInValues, vector<O*> testOutValues);
private:
	VirtLearn<I, O, K>* _learnAgent;
	RBFNet<I, O, K>* _net;
};

template<class I, class O, class K> 
RBFTrainer<I, O, K>::RBFTrainer(){
	;
}

template<class I, class O, class K> 
RBFTrainer<I, O, K>::~RBFTrainer(){
	;
}

template<class I, class O, class K> 
RBFNet<I, O, K>* RBFTrainer<I, O, K>::train(vector<I*> inVec, vector<O*> outVec, vector<I*> testInVec, vector<O*> testOutVec){
	_learnAgent->learning(&inVec, &outVec, &testInVec, &testOutVec, _net);
	return _net;
}

template<class I, class O, class K> 
void RBFTrainer<I, O, K>::initLearnAgent(VirtLearn<I, O, K>* newTeacher){
	_learnAgent = newTeacher;
}

template<class I, class O, class K> 
void RBFTrainer<I, O, K>::initNet(RBFNet<I, O, K>* newNet){
	_net = newNet;
}
