#pragma once

#include <vector>
#include "RBFNeuron.h"


using namespace std;

template <class I, class O, class K>
class RBFNet
{
public:
	RBFNet();
	RBFNet(int, int, int);
	RBFNet (I, int, O);	
	void initNeuronsWithFunc(O (*evaluateFunc)(I, K));
	void setZeroFunc(void (*zeroFunc)(O));
	~RBFNet();

	RBFNeuron<I, O, K>* getNeur(int);	
	int getSize();	
	float test(vector<I*>*, vector<O*>*);
	O evaluate(I);



private:
	int _hiddenLayerSize;
	int _inLayerSize;
	int _outLayerSize;
	void _initNeurons();
	vector<RBFNeuron<I, O, K>*> _hiddenNeur;
	
	void (*_getZeroObj)(O);
	//friend O operator+(O, O);	
};