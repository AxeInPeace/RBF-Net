/*
#include "RBFNet.h"
#include "RBFNeuron.h"
#include <vector>

#define uint unsigned int

using namespace std;

//=========================================== INITIALIZATION =================================================
template<class I, class O, class K>
RBFNet<I, O, K>::RBFNet(){
	_inLayerSize = 0;
	_hiddenLayerSize = 0;
	_outLayerSize = 0;

}

template<class I, class O, class K>
RBFNet<I, O, K>::~RBFNet(){
	for(int i = _hiddenLayerSize - 1; i >= 0; i--) {
		delete _hiddenNeur[i];
	}
}

template<class I, class O, class K>
RBFNet<I, O, K>::RBFNet(int inLayer, int hidLayer, int outLayer){	
	_inLayerSize = inLayer;
	_hiddenLayerSize = hidLayer;
	_outLayerSize = outLayer;

	_initNeurons();
}

template<class I, class O, class K>
RBFNet<I, O, K>::RBFNet(I inObj, int hidLayer, O outObj) { //require method .size() of both objects
	_inLayerSize = inObj.size();
	_hiddenLayerSize = hidLayer;
	_outLayerSize = outObj.size();

	_initNeurons();
}

template<class I, class O, class K>
void RBFNet<I, O, K>::initNeuronsWithFunc(O (*evaluateFunc)(I, K )){
	for (int i = 0; i < _hiddenLayerSize; i++){
		_hiddenNeur[i]->setFunc(evaluateFunc);		
	}
}

template<class I, class O, class K>
void RBFNet<I, O, K>::_initNeurons(){
	_hiddenNeur.resize(_hiddenLayerSize);
	for(int i = 0; i < _hiddenLayerSize; i++){
		_hiddenNeur[i] = new RBFNeuron<I, O, K>();
	}
}

template<class I, class O, class K>
void RBFNet<I, O, K>::setZeroFunc(void (*zeroFunc)(O)){
	_getZeroObj = zeroFunc;
}
//+++++++++++++++++++++++++++++++++++++++++++ INITIALIZATION +++++++++++++++++++++++++++++++++++++++++++++++++


//=========================================== GETTING RESULTS =================================================
template<class I, class O, class K>
RBFNeuron<I, O, K>* RBFNet<I, O, K>::getNeur(int position){
	return _hiddenNeur[position];
}

template<class I, class O, class K>
int RBFNet<I, O, K>::getSize(){
	return _hiddenLayerSize;
}

template<class I, class O, class K>
O RBFNet<I, O, K>::evaluate(I inputValue){
	O ans = 0;

	for (int i = 0; i < _hiddenLayerSize; i++){	
		ans = ans + _hiddenNeur[j]->evaluate(inputValue);
	}
	return ans;
}

template<class I, class O, class K>
float RBFNet<I, O, K>::test(vector<I*>* inVals, vector<O*>* outVals){
	unsigned int numOfTest = inVals.size();
	unsigned int successfullyTested = 0;

	if (numOfTest != outVals.size())
		return -1;

	for (unsigned int i = 0; i < numOfTest; i++){
		if ( this->evaluate(inVals[i]) == outVals[i])
			successfullyTested++;
	}

	return float(successfullyTested)/numOfTest;
}
//+++++++++++++++++++++++++++++++++++++++++++ GETTING RESULTS +++++++++++++++++++++++++++++++++++++++++++++++++

*/