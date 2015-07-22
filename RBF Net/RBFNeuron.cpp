/*
#include "RBFNeuron.h"

template <class I, class O, class K>
RBFNeuron<I, O, K>::RBFNeuron(void)
{
	;
}

template <class I, class O, class K>
RBFNeuron<I, O, K>::~RBFNeuron(void)
{
	for(int i = 0; i < _weight.size(); i++){
		delete _weight[i];
	}
}

template <class I, class O, class K>
O RBFNeuron<I, O, K>::evaluate (I inVal){
	return _evaluateFunc(inVal, koef);
}

template <class I, class O, class K>
O RBFNeuron<I, O, K>::evaluate (I inVal, int weightNum){
	return _evaluateFunc(inVal, koef) * weight[weightNum];
}


template <class I, class O, class K>
void RBFNeuron<I, O, K>::changeKoef(K* newKoef){
	koef = *newKoef;			
}

template <class I, class O, class K>
K RBFNeuron<I, O, K>::getKoef(){
	return koef;
}

template <class I, class O, class K>
void RBFNeuron<I, O, K>::setFunc(O (*newFunc)(I, K)){
	_evaluateFunc = newFunc;
}

template <class I, class O, class K>
void RBFNeuron<I, O, K>::changeWeight (int toNeur, float newWeight){
	_weight[toNeur] = newWeight;
}

template <class I, class O, class K>
float RBFNeuron<I, O, K>::getWeight (int toNeur){
	return _weight[toNeur];
}

*/