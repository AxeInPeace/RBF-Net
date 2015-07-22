#pragma once
#include <vector>

using namespace std;

class Weight
{
public:
	Weight();
	~Weight();
private:
	float weight;
};

template <class I, class O, class K>
class RBFNeuron
{
public:
	RBFNeuron();
	~RBFNeuron();

	O evaluate (I);
	O evaluate (I, int);
	void changeKoef(K*);
	K getKoef();

	void changeWeight (int, float);
	float getWeight (int);
	//void changeWeight(int, int, Weight);

	void setFunc(O (*newFunc)(I, K));
//	O (*)(I, K) getFunc();

private:
	O (*_evaluateFunc)(I inLayer, K koefs);
	K _koef;
	vector <float> _weight;
	//vector <Weight*> _weight;
	//friend O operator*(O, Weight);
};

#include "RBFNeuron.h"

template <class I, class O, class K>
RBFNeuron<I, O, K>::RBFNeuron(void)
{
	;
}

template <class I, class O, class K>
RBFNeuron<I, O, K>::~RBFNeuron(void)
{
	/*
	for(int i = 0; i < _weight.size(); i++){
		delete _weight[i];
	}
	*/
}

template <class I, class O, class K>
O RBFNeuron<I, O, K>::evaluate (I inVal){
	return _evaluateFunc(inVal, _koef);
}

template <class I, class O, class K>
O RBFNeuron<I, O, K>::evaluate (I inVal, int weightNum){
	return _evaluateFunc(inVal, koef) * weight[weightNum];
}


template <class I, class O, class K>
void RBFNeuron<I, O, K>::changeKoef(K* newKoef){
	_koef = *newKoef;			
}

template <class I, class O, class K>
K RBFNeuron<I, O, K>::getKoef(){
	return _koef;
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