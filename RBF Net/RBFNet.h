#pragma once

#include <vector>
#include "RBFNeuron.h"
#include <Eigen/Dense>
#include <string>
#include <fstream>

typedef unsigned int uint;

using Eigen::MatrixXf;
using namespace std;

template <class I, class O, class K>
class RBFNet
{
public:
	RBFNet();
	RBFNet(int, int, int);
	RBFNet (I, int, O);	
	~RBFNet();

	void initNeuronsWithFunc(O (*evaluateFunc)(I, K));
	void setZeroFunc(void (*zeroFunc)(O));
	int putKoefsInNerurons(vector<K*>*);
	int putWeightsInNeurons(MatrixXf);

	RBFNeuron<I, O, K>* getNeur(int);	
	int getSize();	
	float test(vector<I*>*, vector<O*>*);
	O evaluate(I);
	O evaluate(I, int);

	int importFile(string);
	int exportFile(string);

	void initErrorValue(O);

private:
	int _hiddenLayerSize;
	int _inLayerSize;
	int _outLayerSize;
	void _initNeurons();
	vector<RBFNeuron<I, O, K>*> _hiddenNeur;

	O _errVal;
	float _gaussFunc(O, O);
	
	void (*_getZeroObj)(O);
	//friend O operator+(O, O);	
};


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
void RBFNet<I, O, K>::initErrorValue(O errVal){
	_errVal = errVal;
}


template<class I, class O, class K>
void RBFNet<I, O, K>::setZeroFunc(void (*zeroFunc)(O)){
	_getZeroObj = zeroFunc;
}

template<class I, class O, class K>
int RBFNet<I, O, K>::putKoefsInNerurons(vector<K*>* koefs){
	if(koefs->size() != _hiddenLayerSize)
		return -1;
	
	RBFNeuron<I, O, K>* curNeur;
	for (int i = 0; i < _hiddenLayerSize; i++){
		curNeur = _hiddenNeur[i];
		curNeur->changeKoef(koefs->at(i));
	}

	return 0;
}

template<class I, class O, class K>
int RBFNet<I, O, K>::putWeightsInNeurons(MatrixXf weightVector){
	
	RBFNeuron<I, O, K>* curNeur;
	for (int i = 0; i < _hiddenLayerSize; i++){
		curNeur = _hiddenNeur[i];
		curNeur->changeWeight(0, weightVector(i,0));
	}

	return 0;
}


//+++++++++++++++++++++++++++++++++++++++++++ INITIALIZATION +++++++++++++++++++++++++++++++++++++++++++++++++

//=========================================== WORK WITH FILES =================================================
template<class I, class O, class K>
int RBFNet<I, O, K>::importFile(string nameOfFile){
	ifstream file(nameOfFile, ios::in);
	for (int i = 0; i < _hiddenLayerSize; i++){
		RBFNeuron<I, O, K>* curNeur = _hiddenNeur[i];
		K koefs;				
		for(int j = 0; j < koefs->size(); j++){
			file >> koefs[j];
		}
		curNeur->changeKoef(&koefs);
	}
	return 0;
}

template<class I, class O, class K>
int RBFNet<I, O, K>::exportFile(string nameOfFile){
	ofstream file(nameOfFile, ios::out);
	for (int i = 0; i < _hiddenLayerSize; i++){
		RBFNeuron<I, O, K>* curNeur = _hiddenNeur[i];		
		K koefs = (curNeur->getKoef());
		for(int j = 0; j < koefs.size(); j++){
			file << koefs[j] << endl;
		}
		curNeur->changeKoef(&koefs);
	}
	return 0;
}
//+++++++++++++++++++++++++++++++++++++++++++ WORK WITH FILES +++++++++++++++++++++++++++++++++++++++++++++++++

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
		ans = ans + _hiddenNeur[i]->evaluate(inputValue);
	}
	return ans;
}

template<class I, class O, class K>
O RBFNet<I, O, K>::evaluate(I inputValue, int k){
	O ans = 0;

	for (int i = 0; i < _hiddenLayerSize; i++){
		RBFNeuron<I, O, K>* curNeur = _hiddenNeur[i];
		ans = ans + curNeur->evaluate(inputValue, 0);
	}
	return ans;
}


template<class I, class O, class K>
float RBFNet<I, O, K>::test(vector<I*>* inVals, vector<O*>* outVals){
	unsigned int numOfTest = inVals->size();
	
	float sum = 0;

	if (numOfTest != outVals->size())
		return -1;

	O calcRes, trueRes;
	for (unsigned int i = 0; i < numOfTest; i++){
		calcRes = this->evaluate(*((*inVals)[i]));
		trueRes = *((*outVals)[i]);
		sum += _gaussFunc(calcRes, trueRes);
	}

	return (sum / numOfTest);
}

template<class I, class O, class K>
float RBFNet<I, O, K>::_gaussFunc(O evaluateResult, O expectingValue){
	return pow(2.71828f, -pow((expectingValue - evaluateResult) / (_errVal / 2),2));
}
//+++++++++++++++++++++++++++++++++++++++++++ GETTING RESULTS +++++++++++++++++++++++++++++++++++++++++++++++++
