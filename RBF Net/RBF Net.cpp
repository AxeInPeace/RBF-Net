#include "RBFNet.h"
#include "RBFNeuron.h"
#include "RBFTrainer.h"
#include "GeneticLearn.h"
#include <vector>
#include <iostream>

using namespace std;

const float eNum = 2.7182;

float gaussian(float inValue, vector<float> koefs){
	return pow(eNum, -pow((inValue - koefs[0])/koefs[1], 2));
}

vector<float> gaussKoefGener(){
	vector<float> retVal;
	retVal.resize(2);
	retVal[0] = float((rand() % 800) - 400) * 0.5f;
	retVal[1] = float(rand() % 50) * 0.02f;
	
	return retVal;
}



int main(){
	RBFTrainer<float, float, vector<float>> trainer;
	GeneticLearn<float, float, vector<float>> geneticMethod;
	RBFNet<float, float, vector<float>> net(1, 20, 1);

	net.initNeuronsWithFunc(gaussian);
	geneticMethod.initKoefGener(gaussKoefGener);

	trainer.initLearnAgent(&geneticMethod);
	trainer.initNet(&net);

	vector<float*> inVals;
	vector<float*> outVals;
	inVals.resize(20);
	outVals.resize(20);

	for(int i = 0; i < 21; i++){
		*(inVals[i]) = float(i - 10) * 20;
	}
	for(int i = 0; i < 20; i++){
		*(outVals[i]) = float(i - 10) * 20;
	}

	vector<float*> inTestVals;
	vector<float*> outTestVals;
	inTestVals.resize(100);
	outTestVals.resize(100);

	for(int i = 0; i < 100; i++){
		*(inTestVals[i]) = float(i * 4) - 200;
	}
	for(int i = 0; i < 100; i++){
		*(outTestVals[i]) = float(i * 4) - 200;
	}

	trainer.train(inVals, outVals, inTestVals, outTestVals);

	return 0;
}

