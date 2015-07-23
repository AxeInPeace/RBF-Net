#include "RBFNet.h"
#include "RBFNeuron.h"
#include "RBFTrainer.h"
#include "GeneticLearn.h"
#include <vector>
#include <iostream>
#include <time.h>

using namespace std;

const float eNum = 2.7182f;

float gaussian(float inValue, vector<float> koefs){
	return koefs[2] * pow(eNum, -pow((inValue - koefs[0])/koefs[1], 2));
}

vector<float> gaussKoefGener(){
	vector<float> retVal;
	retVal.resize(3);
	retVal[0] = float((rand() % 400)) * 0.5f;
	retVal[1] = float(rand() % 50) * 0.1f + 1;
	retVal[2] = float(rand() % 200);
	
	return retVal;
}

int main(){
	srand(time(NULL));

	RBFTrainer<float, float, vector<float>> trainer;
	GeneticLearn<float, float, vector<float>> geneticMethod;
	RBFNet<float, float, vector<float>> net(1, 20, 1);

	net.initNeuronsWithFunc(gaussian);
	geneticMethod.initKoefGener(gaussKoefGener);
	net.initErrorValue(2);

	trainer.initLearnAgent(&geneticMethod);
	trainer.initNet(&net);

	vector<float*> inVals;
	vector<float*> outVals;
	inVals.resize(21);
	outVals.resize(21);

	for(int i = 0; i < 21; i++){
		inVals[i] = new float;
		*(inVals[i]) = float(i) * 10;
	}
	for(int i = 0; i < 21; i++){
		outVals[i] = new float;
		*(outVals[i]) = float(i) * 10;
	}

	vector<float*> inTestVals;
	vector<float*> outTestVals;
	inTestVals.resize(101);
	outTestVals.resize(101);

	for(int i = 0; i < 101; i++){
		inTestVals[i] = new float;
		*(inTestVals[i]) = float(i * 2);
	}
	for(int i = 0; i < 101; i++){
		outTestVals[i] = new float;
		*(outTestVals[i]) = float(i * 2);
	}

	net = *(trainer.train(inVals, outVals, inTestVals, outTestVals));

	/*
	for(int i = 0; i < 20; i++){
		cout << "m[" << i << "] = " << (net.getNeur(i)->getKoef())[0] << endl;		
	}

	for(int i = 0; i < 20; i++){
		cout << "d[" << i << "] = " << (net.getNeur(i)->getKoef())[1] << endl;
	}
	*/

	float testVal;
	for (int i = 0; i < 10; i++){
		testVal = (rand() % 2000) * 0.1f;
		cout << testVal << " ~ " << net.evaluate(testVal) << endl;
	}

	system("pause");

	return 0;
}