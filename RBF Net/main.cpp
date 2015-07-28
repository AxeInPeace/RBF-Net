#include "RBFNet.h"
#include "RBFNeuron.h"
#include "RBFTrainer.h"
#include "GeneticLearn.h"
#include <vector>
#include <iostream>
#include <time.h>


using namespace std;

const float eNum = 2.7182f;
const int numOfTests = 11;
const float lengthOfLine = 50.0f;

float gaussian(float inValue, vector<float> koefs){
	return koefs[2] * pow(eNum, -pow((inValue - koefs[0])/koefs[1], 2));
}

vector<float> gaussKoefGener(){
	vector<float> retVal;
	retVal.resize(3);
	retVal[0] = float((rand() % 120)) * 0.5f;
	retVal[1] = float(rand() % 50) * 0.1f + 5;
	retVal[2] = float(rand() % 100);
	return retVal;
}

float funcToAprox(float x){
	return x;
}

int main(){
	cout.precision(5);
	cout.setf(ios::fixed);
	RBFTrainer<float, float, vector<float>> trainer;
	GeneticLearn<float, float, vector<float>> geneticMethod;
	RBFNet<float, float, vector<float>> net(1, numOfTests, 1);

	net.initNeuronsWithFunc(gaussian);
	geneticMethod.initKoefGener(gaussKoefGener);
	net.initErrorValue(3);

	trainer.initLearnAgent(&geneticMethod);
	trainer.initNet(&net);

	vector<float*> inVals;
	vector<float*> outVals;
	inVals.resize(numOfTests);
	outVals.resize(numOfTests);

	for(int i = 0; i < numOfTests; i++){
		inVals[i] = new float;
		*(inVals[i]) = float(i) * lengthOfLine / (numOfTests - 1);
	}
	for(int i = 0; i < numOfTests; i++){
		outVals[i] = new float;
		*(outVals[i]) = funcToAprox(float(i) * lengthOfLine / (numOfTests - 1));
	}

	vector<float*> inTestVals;
	vector<float*> outTestVals;
	inTestVals.resize(50);
	outTestVals.resize(50);

	for(int i = 0; i < 50; i++){
		inTestVals[i] = new float;
		*(inTestVals[i]) = float(i);
	}
	for(int i = 0; i < 50; i++){
		outTestVals[i] = new float;
		*(outTestVals[i]) = funcToAprox(float(i));
	}

	net = *(trainer.train(inVals, outVals, inTestVals, outTestVals));
	
	vector<float> koefs;
	for(int i = 0; i < numOfTests; i++){
		koefs = net.getNeur(i)->getKoef();
		cout << "Neur[" << i << "] = " << koefs[2] << "gauss(" << koefs[0] << ", " << koefs[1] << ")" << endl;
	}

	float testVal;
	for (int i = 0; i < 10; i++){
		testVal = i * 10.0f;
		cout << testVal << " ~ " << net.evaluate(testVal) << endl;
	}

	do{
		cin >> testVal;
		cout << testVal << " ~ " << net.evaluate(testVal) << endl;
	} while (testVal != -1);


	system("pause");

	net.exportFile("MYNET.txt");

	return 0;
}