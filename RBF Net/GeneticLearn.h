#pragma once

#include <vector>
#include "RBFNet.h"
#include "VirtLearn.h"

using namespace std;

template <class I, class O, class K>
class GeneticLearn//: public VirtLearn<I, O, K>
{
public:
	GeneticLearn();
	~GeneticLearn();

	void learning(vector<I*>*, vector<O*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	void initKoefGener(K(*koefGen)());

private:
	void _geneticFuncLearning(vector<I*>*, vector<O*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	void _weightLearning(vector<I*>*, vector<O*>*);

	vector<vector<K*>*>* _geneticStartPopulation();
	void _geneticCalcRulette(vector<float>*,vector<vector<K*>*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	vector<vector<K*>*>* _geneticCrossover(vector<float>*, vector<vector<K*>*>*, int);
	vector<vector<K*>*>* _geneticSurvival(vector<vector<K*>*>*, vector<vector<K*>*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	float _geneticFitnessFunc(vector<K*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);

	void _putEveryKoefInNeur(vector<K*>*, RBFNet<I, O, K>*);
	

	int _sizeOfGenome;
	int _sizeOfPopulation;

	K (*_generateRandKoef)();
};
