#pragma once

#include <vector>
#include "RBFNet.h"
#include "VirtLearn.h"
#include "RBFNeuron.h"
#include <algorithm>
#include <Eigen/Dense>

using Eigen::MatrixXf;
using namespace std;

typedef unsigned int uint;

const int POPUL_SIZE = 20;
const float GOOD_CRITERIA = 0.90f;
const int MUTATION_CHANCE = 20;
const int ITERATIONS = 5000;


template <class I, class O, class K>
class GeneticLearn: public VirtLearn<I, O, K>
{
public:
	GeneticLearn();
	~GeneticLearn();

	virtual void learning(vector<I*>*, vector<O*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*) override;
	void initKoefGener(K(*koefGen)());

private:
	void _geneticFuncLearning(vector<I*>* inValues, vector<O*>* outValues, vector<I*>* testInValues, vector<O*>* testOutValues, RBFNet<I, O, K>* net);
	vector<vector<K*>*>* _geneticStartPopulation();
	void _geneticCalcRulette(vector<vector<K*>*>* population, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net);
	vector<vector<K*>*>* _geneticCrossover(vector<vector<K*>*>* population, int newPopulationSize);
	void _geneticMutation(vector<vector<K*>*>* bornPopulation);
	vector<vector<K*>*>* _geneticSurvival(vector<vector<K*>*>* population1, vector<vector<K*>*>* population2, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net);

	float _geneticFitnessFunc(vector<K*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	void _calcPopulationFitness(vector<vector<K*>*>* population, vector<I*>* testInValues, vector<O*>* testOutValues, RBFNet<I, O, K>* net, vector<float>* fitnessResult);

	vector<float> _geneticSort(vector<vector<K*>*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	int _compareUnits(vector<K*>*, vector<K*>*);

	MatrixXf* _weightLearning(vector<K*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	
	int _sizeOfGenome;
	int _sizeOfPopulation;
	int _mutationChance;
	vector<float> _ruletteArray;

	K (*_generateRandKoef)();
	void _matrixOutput(int rows, int cols, MatrixXf mtx);
};

template<class I, class O, class K> 
GeneticLearn<I, O, K>::GeneticLearn(){	
}

template<class I, class O, class K> 
GeneticLearn<I, O, K>::~GeneticLearn(){
}

template<class I, class O, class K> 
void GeneticLearn<I, O, K>::initKoefGener(K(*koefGen)()){
	_generateRandKoef = koefGen;
}


//=========================================== LEARN NET =================================================
template<class I, class O, class K> 
void GeneticLearn<I, O, K>::learning(vector<I*>* inVals, vector<O*>* outVals, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net){
	_sizeOfGenome = net->getSize();
	_sizeOfPopulation = POPUL_SIZE;
	_ruletteArray.resize(_sizeOfPopulation);
	_mutationChance = 20;

	_geneticFuncLearning(inVals, outVals, testInVals, testOutVals, net);
}

template<class I, class O, class K> 
void GeneticLearn<I, O, K>::_geneticFuncLearning(vector<I*>* inVals, vector<O*>* outVals, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net){
	vector<vector<K*>*>* population;	
	population = _geneticStartPopulation(); 
	vector<vector<K*>*>* newPopulation;
	
	int iter = 0;
	float bestSolution = 0;
	do{	
		cout << "iter = " << iter << " ";
		_geneticCalcRulette(population, testInVals, testOutVals, net);

		newPopulation = _geneticCrossover(population, _sizeOfPopulation);
		_geneticMutation(newPopulation);						

		population = _geneticSurvival(population, newPopulation, testInVals, testOutVals, net);		

		net->putKoefsInNerurons(population->at(0));				
		bestSolution = net->test(testInVals, testOutVals);
		cout << "bestSolution = " << bestSolution << endl;		
		iter++;
	} while (bestSolution < GOOD_CRITERIA && iter < ITERATIONS);	

	vector<K*>* unit;
	K* gene;
	for(int i = 0; i < _sizeOfPopulation; i++){
		unit = (*population)[i];
		for(int j = 0; j < _sizeOfGenome; j++){
			gene = (*unit)[j];
			delete gene;
		}
		delete unit;
	}
}

template<class I, class O, class K> 
void GeneticLearn<I, O, K>::_matrixOutput(int rows, int cols, MatrixXf mtx){
	for (int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++)
			cout << mtx(i, j) << " ";
		cout << endl;
	}
}

template<class I, class O, class K> 
MatrixXf* GeneticLearn<I, O, K>::_weightLearning(vector<K*>* koefs, vector<I*>* inVals, vector<O*>* outVals, RBFNet<I, O, K>* net){	

	RBFNeuron<I, O, K>* curNeur;
	MatrixXf netResults(_sizeOfGenome, _sizeOfGenome);
	for(int j = 0; j < _sizeOfGenome; j++){
		curNeur = net->getNeur(j);
		for (int k = 0; k < _sizeOfGenome; k++){
			netResults(k, j) = curNeur->evaluate(*inVals->at(k));
		}
	}		
	
	MatrixXf output(_sizeOfGenome, 1);
	for (int j = 0; j < _sizeOfGenome; j++){
		output(j, 0) = *(outVals->at(j));
	}

	
	MatrixXf* ansMatrix = new MatrixXf(_sizeOfGenome, 1);
	(*ansMatrix) = netResults.colPivHouseholderQr().solve(output);

	return ansMatrix;
	
}

template<class I, class O, class K> 
vector<vector<K*>*>* GeneticLearn<I, O, K>::_geneticStartPopulation(){	
	K* koef;	
	vector<K*>* unit;	

	vector<vector<K*>*>* population = new vector<vector<K*>*>;
	population->resize(uint(_sizeOfPopulation));

	for (int j = 0; j < _sizeOfPopulation; j++){
		unit = new vector<K*>;
		unit->resize(_sizeOfGenome);

		for (int i = 0; i < _sizeOfGenome; i++){
			koef = new K;
			(*koef) = _generateRandKoef();
			(*unit)[i] = koef;			
		}

		(*population)[j] = unit;
	}
	return population;
}

template <class I, class O, class K>
void GeneticLearn<I, O, K>::_geneticCalcRulette(vector<vector<K*>*>* population, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net){
	
	float sum = 0, asum;

	_calcPopulationFitness(population, testInVals, testOutVals, net, &_ruletteArray); 
	
	for(uint i = 0; i < population->size(); i++){
		sum += _ruletteArray[i];		
	}

	asum = 1 / sum;

	_ruletteArray[0] *= asum;
	for(uint i = 1; i < population->size(); i++){
		_ruletteArray[i] = _ruletteArray[i - 1] + _ruletteArray[i] * asum;
	}
}

template <class I, class O, class K>
void GeneticLearn<I, O, K>::_calcPopulationFitness(vector<vector<K*>*>* population, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net, vector<float>* results){
	results->resize(population->size());

	for(uint i = 0; i < population->size(); i++){
		(*results)[i] = _geneticFitnessFunc(population->at(i), testInVals, testOutVals, net);
	}
}

template <class I, class O, class K>//almost complete (need to avoid numbers)
vector<vector<K*>*>* GeneticLearn<I, O, K>::_geneticCrossover(vector<vector<K*>*>* population, int sizeOfNewPopul){	
	vector<vector<K*>*>* newPopul = new vector<vector<K*>*>;
	newPopul->resize(sizeOfNewPopul);
	vector<K*>* newUnit;	
	
	for (int iter = 0; iter < sizeOfNewPopul; iter++){
		float firstParantRoulette = (rand() % 500) * 0.002f; //ARGUMENTS
		float secondParantRoulette = (rand() % 500) * 0.002f; //ARGUMENTS
		int firstParant = -1, secondParant = -1;

		for (int j = 0; j < _sizeOfPopulation; j++){
			if (firstParantRoulette < _ruletteArray[j] && firstParant == -1)
				firstParant = j;
			if (secondParantRoulette < _ruletteArray[j] && secondParant == -1)
				secondParant = j;
		}

		int crossoverGene = rand() % (_sizeOfGenome - 1);
		newUnit = new vector<K*>;
		newUnit->resize(_sizeOfGenome);

		vector<K*>* parantGenome;
		K parantGene;

		parantGenome = (*population)[firstParant];
		for(int i = 0; i < crossoverGene; i++){			
			parantGene = *((*parantGenome)[i]);
			(*newUnit)[i] = new K(parantGene);
		}

		parantGenome = (*population)[secondParant];
		for(int i = crossoverGene; i < _sizeOfGenome; i++){			
			parantGene = *((*parantGenome)[i]);
			(*newUnit)[i] = new K(parantGene);		
		}

		(*newPopul)[iter] = newUnit;
	}

	return newPopul;
}

template <class I, class O, class K> 
void GeneticLearn<I, O, K>::_geneticMutation(vector<vector<K*>*>* population){
	_mutationChance = MUTATION_CHANCE;

	for (int i = 0; i < _sizeOfPopulation; i++){
		int rollMutationDice = rand() % 100;
		if (rollMutationDice < _mutationChance){
			vector<K*>* unit = population->at(i);
			int rollGeneMutationDice = rand() % _sizeOfGenome;			
			K* gene = unit->at(rollGeneMutationDice);
			*gene = _generateRandKoef();			
		}
	}
}

template <class I, class O, class K>
vector<vector<K*>*>* GeneticLearn<I, O, K>::_geneticSurvival(vector<vector<K*>*>* popul1, vector<vector<K*>*>* popul2, vector<I*>* inVals, vector<O*>* outVals, RBFNet<I, O, K>* net){
	vector<vector<K*>*>* newPopul = new vector<vector<K*>*>;
	*newPopul = *popul1;

	uint popul1Size = popul1->size();
	uint popul2Size = popul2->size();
	uint newPopulSize = popul1Size + popul2Size;
	newPopul->resize(newPopulSize);	
	for(uint i = 0; i < popul2Size; i++){
		(*newPopul)[popul1Size + i] = (*popul2)[i];
	}

	vector<float> fitnessValue;
	fitnessValue.resize(newPopulSize);
	fitnessValue = _geneticSort(newPopul, inVals, outVals, net);
	
	vector<K*>* unit;
	vector<vector<K*>*>* resultPopul = new vector<vector<K*>*>;
	resultPopul->resize(popul1Size);
	K* gene;
	uint breedCounter = 0; //count of units having same fitness function
	uint maxCounterValue = newPopulSize / 4;
	uint curPosResultPopul = 1;	

	(*resultPopul)[0] = (*newPopul)[0];
	for (uint i = 1; i < newPopulSize; i++){
		unit = (*newPopul)[i];
		if (fitnessValue[i] == fitnessValue[i - 1])
			breedCounter++;
		else
			breedCounter = 0;


		if (breedCounter >= maxCounterValue && (newPopulSize - i) > (popul1Size - curPosResultPopul)) {					
			for (int j = 0; j < _sizeOfGenome; j++){
				gene = (*unit)[j];
				delete gene;
			}
			delete unit;
		}
		else {
			if (curPosResultPopul < popul1Size){
				(*resultPopul)[curPosResultPopul] = unit;
				curPosResultPopul++;
			}
			else {
				for (int j = 0; j < _sizeOfGenome; j++){
					gene = (*unit)[j];
					delete gene;
				}
				delete unit;
			}
		}
	}	

	return resultPopul;
}

template <class I, class O, class K>
float GeneticLearn<I, O, K>::_geneticFitnessFunc(vector<K*>* koefsVec, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net){
	net->putKoefsInNerurons(koefsVec);		
	return net->test(testInVals, testOutVals);
}

//+++++++++++++++++++++++++++++++++++++++++++ LEARN NET +++++++++++++++++++++++++++++++++++++++++++++++++




//=========================================== SORT POPULATION =================================================
template <class I, class O, class K>
int GeneticLearn<I, O, K>::_compareUnits(vector<K*>* ptr1, vector<K*>* ptr2){ //complete
	if (ptr1->size() != ptr2->size() || ptr1->size() != _sizeOfGenome)
		return -1;
	
	for(int i = 0; i < _sizeOfGenome; i++)
		if (*(ptr1->at(i)) != *(ptr2->at(i)))
			return 0;
	
	return 1;
}

template <class I, class O, class K>
vector<float> GeneticLearn<I, O, K>::_geneticSort(vector<vector<K*>*>* population, vector<I*>* inVals, vector<O*>* outVals, RBFNet<I, O, K>* net) { //complete
	vector<float> fitness;	
	_calcPopulationFitness(population, inVals, outVals, net, &fitness);

	bool swapped = true;
	uint jump = population->size();
	while (jump > 1 || swapped){
		if (jump > 1)
			jump /= 1.24733;
		swapped = false;
		for (uint i = 0; i + jump < population->size(); i++){
			if (fitness[i + jump] > fitness[i]){
				swap(fitness[i + jump], fitness[i]);
				swap((*population)[i + jump], (*population)[i]);		
                swapped = true;
            };
		};
	};	

	return fitness;
}
//+++++++++++++++++++++++++++++++++++++++++++ SORT POPULATION +++++++++++++++++++++++++++++++++++++++++++++++++