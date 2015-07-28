#pragma once

#include <vector>
#include "RBFNet.h"
#include "VirtLearn.h"
#include "RBFNeuron.h"
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>


using Eigen::MatrixXf;
using namespace std;

typedef unsigned int uint;

const int POPUL_SIZE = 10;
const float GOOD_CRITERIA = 0.85f;


template <class I, class O, class K>
class GeneticLearn: public VirtLearn<I, O, K>
{
public:
	GeneticLearn();
	~GeneticLearn();

	virtual void learning(vector<I*>*, vector<O*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*) override;
	void initKoefGener(K(*koefGen)());

private:
	void _geneticFuncLearning(vector<I*>*, vector<O*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);

	vector<vector<K*>*>* _geneticStartPopulation();
	void _geneticCalcRulette(vector<vector<K*>*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	vector<vector<K*>*>* _geneticCrossover(vector<vector<K*>*>*, int);
	void _geneticMutation(vector<vector<K*>*>*);
	vector<vector<K*>*>* _geneticSurvival(vector<vector<K*>*>*, vector<vector<K*>*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	float _geneticFitnessFunc(vector<K*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	void _calcPopulationFitness(vector<vector<K*>*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*, vector<float>*);

	vector<float> _geneticSort(vector<vector<K*>*>*, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	int _compareUnits(vector<K*>*, vector<K*>*);

	void _weightLearning(vector<vector<K*>*>*, int, vector<I*>*, vector<O*>*, RBFNet<I, O, K>*);
	
	int _sizeOfGenome;
	int _sizeOfPopulation;	
	vector<float> _ruletteArray;	
	vector<MatrixXf*> _weightMtx;
	int _mutationChance;

	K (*_generateRandKoef)();

	void _matrixOutput(int rows, int cols, MatrixXf mtx);
};

template<class I, class O, class K> 
GeneticLearn<I, O, K>::GeneticLearn(){	
}

template<class I, class O, class K> 
GeneticLearn<I, O, K>::~GeneticLearn(){
	for(uint i = 0; i < _weightMtx.size(); i++){
		delete _weightMtx[i];
	}
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
	_weightMtx.resize(_sizeOfPopulation * 2);
	for(int i = 0; i < _sizeOfPopulation * 2; i++){
		_weightMtx[i] = new MatrixXf(_sizeOfGenome, 1);
	}
	_mutationChance = 5;

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
		//net->putWeightsInNeurons(_weightMtx[0]);
		bestSolution = net->test(testInVals, testOutVals);
		cout << "bestSolution = " << bestSolution << endl;
		
		iter++;
	} while (bestSolution < GOOD_CRITERIA && iter < 100);
	cout << endl;
	//_weightLearning(population, 0, inVals, outVals, net);
	//net->putWeightsInNeurons(_weightMtx[0]);

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
void GeneticLearn<I, O, K>::_weightLearning(vector<vector<K*>*>* koefs, int startPos, vector<I*>* inVals, vector<O*>* outVals, RBFNet<I, O, K>* net){
	MatrixXf netResults(_sizeOfGenome, _sizeOfGenome);

	for(int i = 0; i < 1; i++){
	//for(int i = startPos; i < startPos + (int)koefs->size(); i++){
		net->putKoefsInNerurons(koefs->at(i));

		RBFNeuron<I, O, K>* curNeur;
		for(int j = 0; j < _sizeOfGenome; j++){
			curNeur = net->getNeur(j);
			for (int k = 0; k < _sizeOfGenome; k++){
				netResults(k, j) = curNeur->evaluate(*inVals->at(k));
			}
		}		

		cout << "F(I) matrix" << endl;
		_matrixOutput(_sizeOfGenome, _sizeOfGenome, netResults);

		MatrixXf output(_sizeOfGenome, 1);
		for (int j = 0; j < _sizeOfGenome; j++){
			output(j, 0) = *(outVals->at(j));
		}

		cout << "O matrix" << endl;
		_matrixOutput(_sizeOfGenome, 1, output);

		Eigen::ConjugateGradient<MatrixXf> solver;
		solver.compute(netResults);
		*(_weightMtx[i]) = solver.solve(output);

		cout << "W matrix" << endl;
		_matrixOutput(_sizeOfGenome, 1, *(_weightMtx[i]));

		system("pause");
	}
}

template<class I, class O, class K> //complete
vector<vector<K*>*>* GeneticLearn<I, O, K>:: _geneticStartPopulation(){	
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

template <class I, class O, class K>// complete
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

template <class I, class O, class K> //complete
void GeneticLearn<I, O, K>::_geneticMutation(vector<vector<K*>*>* population){
	int counterOfEqualUnits = 1;

	for(uint i = 1; i < population->size(); i++){
		if(_compareUnits(population->at(0), population->at(i)) == 1)
			counterOfEqualUnits++;
	}

	_mutationChance = 100 * counterOfEqualUnits / population->size();
	if(_mutationChance < 5)
		_mutationChance = 5;
	cout << "mutationChance = " << _mutationChance << " ";

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

template <class I, class O, class K> //complete
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
	K* gene;
	for (uint i = popul1Size; i < newPopulSize; i++){
		 unit = (*newPopul)[i];
		 for (int j = 0; j < _sizeOfGenome; j++){
			gene = (*unit)[j];
			delete gene;
		 }
		 delete unit;
	}

	newPopul->resize(_sizeOfPopulation);
	return newPopul;
}

template <class I, class O, class K>
float GeneticLearn<I, O, K>::_geneticFitnessFunc(vector<K*>* koefsVec, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net){
	net->putKoefsInNerurons(koefsVec);	
	//net->putWeightsInNeurons(_weightMtx[i]);
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
				//swap(_weightMtx[i + jump], _weightMtx[i]);
                swapped = true;
            };
		};
	};	

	return fitness;
}
//+++++++++++++++++++++++++++++++++++++++++++ SORT POPULATION +++++++++++++++++++++++++++++++++++++++++++++++++