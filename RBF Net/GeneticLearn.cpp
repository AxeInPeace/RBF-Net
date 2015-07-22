/*
#include "RBFNeuron.h"
#include "RBFNet.h"


#define uint unsigned int

const int prostotak = 100;
const int POPUL_SIZE = 100;
const float GOOD_CRITERIA = 0.95;

template<class I, class O, class K> 
GeneticLearn<I, O, K>::GeneticLearn(){
	;
}


template<class I, class O, class K> 
GeneticLearn<I, O, K>::~GeneticLearn(){
	;
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
	_geneticFuncLearning(inVals, outVals, testInVals, testOutVals, net);
}

template<class I, class O, class K> 
void GeneticLearn<I, O, K>::_geneticFuncLearning(vector<I*>* inVals, vector<O*>* outVals, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net){
	vector<vector<K*>*>* population;	
	population = _geneticStartPopulation(); 
	vector<vector<K*>*> newPopulation;

	do{
		vector<float> ruletteArray;
		ruletteArray.resize(population.size());
		_geneticCalcRulette(&ruletteArray, &population, testInVals, testOutVals, net);

		newPopulation = _geneticCrossover(&ruletteArray, &population, POPUL_SIZE);
		_geneticMutation(&newPopulation);

		population = _geneticSurvival(&population, &newPopulation, testInVals, testOutVals);
		_putEveryKoefInNeur((*population)[0], net);
		
	} while (net->test(testInVals, testOutVals) < GOOD_CRITERIA);

	vector<K*>* unit;
	K* gene;
	for(int i = 0; i < _sizeOfPopulation; i++){
		unit = (*population)[i]
		for(int j = 0; j < _sizeOfGenome; j++){
			gene = (*unit)[j];
			delete gene;
		}
		delete unit;
	}
}

template<class I, class O, class K> 
void GeneticLearn<I, O, K>::_weightLearning(vector<I*>* inVals, vector<O*>* outVals){
	;
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

		for (uint i = 0; i < _sizeOfGenome; i++){
			koef = new K;
			(*koef) = _generateRandKoef();
			(*unit)[i] = koef;			
		}

		(*population)[j] = unit;
	}
	return population;
}

template <class I, class O, class K>// complete
void GeneticLearn<I, O, K>::_geneticCalcRulette(vector<float>* resultArray,vector<vector<K*>*>* population, vector<I*>* testInVals, vector<O*>* outTestVals, RBFNet<I, O, K>* net){
	
	float sum = 0, asum;

	for(int i = 0; i < population->size(); i++){
		_putEveryKoefInNeur((*population)[i], net);
		(*resultArray)[i] = net->test(testInVals, outTestVals);
		sum += (*resultArray)[i];		
	}

	asum = 1 / sum;

	(*resultArray)[0] *= asum;
	for(int i = 1; i < population->size(); i++){
		(*resultArray)[i] = (*resultArray)[i - 1] + (*resultArray)[i] * asum;
	}
}

template <class I, class O, class K>//almost complete (need to avoid numbers)
vector<vector<K*>*>* GeneticLearn<I, O, K>::_geneticCrossover(vector<float>* rouletteArray, vector<vector<K*>*>* population, int sizeOfNewPopul){	
	vector<vector<K*>*>* newPopul;
	newPopul->resize(sizeOfNewPopul);
	vector<K*>* newUnit;	
	
	for (int iter = 0; iter < sizeOfNewPopul; iter++){
		float firstParantRoulette = (rand() % 500) * 0.002; //ARGUMENTS
		float secondParantRoulette = (rand() % 500) * 0.002; //ARGUMENTS
		int firstParant, secondParant;

		for (int j = 0; j < _sizeOfPopulation; j++){
			if (firstParantRoulette < rouletteArray[j])
				firstParant = j;
			if (secondParantRoulette < rouletteArray[j])
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
vector<vector<K*>*>* GeneticLearn<I, O, K>::_geneticSurvival(vector<vector<K*>*>* popul1, vector<vector<K*>*>* popul2, vector<I*>* inVals, vector<O*>* outVals, RBFNet<I, O, K>* net){
	vector<vector<K*>*>* newPopul = new vector<vector<K*>*>;
	*newPopul = *popul1;

	uint popul1Size = popul1->size();
	uint popul2Size = popul2->size();
	uint newPopulSize = popul1Size + popul2Size;
	newPopul->resize(newPopulSize);	
	for(int i = 0; i < popul2Size; i++){
		(*newPopul)[popul1Size + i] = (*popul2)[i];
	}
	
	qsort(newPopul, newPopulSize, sizeof(*newPopul), [](const void* ptr1, const void* ptr2) -> int {
		_putEveryKoefInNeur(ptr1, net);
		float val1 = net->test(inVals, outVals);
		_putEveryKoefInNeur(ptr2, net);
		float val2 = net->test(inVals, outVals);
		(val1 > val2)?	return 1: return -1;
	});


	vector<K*>* unit;
	K* gene;
	for (int i = popul1Size; i < newPopulSize; i++){
		 unit = (*newPopul)[i];
		 for (int j = 0; j < _sizeOfGenome; j++){
			gene = (*unit)[i];
			delete gene;
		 }
		 delete unit;
	}

	newPopul->resize(_sizeOfPopulation);
	return newPopul;
}

template <class I, class O, class K>
float GeneticLearn<I, O, K>::_geneticFitnessFunc(vector<K*>* koefsVec, vector<I*>* testInVals, vector<O*>* testOutVals, RBFNet<I, O, K>* net){
	_putEveryKoefInNeur(koefsVec, net);
	return net->test(testInVals, testOutVals);
}

//+++++++++++++++++++++++++++++++++++++++++++ LEARN NET +++++++++++++++++++++++++++++++++++++++++++++++++

template <class I, class O, class K>
void GeneticLearn<I, O, K>::_putEveryKoefInNeur(vector<K*>* koefs, RBFNet<I, O, K>* net){
	RBFNeuron<I, O, K>* curNeuron;
	K* curKoef;

	for (int i = 0; i < _sizeOfGenome; i++){
		curNeur = net->getNeur[i];
		curKoef = (*koefs)[i];
		curNeuron->changeKoef(curKoef);
	}
}


*/