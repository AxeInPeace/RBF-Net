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

