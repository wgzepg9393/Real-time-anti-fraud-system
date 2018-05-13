#ifndef MIX_H
#define MIX_H

#include <vector>
#include "global_parameter.h"

using namespace std;

struct Mix
{
	double Mean[DIM];     // mean vector
	double variance[DIM]; // variance vector
	double weight;        // weight
	vector <int> Sample_Index;  // This vector stores the indices of data points which belongs to this mixture
	int Total_Sample;    // This counts the total number of samples of this mixture
};

#endif // MIX_H
