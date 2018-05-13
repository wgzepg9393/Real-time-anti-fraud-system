#ifndef GMM_H
#define GMM_H

//#include <vector>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "mix.h"
#include "global_parameter.h"


using namespace std;

class GMM
{
public:
	GMM() {}                            // Constructor
	void Read_Feature(char *);          // Read in features from a file
	void Random_Init();                 // Randomly assign initial centers
	void Iterate();                     // Do k-mean iterations
	void EM();                          // Do EM

	double findMax(vector<double>);
	double GetProbability_sample(const double*);  //Input one sample,and get it's probability
	double GetProbability(const double*, int);

private:
	float Feature[NumData][DIM];          // Each row is a feature
	Mix Mixtures[Num_Mix];                // Each element is a mixture
	double E_Prob[Num_Mix][NumData];       // This matrix stores Pr(class i|data j) in E step
	void Recluster();                     // Recluster samples in k-mean
	int Check_Badcell();                  // Check if there is a mixture with less than Min_samples in k-mean
	void Adjust_Badmean(int);             // If there is a bad cell, add a new center in the most popular cluster
	float  Update_Mean();                 // Update centers after reclustering
	void Update_variance();               // update variances after k-mean
	void Update_weight();                 // update weights after k-mean
	float Distance(int, int);             // Compute L2-norm
	int find_closest_mixture(int);        // find closest mixture to a pattern in k-mean
	float Compute_Gaussian(int, float *); // Compute gaussian for a mixture
	double Ln_Compute_Gaussian(int, float *);
	void E_step();                        // This performs E-step in EM
	float M_step();                       // This performs M-step in EM
};



#endif // GMM_H
