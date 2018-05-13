//建立高斯混合模型
/*
#include "gmm.h"

void GMM::Read_Feature(char *filename) // 这个函数从一个文件中读取特征数据，并且存到一个特征数组里面
{
	FILE *fin;//文件指针
	int i, j;
	if ((fin = fopen(filename, "r")) == NULL)  // file error
	{
		//QMessageBox::information(NULL, QObject::tr("错误"), QObject::tr("无法打开文件."));
		exit(0);
	}
	//for (i = 0; i<NumData; i++) // 读数据，并将数据保存到Feature里面,NumData=90
	for (i = 0; i<NumData; i++) // 读数据，并将数据保存到Feature里面,NumData=90
	{
		for (j = 0; j<DIM; j++)
		{
			fscanf(fin, "%f", &Feature[i][j]);
			Feature[i][j] = Feature[i][j] * factor; //这个实数factor会影响最后的结果，但是不知道为什么会影响
		}
	}
	fclose(fin);//要关闭文件
}

void GMM::Random_Init()  // Just use the first # of mixtures features to initialize the centers respectivly
{
	for (int i = 0; i<Num_Mix; i++)
		for (int j = 0; j<DIM; j++)
			Mixtures[i].Mean[j] = Feature[i][j]; // Initialize mean
}


void GMM::Recluster()    // This function recluster samples according to current means
{
	int i, closest_mix;

	for (i = 0; i<Num_Mix; i++)
	{
		Mixtures[i].Sample_Index.clear(); //Clear sample set for each cluster first
		Mixtures[i].Total_Sample = 0;
	}
	for (i = 0; i<NumData; i++)
	{
		closest_mix = find_closest_mixture(i);  //Find nearest mixture for each data point
		Mixtures[closest_mix].Sample_Index.push_back(i); // cluster this data into the nearest mixture
		Mixtures[closest_mix].Total_Sample++;
	}
}

int GMM::Check_Badcell()    // Look for a bad cell after reclustering
{
	int i;
	for (i = 0; i<Num_Mix; i++)
		if (Mixtures[i].Total_Sample < Min_samples)   // bad cell found
		{
			return (i); // return bad cell id
		}
	return (-1);     // otherwise, return -1 (not found)
}

void GMM::Adjust_Badmean(int bad_id)   // This function adjust the center of the bad cell
{
	int popular_id = 0;                // This is the cluster id with the most samples
	int feat_id;
	int i, j, k;
	float New_mean[DIM];               // Use a sample from the most popular cluster to replace the bad mean
	for (i = 1; i<Num_Mix; i++)        // Look for the most popular cluster
		if (Mixtures[i].Total_Sample>Mixtures[popular_id].Total_Sample)
			popular_id = i;
	if (Mixtures[popular_id].Total_Sample<Min_samples)    // Even the most popular cluster has fewer samples than the threshold
	{
		exit(0);
	}
	for (k = 0; k<DIM; k++)  // create a new mean for the bad cluster
		New_mean[k] = 0;
	for (j = 0; j<Mixtures[popular_id].Total_Sample; j++)   // compute the mean of the most popular cluster
	{
		feat_id = Mixtures[popular_id].Sample_Index[j];
		for (k = 0; k<DIM; k++)
			New_mean[k] = New_mean[k] + 1.0 / (Mixtures[popular_id].Total_Sample)*Feature[feat_id][k];
	}

	for (j = 0; j<DIM; j++)
		Mixtures[bad_id].Mean[j] = New_mean[j] + Perturb; // use the perturbed mean of the most popular cluster
}

int GMM::find_closest_mixture(int feature)  // This function finds the closest mixture to a pattern
{
	int i, mix_id;
	float minimum_distance = Infinity, d;

	for (i = 0; i<Num_Mix; i++)
	{
		d = Distance(feature, i);
		if (d<minimum_distance)
		{
			minimum_distance = d;
			mix_id = i;
		}
	}
	return mix_id;
}

float GMM::Distance(int pattern, int mix)     // Compute square 2-norm between a pattern and a mixture mean,计算欧式距离
{
	int i;
	float d = 0;
	for (i = 0; i<DIM; i++)
	{
		d += (Mixtures[mix].Mean[i] - Feature[pattern][i])*(Mixtures[mix].Mean[i] - Feature[pattern][i]);
	}
	return d;
}

float GMM::Update_Mean()     // This function update mean for all mixtures, and return the square change of mean over all mixtures
{
	int i, j, feat_idx, k;
	float change = 0, temp;
	float new_mean[DIM];
	for (i = 0; i<Num_Mix; i++)  // For each mixture, update mean
	{
		for (k = 0; k<DIM; k++) new_mean[k] = 0;       // clear new mean for each mixture
		for (j = 0; j<Mixtures[i].Total_Sample; j++)
		{
			feat_idx = Mixtures[i].Sample_Index[j];    // find the right feature
			for (k = 0; k<DIM; k++)
				new_mean[k] = new_mean[k] + 1.0F / (float)(Mixtures[i].Total_Sample)*Feature[feat_idx][k];  // compute feature mean
		}   //done with 1 mixture mean update
		temp = 0;
		for (k = 0; k<DIM; k++)
		{
			temp = temp + (new_mean[k] - Mixtures[i].Mean[k])*(new_mean[k] - Mixtures[i].Mean[k]);   // mean change in one mixture
			Mixtures[i].Mean[k] = new_mean[k];
		}
		change = change + temp;  // change is the total mean change over all clusters
	}
	return change;
}

void GMM::Update_variance()  // Compute variance(方差) of each mixture
{
	int i, j, k, feat_idx;
	for (i = 0; i<Num_Mix; i++)
	{
		for (k = 0; k<DIM; k++) Mixtures[i].variance[k] = 0;
		for (j = 0; j<Mixtures[i].Total_Sample; j++)
		{
			feat_idx = Mixtures[i].Sample_Index[j];
			for (k = 0; k<DIM; k++)
				Mixtures[i].variance[k] = Mixtures[i].variance[k] + 1.0F / (float)(Mixtures[i].Total_Sample)*(Feature[feat_idx][k] - Mixtures[i].Mean[k])*(Feature[feat_idx][k] - Mixtures[i].Mean[k]);
		}
	}
}

void GMM::Update_weight()
{
	int i;
	int counter = 0, max_id;
	float sum = 1.0, max_weight = -1;
	float temp[Num_Mix];
	for (i = 0; i<Num_Mix; i++)
		Mixtures[i].weight = (float)(Mixtures[i].Total_Sample) / (float)NumData; // update weight; NumData=90
	for (i = 0; i<Num_Mix; i++)   // look for the largest weight
		if (Mixtures[i].weight>max_weight)
		{
			max_weight = Mixtures[i].weight;
			max_id = i;
		}
	if (max_weight<Min_weight)
	{
		exit(0);
	}
	for (i = 0; i<Num_Mix; i++)
	{
		temp[i] = Mixtures[i].weight;   // temp stores weights before smoothing
		if (Mixtures[i].weight<Min_weight)
		{
			sum = sum - Mixtures[i].weight;
			Mixtures[i].weight = Min_weight;   // floor bad weights
			counter++;
		}
	}

	if (counter>0)   // discounting un-floored weights
	{
		for (i = 0; i<Num_Mix; i++)
			if (Mixtures[i].weight == temp[i])
				Mixtures[i].weight = (1.0F - (float)counter*Min_weight)*(Mixtures[i].weight) / sum;
	}
}


void GMM::Iterate()
{
	int iter = 1;      // this is the counter of iterations
	int bad_cell;
	float change = Infinity;
	while ((iter <= Max_Iter) && (change >= threshold))
	{
		Recluster();
		bad_cell = Check_Badcell();
		if (bad_cell >= 0)
		{
			Adjust_Badmean(bad_cell);
			Recluster();
		}
		change = Update_Mean(); // Update mean after reclustering
		iter++;
	}
	Update_variance();
	Update_weight();
}

float GMM::Compute_Gaussian(int mix_id, float *feature) // This function computes a gaussian for a mixture
{
	int k;  // k is dimension
	float sum = 0.0, product = 1.0, output;
	float two_pi = 8.0F*atan(1.0F);
	for (k = 0; k<DIM; k++)
	{
		sum = sum + (feature[k] - Mixtures[mix_id].Mean[k])*(feature[k] - Mixtures[mix_id].Mean[k]) / (Mixtures[mix_id].variance[k]);
		product = product*sqrt(Mixtures[mix_id].variance[k]);
	}
	sum = exp(-0.5*sum);
	product = 1.0F / product;
	output = 1.0F / pow(two_pi, (float)DIM / 2.0F)*product*sum;
	return output;
}

double GMM::Ln_Compute_Gaussian(int mix_id, float *feature)  // This function computes a gaussian for a mixture
{
	int k;  // k is dimension
			//double product = 1.0;
	double lnN = 0.0;
	double sum = 0.0, product = 0.0, output;
	double two_pi = 8.00000*atan(1.00000);  //就是2π
	for (k = 0; k<DIM; k++)
	{
		sum = sum + (feature[k] - Mixtures[mix_id].Mean[k])*(feature[k] - Mixtures[mix_id].Mean[k]) / (Mixtures[mix_id].variance[k]);
		//product = product*sqrt(Mixtures[mix_id].variance[k]);
		product = product + (-0.5)*log((Mixtures[mix_id].variance[k]));
	}
	//sum = exp(-0.5*sum);
	//product = 1.0000000 / product;
	//output = 1.000000 / pow(two_pi, (double)DIM / 2.0000000)*product*sum;
	lnN = (-1.0)*((double)DIM / 2.0)*log(two_pi) + product + (-0.5*sum);
	//return output;
	return lnN;
}

//这里用的c++ 11的写法
double GMM::findMax(vector<double> vec) {
	double max = -9999999999999.0;
	for (auto v : vec) {
		if (max < v) max = v;
	}
	return max;
}

void GMM::E_step()   // This function computes all Pr[class i|data j]
{
	vector<double> z_ij;
	int i, j;
	double denomenator;
	for (j = 0; j<NumData; j++)
	{
		denomenator = 0.000;
		for (i = 0; i<Num_Mix; i++)
		{
			double Ln_com_Gau = Ln_Compute_Gaussian(i, Feature[j]);
			double mix_weight = log((double)Mixtures[i].weight);
			E_Prob[i][j] = mix_weight + Ln_com_Gau;  // Pr[class i|data j]  // ln 的函数表示
			z_ij.push_back(mix_weight + Ln_com_Gau);

			denomenator = exp(E_Prob[i][j]) + denomenator;
		}

		//找出z_ij中最大的值
		double maxNumber = findMax(z_ij);
		//清空z_ij容器,也就是说z_ij里面一个元素都没有了
		z_ij.clear();
		denomenator = exp(((-1.0)*maxNumber))*denomenator;
		for (i = 0; i<Num_Mix; i++)
			E_Prob[i][j] = (exp(E_Prob[i][j])* exp((-1.0)*maxNumber)) / denomenator;
	}
}

float GMM::M_step()  // This function performs the M step
{
	int j, k, i;
	float sum1, sum2, sum3;
	float new_mean[DIM];
	float new_var[DIM];
	float new_weight[Num_Mix];
	float change_mean = 0.0F, change_var = 0.0F, change_weight = 0.0F, change;
	for (j = 0; j<Num_Mix; j++)
	{
		sum2 = 0.0;
		for (k = 0; k < NumData; k++)
		{
			sum2 = sum2 + E_Prob[j][k];
		}

		for (i = 0; i<DIM; i++)
		{
			sum1 = 0.0; sum3 = 0.0;
			for (k = 0; k < NumData; k++)
			{
				sum1 = sum1 + E_Prob[j][k] * Feature[k][i];
			}
			new_mean[i] = sum1 / sum2;	        // update mean
			change_mean = change_mean + fabs(new_mean[i] - Mixtures[j].Mean[i]); // compute change of mean
			Mixtures[j].Mean[i] = new_mean[i];

			for (k = 0; k < NumData; k++)
			{
				sum3 = sum3 + E_Prob[j][k] * (Feature[k][i] - new_mean[i])*(Feature[k][i] - new_mean[i]);
			}
			new_var[i] = sum3 / sum2;          // update variance
			change_var = change_var + fabs(new_var[i] - Mixtures[j].variance[i]); // compute change of variance
			Mixtures[j].variance[i] = new_var[i];
		}
		new_weight[j] = sum2 / (float)NumData; // update weight
		change_weight = change_weight + fabs(new_weight[j] - Mixtures[j].weight); // compute change of weight
		Mixtures[j].weight = new_weight[j];
	}
	change = change_mean + change_var + change_weight;   // total change of all parameters
	return change;
}

void GMM::EM()
{
	float change = Infinity;
	int iter = 1, i, j;
	while (change >= threshold_EM)
	{
		E_step();  // do E-step
		change = M_step(); // do M-step;
		iter++;
	} // done EM
}


double GMM::GetProbability_sample(const double* sample)
{
	double p = 0;
	for (int i = 0; i < Num_Mix; i++) //m_mixNum 是Gaussian数目,我们的项目中是2，因为是男女这两个类嘛
	{
		// 为了调试
		//其中m_priors[i]为Gaussian权重
		p += (Mixtures[i].weight) * GetProbability(sample, i);
	}
	return p;
}

//m_vars 是Gaussian方差，m_means是Gaussian均值。
double GMM::GetProbability(const double* x, int j) //j参数是表示第几个高斯
{
	//double p = 1.0;
	double lnp = (-0.5)*log(2 * 3.1415926);
	double mul_var = 1.0;
	double mul_exp_mi = 0.0;
	for (int d = 0; d < DIM; d++)  //DIM是样本维数，我们项目中是128维
	{
		//p *= 1 / sqrt(2 * 3.141592653589793238 * Mixtures[j].variance[d]);
		//p *= exp(-0.5 * (x[d] - Mixtures[j].Mean[d]) * (x[d] - Mixtures[j].Mean[d]) / Mixtures[j].variance[d]);
		mul_var = mul_var*Mixtures[j].variance[d];
		mul_exp_mi = mul_exp_mi + (-0.5 * (x[d] - Mixtures[j].Mean[d]) * (x[d] - Mixtures[j].Mean[d]) / Mixtures[j].variance[d]);
	}
	double total_lnp = lnp - (0.5)*log(mul_var) + mul_exp_mi;
	double FinalProbability = exp(total_lnp);
	return FinalProbability;
}
*/