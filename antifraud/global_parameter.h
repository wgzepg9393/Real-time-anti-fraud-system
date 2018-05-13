#ifndef GLOBAL_PARAMETER_H
#define GLOBAL_PARAMETER_H

const double fscale = 0.3; // ��֡ͼ�������ϵ��

const int NumData = 2240; //���õ����ݵ�����


const int DIM = 128; // ���ݵ�ά��

 //const int Num_Mix=2; // ���ģ�͵�����
const int Num_Mix = 6; // ���ģ�͵�����

//const int Min_samples = 2;            // This is the minimum number of samples in each cluster
const int Min_samples = 2;            // This is the minimum number of samples in each cluster

const float Min_weight = 0.02;        // This is the weight floor for k-mean
const int Max_Iter = 1000;            // This is the maximum number of iterations for k-mean
const float threshold = 0.0001;       // This is the stop criteria for k-mean
const float Infinity = 1e+32;
const float Perturb = 0.001;          // When a bad cell appears, replace its center by the center
									  // of the most popular cell plus this perturbation
const float threshold_EM = 0.0001;    // stop criteria for EM
const float factor = 10;
//const float face_threshold=94.91;             //������������ߵ���ֵ,С�ڸ���ֵʱ��ߣ����ڸ���ֵ������,��ֵԽ��©����ԽС
//const float face_threshold=94.896;             //������������ߵ���ֵ,С�ڸ���ֵʱ��ߣ����ڸ���ֵ������,��ֵԽ��©����ԽС
const float face_threshold = 94.883;             //������������ߵ���ֵ,С�ڸ���ֵʱ��ߣ����ڸ���ֵ������,��ֵԽ��©����ԽС


#endif // GLOBAL_PARAMETER_H

