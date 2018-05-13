#ifndef GLOBAL_PARAMETER_H
#define GLOBAL_PARAMETER_H

const double fscale = 0.3; // 抽帧图像的缩放系数

const int NumData = 2240; //被用的数据的数量


const int DIM = 128; // 数据的维度

 //const int Num_Mix=2; // 混合模型的数量
const int Num_Mix = 6; // 混合模型的数量

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
//const float face_threshold=94.91;             //设置人脸和面具的阈值,小于该阈值时面具，大于该阈值是人脸,该值越大漏检率越小
//const float face_threshold=94.896;             //设置人脸和面具的阈值,小于该阈值时面具，大于该阈值是人脸,该值越大漏检率越小
const float face_threshold = 94.883;             //设置人脸和面具的阈值,小于该阈值时面具，大于该阈值是人脸,该值越大漏检率越小


#endif // GLOBAL_PARAMETER_H

