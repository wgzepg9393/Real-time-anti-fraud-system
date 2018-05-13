#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include "io_data.h"
#include "global_parameter.h"
#include "opencv.h"
#include "dlib.h"
#include "gmm.h"
#include "mix.h"
#include "dlib_feature_extraction.h"
//#include "gmm.cpp"

using namespace std;
using namespace dlib;

std::vector<cv::Rect> faces_num; // 检测到地人脸的数量
int Dlib_non_face = 0;
int right_face_number = 0; // 识别人脸正确的数量
int face_number = 0; // 识别出的人脸数量
//GMM gmm;  // 构建一个gmm
///////////////////////////////////////////////////////////////////////////////////////////////////////
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;


frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;
anet_type net;

std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel>& img)
{
	// All this function is making 100 copies of img, all slightly jittered by being
	// zoomed, rotated, and translated a little bit differently.

	thread_local random_cropper cropper;
	cropper.set_chip_dims(150, 150);
	cropper.set_randomly_flip(true);
	cropper.set_max_object_size(0.99999);
	cropper.set_background_crops_fraction(0);
	cropper.set_min_object_size(0.97);
	cropper.set_translate_amount(0.02);
	cropper.set_max_rotation_degrees(3);

	std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
	raw_boxes[0] = shrink_rect(get_rect(img), 3);
	std::vector<matrix<rgb_pixel>> crops;

	matrix<rgb_pixel> temp;
	for (int i = 0; i < 100; ++i)
	{
		cropper(img, raw_boxes, temp, ignored_crop_boxes);
		crops.push_back(move(temp));
	}
	return crops;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////


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
double GMM::findMax(std::vector<double> vec) {
	double max = -9999999999999.0;
	for (auto v : vec) {
		if (max < v) max = v;
	}
	return max;
}

void GMM::E_step()   // This function computes all Pr[class i|data j]
{
	std::vector<double> z_ij;
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

GMM gmm;  // 构建一个gmm

///////////////////////////////////////////////
void build_gmm()
{
	//GMM gmm;  // 构建一个gmm
	char train_filename[] = "data2.txt"; // data.txt存放的是用于建模的数据

	gmm.Read_Feature(train_filename);// 从文件读特征
	gmm.Random_Init();             // 随机分配初始中心
	gmm.Iterate();                 // k-mean 迭代
	gmm.EM();                      // 执行EM算法
}


int main() // 处理视频，对视频进行检测
{
	//GMM gmm;  // 构建一个gmm
	cout << "正在构建模型..."<<endl;
	
	build_gmm();
	cout << "模型构建完成，开始处理视频..."<<endl;
	cout << "进行第一次关键帧抽取..." << endl;
	// 导入视频帧
	string video_path_str = "video001.mp4";
	cv::VideoCapture m(video_path_str);
	cv::Size outSize;
	cv::Mat img;
	cv::Mat test_img;
	// 对视频的帧大小，数目进行定义
	int frame_num = m.get(CV_CAP_PROP_FRAME_COUNT);
	int frame_height = m.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frame_width = m.get(CV_CAP_PROP_FRAME_WIDTH);

	string face_cascade_name = "haarcascade_frontalface_alt2.xml"; // 加载人脸检测器
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(face_cascade_name)) {
		printf("[error] no cascade\n");
	}
	// 导入提取人脸特征的网络
	std::vector<pair<cv::Point, cv::Point>> face_rectangle;
	matrix<rgb_pixel> Dlib_img;  // 将Mat格式的face0转换为Dlib格式的Dlib_img;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	string s; // 视频截出的帧的序号

	for (int j = 0; j<frame_num; j = j + 250)
	{
		feature_data.open("feature_data.txt", ios::app);

		m.set(CV_CAP_PROP_POS_FRAMES, j);
		m.read(img);

		s = to_string(j) + ".bmp";
		cv::cvtColor(img, img, CV_BGR2GRAY);

		if (frame_height>900 || frame_width>900)
		{
			outSize.width = img.cols*fscale;
			outSize.height = img.rows*fscale;
			cv::resize(img, img, outSize, 0, 0, CV_INTER_LINEAR);
		}
		imwrite("temp_img.bmp", img);
		test_img = cv::imread("temp_img.bmp");

		//face_cascade.detectMultiScale(test_img, faces_num, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(40, 100));
		face_cascade.detectMultiScale(test_img, faces_num, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
		cout << "第"<<j<<"帧的人脸个数为"<<faces_num.size() << endl;

		if (faces_num.size() == 1)
		{
			cv::Point centera(faces_num[0].x, faces_num[0].y);
			cv::Point centerb(faces_num[0].x + faces_num[0].width, faces_num[0].y + faces_num[0].height);
			face_rectangle.push_back(std::make_pair(centera, centerb));
			cv::rectangle(test_img, centera, centerb, cv::Scalar(0, 255, 0));

			//imwrite(s,test_img); //选择保存视频帧

			cv::Mat face_img = cv::imread("temp_img.bmp");
			dlib::assign_image(Dlib_img, cv_image<rgb_pixel>(face_img)); // 将mat格式的转为dlib格式的，提特征
			std::vector<matrix<rgb_pixel>> faces;
			for (auto face : detector(Dlib_img))
			{
				auto shape = sp(Dlib_img, face);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(Dlib_img, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(face_chip);
			}
			if (faces.size() == 0)
			{
				Dlib_non_face++; // 第二次抽取的关键帧，OpenCV检测出人脸，但是Dlib没有检测出人脸，这个认为是面具
				//result_data.open("result.txt", ios::app);
				//result_data<<"视频帧"<<s<<"没有检测到人脸"<<endl;
				//result_data.close();
				continue;
			}

			face_number++; // 第二次关键帧抽取的人脸数目
			std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
			feature_data << trans(face_descriptors[0]) << endl;
		}
		feature_data.close();
	}
	cout << "第二次关键帧抽取完成..." << endl;

	//ofstream sum_data;
	double sum = 0, temp;
	int row_num = 0, face_frame_num = 0;
	ifstream in1("feature_data.txt");
	ifstream in2("feature_data.txt");
	//ofstream discard_frame_data;
	ofstream third_key_frame_data;
	double feature_data1[DIM];
	double feature_data2[DIM];
	std::vector<double> all_feature_data;
	//discard_frame_data.open("discard_frame_data.txt",ios::app);
	third_key_frame_data.open("third_key_frame_data.txt", ios::app);

	char c;
	while (in2.get(c))
	{
		if (c == '\n')
			row_num++;
	}
	row_num = row_num + 1;
	face_frame_num = row_num / 2;

	while (in1.good() && !in1.eof())
	{
		in1 >> temp;
		all_feature_data.push_back(temp); // 将数据读入到all_feature_data中
	}
	in1.close();

	//////////////////////////////////第三次关键帧抽取算法///////////////////
	int k = 0;

	for (int n = 0; n<face_frame_num - 1;)
	{
		if (k == 0)
		{
			for (int i = n*DIM; i < (n + 1)*DIM; i++)
			{
				feature_data1[i - n*DIM] = all_feature_data[i];
				int j = i + 128;
				feature_data2[i - n*DIM] = all_feature_data[j];
				sum = abs(feature_data1[i - n*DIM] - feature_data2[i - n*DIM]) + sum;
			}
		}
		else {
			for (int i = n*DIM; i < (n + 1)*DIM; i++)
			{
				feature_data1[i - n*DIM] = all_feature_data[i];
				//int j = i + k*128;
				int j = i + (k + 1) * 128;

				feature_data2[i - n*DIM] = all_feature_data[j];
				sum = abs(feature_data1[i - n*DIM] - feature_data2[i - n*DIM]) + sum;
			}
		}

		if (sum>1)
		{
			//sum_data.open("sum_data.txt",ios::app);
			//sum_data<<sum<<endl;
			third_key_frame_data.open("third_key_frame_data.txt", ios::app);
			for (int j = n*DIM; j<(n + 1)*DIM; j++)
			{
				third_key_frame_data << all_feature_data[j] << " ";
			}
			n = n + k + 1;
			k = 0;
			third_key_frame_data << "\n" << "\n";
			third_key_frame_data.close();
			sum = 0;
		}
		else {
			k++;
			sum = 0;
		}
		//sum_data.close();
	}
	cout << "第三次关键帧抽取完成..." << endl;
	////////////////////////////////////////////////////////////////////////////////////////

	int anti_row_num = 0, third_face_frame_num = 0;
	double anti_temp;
	ifstream in("third_key_frame_data.txt");
	ifstream anti_in2("third_key_frame_data.txt");

	char anti_c;
	while (anti_in2.get(anti_c))
	{
		if (anti_c == '\n')
			anti_row_num++;
	}
	anti_row_num = anti_row_num + 1;
	third_face_frame_num = anti_row_num / 2;
	//cout<<"third_face_frame_num"<<third_face_frame_num;
	////////////////////////////////////////////////////
	//ofstream result_data;

	//double third_feature_data[DIM];

	std::vector<double> all_third_key_frame_data;
	while (in.good() && !in.eof())
	{
		in >> anti_temp;
		all_third_key_frame_data.push_back(anti_temp);
	}

	in.close();

	result_data.open("result.txt", ios::app);

	//ifstream read_path("video_path.txt");
	//string read_path_line;
	//std::vector<string> read_video_path_vec;
	//if (read_path) // 有该文件
	//{
		//while (getline(read_path, read_path_line))
		//{
			//read_video_path_vec.push_back(read_path_line);
		//}
	//}

	//string read_video_path_str1 = read_video_path_vec[0];
	result_data << video_path_str << "  的结果是：" << endl;

	if (third_face_frame_num == 0)
	{
		result_data << "视频中不含人脸，或该人戴了面具" << endl;
		result_data.close();
		right_face_number = 0;
		face_number = 0;
		third_face_frame_num = 0;
		Dlib_non_face = 0;
		
		string filename2 = "feature_data.txt";
		string filename3 = "third_key_frame_data.txt";

		
		fstream(filename2, ios::out);
		//fstream(filename3, ios::out); // 第三次关键帧清零
	}
	else {
		for (int n = 0; n < third_face_frame_num; n++)
		{
			//result_data.open("result.txt", ios::app);
			double third_feature_data[DIM];
			for (int i = n*DIM; i < (n + 1)*DIM; i++)
			{
				third_feature_data[i - n*DIM] = all_third_key_frame_data[i];
			}


			for (int j = 0; j < 128; ++j)
			{
				third_feature_data[j] = third_feature_data[j] * factor;
			}

			double result = gmm.GetProbability_sample(third_feature_data);

			result = (log(log(result) + 170) + 89.5);

			if (result < face_threshold)
			{
				result_data << result << "  有戴面具" << endl;
			}
			else
			{
				result_data << result << "  没有戴面具" << endl;
				right_face_number++;
			}

		}
		result_data << "第二次关键帧抽取后识别出的人脸数量" << "  " << face_number << endl;
		result_data << "第三次关键帧抽取后识别出的人脸数量" << "  " << third_face_frame_num << endl;
		//result_data <<"正确的数量"<<"  "<<right_face_number-Dlib_non_face<<endl;
		result_data << "正确的数量" << "  " << right_face_number << endl;

		//result_data <<"戴面具的概率"<<"  "<<(1-((double)(right_face_number-Dlib_non_face)/third_face_frame_num))*100<<"%"<<endl;
		result_data << "戴面具的概率" << "  " << (1 - ((double)(right_face_number) / third_face_frame_num)) * 100 << "%" << endl;

		result_data.close();
		right_face_number = 0;
		face_number = 0;
		third_face_frame_num = 0;
		Dlib_non_face = 0;
		
		string filename2 = "feature_data.txt";
		string filename3 = "third_key_frame_data.txt";

		
		fstream(filename2, ios::out);
		//fstream(filename3, ios::out);
		cout << "检测完成，请在result.txt中查看详细结果" << endl;
	}
}