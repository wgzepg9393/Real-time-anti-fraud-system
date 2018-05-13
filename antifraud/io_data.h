#ifndef IO_DATA_H
#define IO_DATA_H

//这是定义输入输出的变量
#include <fstream>
#include <iostream>
#include <string>

using namespace std;


ofstream feature_data; // 输出图片特征的.txt文件,每次要清空，保证只读取一个图像文件
ofstream result_data; // 输出结果文件

#endif // IO_DATA_H
