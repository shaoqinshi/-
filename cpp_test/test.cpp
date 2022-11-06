#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Jacobi"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
using namespace std;


int main()
{
    cv::Mat mat1 = cv::imread("/home/heht/cpp/project/cpp_test/1.png");
    Eigen::Vector3d v1(1,2,3);
    cout<<v1<<endl;
    cv::imshow("img",mat1);
    cv::waitKey(0);
    return 1;
}

