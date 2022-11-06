#include <iostream>
#include <chrono>    //计时模块，非核心代码
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
using namespace std;
using namespace Eigen;

struct myCostFun
{
    myCostFun(double x_,double y_):x(x_),y(y_){}

    template<typename T>
    bool operator()(const T* parm,T* residual)const
    {
        residual[0] = y - exp(parm[0] * x * x + parm[1] * x + parm[2]);
        return true;
    }

    static ceres::CostFunction* Create(const double xx,const double yy)
    {
        return (new ceres::AutoDiffCostFunction<
                myCostFun,1,3>(
                new myCostFun(xx,yy)));
    }

    double x,y;
};

void cereFun(vector<double> x, vector<double> y_, int inter,
             double *num)
{
    ceres::Problem problem;
    problem.AddParameterBlock(num,3);
    for(int i=0; i<x.size(); i++)
    {
        ceres::CostFunction *costFunction = myCostFun::Create(x[i],y_[i]);
        problem.AddResidualBlock(costFunction,NULL,num);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.5;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        cout << "vision only BA converge" << endl;
    }
    cout<<summary.BriefReport()<< endl;
}

void gauss_newton(vector<double> x, vector<double> y_, int inter,
                  double &ae,double &be,double &ce)
{
    for(int iter=0; iter<inter;iter++)
    {
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        double cost = 0, lastcost = 0;
        cout <<H<<endl;
        for(int i=0; i<x.size();i++)
        {
            double xi=x[i], yi=y_[i];
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J;
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

            H += J * J.transpose();
            b += -error * J;
            cost += error * error;
        }
        Vector3d dx = H.ldlt().solve(b);
        if(isnan(dx[0])){
            cout<<"result is nan"<<endl;
            break;
        }
        if (iter>90 && cost >= lastcost) {   //误差变大，找到最小值，退出迭代
            cout << "cost: " << cost << ">= last cost: " << lastcost << ", break." << endl;
            break;
        }
        double rate = 1.5;
        ae += rate*dx[0];
        be += rate*dx[1];
        ce += rate*dx[2];

        lastcost = cost;

        cout <<"iter: "<<iter<< "   total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
             "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }
}
int main() {
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double a  = 2.0, b  = -1.0, c = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    vector<double> x_data, y_data;      // 数据
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }
    int iteration = 100;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    gauss_newton(x_data,y_data,iteration,a,b,c);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout<<"time_used is "<<time_used.count()<<endl;
    cout<<"estimated is "<<a<<"\t"<<b<<"\t"<<c<<endl;

    double params[3] = {2.0,-1.0,5.0};
    cereFun(x_data,y_data,iteration,params);

    return 0;
}

