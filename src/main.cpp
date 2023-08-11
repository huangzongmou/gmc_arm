#include "SparseOptFlow.h"
#include<memory>
#include<string>
// #include <iostream>
#include <chrono>



int main()
{


    std::string _gmc_method_name = "sparseOptFlow";
    std::string config_dir = "../config/";
    cv::Mat image1 = cv::imread("../img/234.jpg");
    cv::Mat image2 = cv::imread("../img/235.jpg");
    std::unique_ptr<SparseOptFlow_GMC> _gmc_algo = std::make_unique<SparseOptFlow_GMC>(config_dir);

    // auto start_time = std::chrono::high_resolution_clock::now();
    // for(uint16_t i=0; i<1000; i++)
    // {
    //     _gmc_algo->apply(image1);
    // }
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // std::cout << "运行时间: " << duration.count() << " 微秒" << std::endl;
    
    _gmc_algo->apply(image1);
    _gmc_algo->apply(image2);

    u_int32_t rows = image1.rows/2; // 图像的行数
    u_int32_t cols = image1.cols/2; // 图像的列数


    cv::Point center(cols,rows);
    cv::circle(image1, center, 60, (255, 0, 0), 3);
    cv::circle(image2, center, 60, (255, 0, 0), 3);
    std::cout<<rows<<","<<cols<<std::endl;
    _gmc_algo->Affine(cols,rows);
    std::cout<<rows<<","<<cols<<std::endl;
    center = cv::Point(cols,rows);
    cv::circle(image2, center, 60, (0, 0, 255), 3);
    cv::imwrite("test.jpg",image1);
    cv::imwrite("test1.jpg",image2);
    // Track::multi_gmc(tracks_pool, H);
    // Track::multi_gmc(unconfirmed_tracks, H);
    return 0;
}