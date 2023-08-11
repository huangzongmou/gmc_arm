#pragma once
#include <string>
#include <opencv2/opencv.hpp>
class SparseOptFlow_GMC{
private:
    std::string _algo_name = "sparseOptFlow";
    float _downscale;

    bool _first_frame_initialized = false;
    cv::Mat _prev_frame;
    std::vector<cv::Point2f> _prev_keypoints;

    // Parameters
    int _maxCorners, _blockSize, _ransac_max_iters;
    double _qualityLevel, _k, _minDistance;
    double Affine_H[2][3] = {{1,0,0},{0,1,0}};
    double Affine_init[2][3] = {{1,0,0},{0,1,0}};
    bool _useHarrisDetector;
    float _inlier_ratio, _ransac_conf;


private:
    void _load_params_from_config(const std::string &config_dir);

public:
    explicit SparseOptFlow_GMC(const std::string &config_dir);
    void apply(const cv::Mat &frame_raw);
    void Affine(u_int32_t &x, u_int32_t &y);
};