#include "SparseOptFlow.h"
#include "INIReader.h"

// Optical Flow
SparseOptFlow_GMC::SparseOptFlow_GMC(const std::string &config_dir) {
    _load_params_from_config(config_dir);
}

void SparseOptFlow_GMC::_load_params_from_config(const std::string &config_dir) {
    INIReader gmc_config(config_dir + "/gmc.ini");
    if (gmc_config.ParseError() < 0) {
        std::cout << "Can't load " << config_dir << "/gmc.ini" << std::endl;
        exit(1);
    }

    _useHarrisDetector = gmc_config.GetBoolean(_algo_name, "use_harris_detector", false);

    _maxCorners = gmc_config.GetInteger(_algo_name, "max_corners", 1000);
    _blockSize = gmc_config.GetInteger(_algo_name, "block_size", 3);
    _ransac_max_iters = gmc_config.GetInteger(_algo_name, "ransac_max_iters", 500);

    _qualityLevel = gmc_config.GetReal(_algo_name, "quality_level", 0.01);
    _k = gmc_config.GetReal(_algo_name, "k", 0.04);
    _minDistance = gmc_config.GetReal(_algo_name, "min_distance", 1.0);


    _downscale = gmc_config.GetFloat(_algo_name, "downscale", 2.0F);
    _inlier_ratio = gmc_config.GetFloat(_algo_name, "inlier_ratio", 0.5);
    _ransac_conf = gmc_config.GetFloat(_algo_name, "ransac_conf", 0.99);
}

void SparseOptFlow_GMC::apply(const cv::Mat &frame_raw) {
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;



    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);

    std::cout<<"frame.channels()"<<frame.channels()<<std::endl;
    // Downscale
    if (_downscale > 1.0F) {
        width /= _downscale, height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }


    // Detect keypoints
    std::vector<cv::Point2f> keypoints;
    cv::goodFeaturesToTrack(frame, keypoints, _maxCorners, _qualityLevel, _minDistance, cv::noArray(), _blockSize, _useHarrisDetector, _k);

    // std::cout<<keypoints.size()<<std::endl;
    if (!_first_frame_initialized || _prev_keypoints.size() == 0) {
        /**
         *  If this is the first frame, there is nothing to match
         *  Save the keypoints and descriptors, return identity matrix 
         */
        _first_frame_initialized = true;
        _prev_frame = frame.clone();
        _prev_keypoints = keypoints;
        return;
    }


    // Find correspondences between the previous and current frame
    std::vector<cv::Point2f> matched_keypoints;
    std::vector<uchar> status;
    std::vector<float> err;
    // std::cout<<_prev_keypoints.size()<<std::endl;

    
    try {
        cv::calcOpticalFlowPyrLK(_prev_frame, frame, _prev_keypoints, matched_keypoints, status, err);
    } catch (const cv::Exception &e) {
        std::cout << "Warning: Could not find correspondences for GMC" << std::endl;
        std::memcpy(Affine_H, Affine_init, sizeof(Affine_H));
        return;
    }


    // Keep good matches
    std::vector<cv::Point2f> prev_points, curr_points;
    // std::cout << matched_keypoints.size() <<std::endl;
    for (size_t i = 0; i < matched_keypoints.size(); i++) {
        if (status[i]) {
            prev_points.push_back(_prev_keypoints[i]);
            curr_points.push_back(matched_keypoints[i]);
        }
    }


    // Estimate affine matrix
    if (prev_points.size() > 4) {
        cv::Mat inliers;
        // cv::Mat homography = cv::findHomography(prev_points, curr_points, cv::RANSAC, 3, inliers, _ransac_max_iters, _ransac_conf);
        cv::Mat homography = cv::estimateAffinePartial2D(prev_points, curr_points, inliers,cv::RANSAC, 3, _ransac_max_iters, _ransac_conf, 10);

        double inlier_ratio = cv::countNonZero(inliers) / (double) inliers.rows;
        std::cout<<inlier_ratio<<std::endl;
        if (inlier_ratio > _inlier_ratio) {

            std::memcpy(Affine_H, homography.data, sizeof(Affine_H));
            
            if (_downscale > 1.0) {
                Affine_H[0][2] *= _downscale;
                Affine_H[1][2] *= _downscale;
            }
        } else {
            std::memcpy(Affine_H, Affine_init, sizeof(Affine_H));
            std::cout << "Warning: Could not estimate affine matrix" << std::endl;
        }
    }

    _prev_frame = frame.clone();
    _prev_keypoints = keypoints;

}

void SparseOptFlow_GMC::Affine(u_int32_t &x, u_int32_t &y)
{
    x = x * Affine_H[0][0] + Affine_H[0][2];
    y = y * Affine_H[1][1] + Affine_H[1][2];
}
