#include <numeric>
#include "matching2D.hpp"

using namespace std;

extern std::string g_detType;
extern int g_kptsNum;
extern float g_detTime_ms;
extern std::string g_descType;
extern float g_descTime_ms;
extern int g_matchedKpts;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if (descriptorType == "DES_BINARY")
        {
            normType = cv::NORM_HAMMING;
        }
        else
        {
            normType = cv::NORM_L2;
        }
        
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
        }

        if (descRef.type() != CV_32F)
        {
            // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector< std::vector<cv::DMatch> > knnMatches;

        matcher->knnMatch(descSource, descRef, knnMatches, 2); // Finds the best match for each descriptor in desc1

        // perform distance ratio filtration test
        const float ratioThresh = 0.8F;
        for (size_t i = 0; i != knnMatches.size(); i++)
        {
            if (knnMatches[i][0].distance < (ratioThresh * knnMatches[i][1].distance))
            {
                matches.push_back(knnMatches[i][0]);
            }
        }
    }

    std::cout << "Matched " << matches.size() << " keypoints" << std::endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool bUseOrientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, bUseOrientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB_UPRIGHT;

        extractor = cv::AKAZE::create(descriptor_type);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        ; // do nothing
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    g_descTime_ms = 1000 * t / 1.0;
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    const int blockSize{ 2 };
    const int apertureSize{ 3 };
    const double k{ 0.04 };

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat dstNorm;
    cv::Mat dstNormScaled;

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    const int minResponse{ 100 };    
    // perform non-maxima suppression
    for (size_t i = 0; i < dstNorm.rows; i++)
    {
        for (size_t j = 0; j < dstNorm.cols; j++)
        {
            const int response{ static_cast<int>(dstNorm.at<float>(i, j)) };

            if (response > minResponse)
            {
                cv::KeyPoint keypoint;
                keypoint.pt = cv::Point2f(j, i);  // j (col) -> x-axis, i (row) -> y-axis
                keypoint.response = response;
                keypoint.size = 2 * apertureSize;

                bool isOverlaped = false;
                for (auto itr = keypoints.begin(); itr != keypoints.end(); itr++)
                {
                    const float overlapArea{ cv::KeyPoint::overlap(keypoint, *itr)};
                    
                    // sensitive to any overlap with threshold zero
                    if (overlapArea > 0.0)
                    {
                        isOverlaped = true;

                        // if new keypoint has higher reponse/confidence than that stored once then replace with it.
                        if (keypoint.response > itr->response)
                        {
                            *itr = keypoint;
                            break; // TODO: check if it should be moved inside the above if condition
                        }
                    }
                }

                if (!isOverlaped)
                {
                    keypoints.push_back(keypoint);
                }
                         
            }  
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    g_detTime_ms = 1000 * t / 1.0;

    if (bVis)
    {
        // visualize the keypoints
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    g_detTime_ms = 1000 * t / 1.0;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    double t;

    if(detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNonMaximaSuppression = true;
        cv::FastFeatureDetector::DetectorType detType = cv::FastFeatureDetector::TYPE_9_16;

        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNonMaximaSuppression, detType);

        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        // cout << detectorType << " detection in " << 1000 * t / 1.0 << " ms" << " with " << keypoints.size() << " keypoints" << endl;
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        cv::Ptr<cv::BRISK> detector = cv::BRISK::create();

        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        // cout << detectorType << " detection in " << 1000 * t / 1.0 << " ms" << " with " << keypoints.size() << " keypoints" << endl;
    }
    else if (detectorType.compare("ORB") == 0)
    {
        int nFeatures = 30000;
        cv::Ptr<cv::ORB> detector = cv::ORB::create(nFeatures);
        
        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        // cout << detectorType << " detection in " << 1000 * t / 1.0 << " ms" << " with " << keypoints.size() << " keypoints" << endl;
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();

        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        // cout << detectorType << " detection in " << 1000 * t / 1.0 << " ms" << " with " << keypoints.size() << " keypoints" << endl;
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        int nFeatures = 10000;
        cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(nFeatures);

        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        // cout << detectorType << " detection in " << 1000 * t / 1.0 << " ms" << " with " << keypoints.size() << " keypoints" << endl;
    }
    else
    {
        ; // do nothing
    }
    
    g_detTime_ms = 1000 * t / 1.0;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "FAST Detector Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
}