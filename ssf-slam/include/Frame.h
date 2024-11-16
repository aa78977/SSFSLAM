/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include <unistd.h>
#include "Detector2D.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
class Tracking;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras. new fix
    Frame(Tracking* pTracker, cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,Frame &F1);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);//default version
    int RmDynamicPointWithSemanticAndGeometry(const cv::Mat &imGrayPre, const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imDepthPre);
    //bool CheckEpiLineDistToRmDynamicPoint(const cv::KeyPoint &kp1,const cv::Point2f &kp2, cv::Mat &F12
    //                                     ,const double threshold = 1.0);
    //bool isInDynamicRegion(const cv::Point2f  &kp,const std::vector<cv::Rect_<float> >& vDynamicBorder_);
    //bool isInDynamicRegion(const cv::KeyPoint &kp,const std::vector<cv::Rect_<float> >& vDynamicBorder_);
    bool isInDynamicMask(const cv::KeyPoint &kp,const std::vector<cv::Rect_<float> >& vDynamicBorder_, const cv::Mat &imDepth);
    bool isInDynamicMask(const cv::Point2f &kp,const std::vector<cv::Rect_<float> >& vDynamicBorder_, const cv::Mat &imDepth);
    int RmDynamicPointWithSemanticAndSceneFlow(const cv::Mat &imGrayPre, const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imDepthPre);
    //int FindBestMatch(const cv::Point2f& point, const std::vector<cv::KeyPoint>& mvKeys_Fir);
    void ComputeRt(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    void Normalize(const vector<cv::Point2f> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
    void CheckRt(cv::Mat &R1, cv::Mat &R2, cv::Mat &t_, cv::Mat &R, cv::Mat &t);
    cv::Mat ComputePz(const cv::KeyPoint &kp, const cv::Mat &imDepth, cv::Mat &Pz);
    cv::Mat ComputePz(const cv::Point2f &kp, const cv::Mat &imDepth, cv::Mat &Pz);
    float ComputeAverageDepth(const vector<cv::Point2f>& depthCluster, const cv::Mat &imDepth);
    struct Connection {
        int point1; // 关键点1在深度聚类中的索引
        int point2; // 关键点2在深度聚类中的索引
        float distance; // 关键点之间的距离
    };
    void ClusterConnectedPoints(const std::vector<cv::Point2f>& depthCluster, const std::vector<Connection>& Connection, std::vector<std::vector<cv::Point2f>>& ConnectCluster);
    float ComputeDistance(const cv::Point2f& kp1, const cv::Point2f& kp2);
    cv::Mat ComputeSceneFlow(cv::Point2f& keypoint, const cv::Mat& R, const cv::Mat& t, const cv::Mat &imDepth, cv::Mat &V);
    struct SceneFlowInfo {
        cv::Point2f keypoint;
        cv::Mat sceneFlow;
        SceneFlowInfo(cv::Point2f kp, cv::Mat sf) : keypoint(kp), sceneFlow(sf) {}
    };
    bool PointInsideConvexHulls(const cv::KeyPoint& point, const std::vector<std::vector<cv::Point2f>>& allConvexHulls);
    std::vector<cv::Point2f> ComputeConvexHull(const std::vector<cv::Point2f>& clusterPoints);


    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    Tracking* mpTracker;
    std::vector<Object2D> mvObjects2D;
    bool mbHaveDynamicObjectForMapping;
    bool mbHaveDynamicObjectForRmDynamicFeature;
    bool mbHavePassiveDynamicObjectForRmDynamicFeature;
    std::vector<cv::Rect_<float> > mvPotentialDynamicBorderForMapping;
    std::vector<cv::Rect_<float> > mvPotentialDynamicBorderForRmDynamicFeature;
    
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;
    //std::vector<cv::KeyPoint> mvKeys_Fir;


    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();




    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc

    float depth_threshold = 0.25;
    float distance_threshold = 30;
    float sceneflow_threshold = 1;
    float connection_threshold = 0.7;
    //float acos_threshold = 0.8;
    //float odistance_threshold = 0.7;

};

}// namespace ORB_SLAM

#endif // FRAME_H