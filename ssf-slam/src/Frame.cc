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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "Tracking.h"
#include <thread>

// The previous image
std::vector<cv::Point2f> Prepoint,PrepointRmDynamic,PrepointS,Curpoint,CurpointRmDynamic,CurpointS;
std::vector<uchar> State;
std::vector<float> Err;
cv::Mat imGrayPre;
cv::Mat imDepthPre;
bool bPreFrameHavePotentialDynamicObj;
std::vector<cv::Rect_<float> > vPreFramePotentialDynamicBorder;
std::vector<std::vector<cv::Point2f>> dynamicConvexHull;


namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// For RGB-D
Frame::Frame(Tracking* pTracker, cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,Frame &F1)
    :mpTracker(pTracker),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ExtractORB(0,imGray);//compute mvKeys and mDescriptors

    //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //double tExtract= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    //std::cout << "ExtractORB time =" << tExtract*1000 <<  std::endl;

//    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    cv::Mat  imGrayT = imGray;
    cv::Mat  imDepthT = imDepth;
    if(imGrayPre.data && imDepthPre.data)
    {
        RmDynamicPointWithSemanticAndGeometry(imGrayPre,imGray,imDepth,imDepthPre);
        //RmDynamicPointWithSemanticAndSceneFlow(imGrayPre,imGray,imDepth,imDepthPre);
        std::swap(imGrayPre, imGrayT);
        std::swap(imDepthPre, imDepthT);
    }
    else
    {
        std::swap(imGrayPre, imGrayT);
        std::swap(imDepthPre, imDepthT);
    }
/*
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double tRmDynamic = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
    std::cout << "RmDynamicPoint time =" << tRmDynamic*1000 <<  std::endl;
*/

    N = mvKeys.size(); 

    if(mvKeys.empty()) return;
    UndistortKeyPoints();
    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }
    mb = mbf/fx;
    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}
int Frame::RmDynamicPointWithSemanticAndGeometry(const cv::Mat &imGrayPre, const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imDepthPre)
{
    //transform CurrentFrame's mvKeys to Currentpoint
    Curpoint.clear();
    Prepoint.clear();
    CurpointRmDynamic.clear();
    PrepointRmDynamic.clear();

    for(auto it = mvKeys.begin(); it != mvKeys.end(); ++it)
    {
        Curpoint.push_back(it->pt);
    }
    //double mom = nmatches;

    //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(imGray, imGrayPre, Curpoint, Prepoint, State, Err, cv::Size(21, 21), 3, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01));
    /*
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "calcOpticalFlow time =" << ttrack*1000 <<  std::endl;
    */
    //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    int Cur_keypoint_sum = Curpoint.size();
    int Pre_keypoint_sum;
    cv::Mat FundMat;
    int Cur_keypoint_sumS = mvKeys.size();
    if(bPreFrameHavePotentialDynamicObj)
    {
        for(auto itc = Curpoint.begin(), itp = Prepoint.begin(); itp != Prepoint.end(); ++itc,++itp)
        {
            if(!isInDynamicMask(*itp,vPreFramePotentialDynamicBorder, imDepthPre))
            {
                CurpointRmDynamic.push_back(*itc);
                PrepointRmDynamic.push_back(*itp);
            }
        }
        Pre_keypoint_sum = PrepointRmDynamic.size();
    }
    else 
    {
        Pre_keypoint_sum = Prepoint.size();
    }/*
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "obtain prior dyna p     time =" << ttrack*1000 <<  std::endl;
*/
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    if(Pre_keypoint_sum > 20 && bPreFrameHavePotentialDynamicObj)
        FundMat = cv::findFundamentalMat(CurpointRmDynamic, PrepointRmDynamic, cv::FM_RANSAC, 1.0, 0.99);
    else
        FundMat = cv::findFundamentalMat(Curpoint, Prepoint, cv::FM_RANSAC, 1.0, 0.99);

    int  offset = 0;
    auto it_cur = mvKeys.begin();
    auto it_pre = Prepoint.begin();
    

    //step1. 场景流计算。
    FundMat.convertTo(FundMat, mK.type());
    cv::Mat EssentialMat = mK.t()*FundMat*mK;

    cv::Mat R1, R2, t_;
    ComputeRt(EssentialMat, R1, R2, t_);

    cv::Mat R = cv::Mat();
    cv::Mat t = cv::Mat();
    CheckRt(R1, R2, t_, R, t);
    //std::cout << "R = " << R << std::endl;
    auto it_curRD = CurpointRmDynamic.begin();
    auto it_preRD = PrepointRmDynamic.begin();

    std::vector<vector<cv::Point2f>> depthClusters;

    while (it_curRD != CurpointRmDynamic.end()) {
        
        const float u = it_curRD->x;
        const float v = it_curRD->y;
        const float z = imDepth.at<float>(v, u);
        if(z > 0){
            bool belongsToExistingCluster = false;

            for (int i = 0; i < depthClusters.size(); ++i) {
                std::vector<cv::Point2f>& depthCluster = depthClusters[i];

                float avgDepth = ComputeAverageDepth(depthCluster, imDepth);
                //std::cout << "avgDepth " << avgDepth << std::endl;

                if (abs(z - avgDepth) < depth_threshold) {
                    depthCluster.push_back(*it_curRD);
                    belongsToExistingCluster = true;
                    //std::cout << "Added to cluster " << i << std::endl;
                    break;
                }
            }
            if (!belongsToExistingCluster) {
                vector<cv::Point2f> newDepthCluster;
                newDepthCluster.push_back(*it_curRD);
                depthClusters.push_back(newDepthCluster);
            }

            it_curRD++;
        }
        else{
            it_curRD++;
            //std::cout << "存在不合理的深度值 " << std::endl;
        }
    }

    std::vector<SceneFlowInfo> sceneFlowInfos;
    std::vector<SceneFlowInfo> sceneFlowInfosDivideDepth;
    for (int i = 0; i < depthClusters.size(); ++i) 
    {
        std::vector<cv::Point2f>& depthCluster_int = depthClusters[i];
        

        for (int j = 0; j < depthCluster_int.size(); ++j) {
            cv::Mat V;
            const float u = depthCluster_int[j].x;
            const float v = depthCluster_int[j].y;
            const float z = imDepth.at<float>(v, u);
            ComputeSceneFlow(depthCluster_int[j], R, t, imDepth, V);
            if (!V.empty()) {
                SceneFlowInfo sfInfo(depthCluster_int[j], V);
                sceneFlowInfos.push_back(sfInfo);

                cv::Mat VDivideDepth = V / z;
                SceneFlowInfo sfInfoDD(depthCluster_int[j], VDivideDepth);
                sceneFlowInfosDivideDepth.push_back(sfInfoDD);

            }
        }
    }

    std::vector<std::vector<cv::Point2f>> newDepthClusters;

    for (int i = 0; i < depthClusters.size(); ++i) //遍历每个深度聚类
    {
        std::vector<cv::Point2f>& depthCluster_int = depthClusters[i];//某个深度聚类中的每个点

        std::vector<Connection> connections;

        for (int j = 0; j < depthCluster_int.size(); ++j) {
            for (int k = j + 1; k < depthCluster_int.size(); ++k) {
                float distance = ComputeDistance(depthCluster_int[j], depthCluster_int[k]);

                if (distance < distance_threshold) {
                    cv::Mat V1, V2;
                    for (int m = 0; m < sceneFlowInfos.size(); ++m) {
                        const SceneFlowInfo& sfInfo = sceneFlowInfos[m];
                        if(sfInfo.keypoint == depthCluster_int[j]){
                            V1 = sfInfo.sceneFlow;
                            break;
                        }
                    }
                    for (int m = 0; m < sceneFlowInfos.size(); ++m) {
                        const SceneFlowInfo& sfInfo = sceneFlowInfos[m];
                        if(sfInfo.keypoint == depthCluster_int[k]){
                            V2 = sfInfo.sceneFlow;
                            break;
                        }
                    }
                    
                    if(!V1.empty() && !V2.empty()){
                        double acos = abs(V1.dot(V2) / (cv::norm(V1) * cv::norm(V2)));

                        if(acos >= connection_threshold){
                            connections.push_back({j, k, distance});
                        }
                    }
                }
            }
        }

        std::vector<std::vector<cv::Point2f>> ConnectCluster;
        ClusterConnectedPoints(depthCluster_int, connections, ConnectCluster);
        newDepthClusters.insert(newDepthClusters.end(), ConnectCluster.begin(), ConnectCluster.end());
    }
    depthClusters = newDepthClusters;

    cv::Mat averageSceneFlowDD = cv::Mat::zeros(3, 1, CV_32F);
    for (const SceneFlowInfo& sfInfo : sceneFlowInfosDivideDepth) {
        averageSceneFlowDD += sfInfo.sceneFlow;
    }
    averageSceneFlowDD /= sceneFlowInfosDivideDepth.size();

    mbHavePassiveDynamicObjectForRmDynamicFeature = false;
    

    for (int i = 0; i < depthClusters.size(); ++i) {
        //std::cout << "Cluster " << i << ":\n";
        const std::vector<cv::Point2f>& depthCluster = depthClusters[i];
        cv::Mat totalFlow = cv::Mat::zeros(3, 1, CV_32F);
        int count = 0;
        for (int j = 0; j < depthCluster.size();++j){
            const cv::Point2f& keypoint = depthCluster[j];

            for (int k = 0; k < sceneFlowInfos.size(); ++k) {
                const SceneFlowInfo& sfInfo = sceneFlowInfos[k];
                if (sfInfo.keypoint == keypoint) {
                    if(!sfInfo.sceneFlow.empty()){
                        totalFlow += sfInfo.sceneFlow;
                        count++;
                    }
                }
            }
        }

        if (count > 0) {
            totalFlow /= count;  
        }
        float avgDepth = ComputeAverageDepth(depthCluster, imDepth);

        cv::Mat VDivideDepth = totalFlow / avgDepth;
        
        double norm = cv::norm(averageSceneFlowDD) * cv::norm(VDivideDepth);

        if(norm != 0){
            double odistance = cv::norm(averageSceneFlowDD - VDivideDepth);
            if (odistance > 1.5*cv::norm(averageSceneFlowDD)) {
                mbHavePassiveDynamicObjectForRmDynamicFeature = true;
                std::vector<cv::Point2f> convexHull = ComputeConvexHull(depthCluster);
                dynamicConvexHull.push_back(convexHull);
            }
        }
    }/*
    for (const auto& hull : dynamicConvexHull) {
        std::cout << "Convex Hull Points:" << std::endl;
        for (const auto& point : hull) {
            std::cout << "x: " << point.x << ", y: " << point.y << std::endl;
        }
    }
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
    std::cout << "sceneflow  time =" << ttrack*1000 <<  std::endl;*/
    //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    while (!mpTracker->isDetectImageFinished()) 
    {
        usleep(1);
    }
    if(!mpTracker->mpDetector2d->mvObjects2D.empty()) 
    {
        mvObjects2D = mpTracker->mpDetector2d->mvObjects2D;
        mbHaveDynamicObjectForRmDynamicFeature = mpTracker->mpDetector2d->mbHaveDynamicObjectForRmDynamicFeature;
        mbHaveDynamicObjectForMapping = mpTracker->mpDetector2d->mbHaveDynamicObjectForMapping;
        //Record the dynamic information for previous frame
        bPreFrameHavePotentialDynamicObj = mbHaveDynamicObjectForRmDynamicFeature;     
    }
    else
        bPreFrameHavePotentialDynamicObj = false;
    
    if(mbHaveDynamicObjectForRmDynamicFeature)
    {
        mvPotentialDynamicBorderForRmDynamicFeature = mpTracker->mpDetector2d->mvPotentialDynamicBorderForRmDynamicFeature;
        //Record the dynamic information for previous frame
        vPreFramePotentialDynamicBorder = mvPotentialDynamicBorderForRmDynamicFeature;
    }
    if(mbHaveDynamicObjectForMapping)
        mvPotentialDynamicBorderForMapping = mpTracker->mpDetector2d->mvPotentialDynamicBorderForMapping;

    cv::Mat mDescriptors_Temp;
    std::vector<cv::KeyPoint> mvKeys_Temp = mvKeys;/*
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t1).count();
    std::cout << "waiting for detect  time =" << ttrack*1000 <<  std::endl;*/
    while (it_cur != mvKeys.end()) {
        //std::cout << "keypoint..." << std::endl;
        if (mbHaveDynamicObjectForRmDynamicFeature && isInDynamicMask(*it_cur,mvPotentialDynamicBorderForRmDynamicFeature, imDepth)||(mbHavePassiveDynamicObjectForRmDynamicFeature && PointInsideConvexHulls(*it_cur, dynamicConvexHull)))
        //if (mbHaveDynamicObjectForRmDynamicFeature && isInDynamicMask(*it_cur,mvPotentialDynamicBorderForRmDynamicFeature, imDepth))

        //if (mbHavePassiveDynamicObjectForRmDynamicFeature && PointInsideConvexHulls(*it_cur, dynamicConvexHull))
        {
            mvKeys.erase(it_cur);
            Cur_keypoint_sumS--;
        }
        else{
                mDescriptors_Temp.push_back(mDescriptors.row(offset));
                it_cur++; 

        }
        offset++;
    }
    mDescriptors = mDescriptors_Temp;

    mDescriptors = mDescriptors_Temp;
    if(mbHavePassiveDynamicObjectForRmDynamicFeature && Cur_keypoint_sum < mpORBextractorLeft->GetnFeatures()*0.1) //0.5
    {
        std::swap(mvKeys, mvKeys_Temp);
    }
    else
        std::swap(mDescriptors, mDescriptors_Temp);


    return Cur_keypoint_sum;
}
// 检查点是否在凸包内
bool Frame::PointInsideConvexHulls(const cv::KeyPoint& point, const std::vector<std::vector<cv::Point2f>>& allConvexHulls) {
    for (const auto& convexHull : allConvexHulls) {
        if (cv::pointPolygonTest(convexHull, point.pt, false) >= 0) {
            return true; 
        }
    }
    return false; 
}

std::vector<cv::Point2f> Frame::ComputeConvexHull(const std::vector<cv::Point2f>& clusterPoints) 
{
    std::vector<cv::Point2f> convexHull;
    cv::convexHull(clusterPoints, convexHull);

    std::vector<cv::Point2f> result;
    for (const auto& point : convexHull) {
        result.push_back(point); 
    }

    return result;
}
bool Frame::isInDynamicMask(const cv::Point2f &kp,const std::vector<cv::Rect_<float> >& vDynamicBorder_, const cv::Mat &imDepth)
{
    float x = kp.x;
    float y = kp.y;
    float z = imDepth.at<float>(y,x);
    for(unsigned int i = 0; i < vDynamicBorder_.size(); i++)
    {
        cv::Rect_<float> rect2d = vDynamicBorder_[i];

        if(x > rect2d.x && x < rect2d.x + rect2d.width && y > rect2d.y && y < rect2d.y + rect2d.height)
        {
            float dynamicCenterX = rect2d.x + rect2d.width / 2.0f;
            float dynamicCenterY = rect2d.y + rect2d.height / 2.0f;
            float dynamicCenterDepth = imDepth.at<float>(dynamicCenterY, dynamicCenterX);
            if (std::abs(z - dynamicCenterDepth) < 0.25)
            {
                return true;
            }
        }
    }
    return false;
}
bool Frame::isInDynamicMask(const cv::KeyPoint &kp,const std::vector<cv::Rect_<float> >& vDynamicBorder_, const cv::Mat &imDepth)
{
    float x = kp.pt.x;
    float y = kp.pt.y;
    float z = imDepth.at<float>(y,x);
    for(unsigned int i = 0; i < vDynamicBorder_.size(); i++)
    {
        cv::Rect_<float> rect2d = vDynamicBorder_[i];

        if(x > rect2d.x && x < rect2d.x + rect2d.width && y > rect2d.y && y < rect2d.y + rect2d.height)
        {
            float dynamicCenterX = rect2d.x + rect2d.width / 2.0f;
            float dynamicCenterY = rect2d.y + rect2d.height / 2.0f;
            float dynamicCenterDepth = imDepth.at<float>(dynamicCenterY, dynamicCenterX);
            if (std::abs(z - dynamicCenterDepth) < 0.25)
            {
                return true;
            }
        }
    }
    return false;
}
cv::Mat Frame::ComputeSceneFlow(cv::Point2f& keypoint, const cv::Mat& R, const cv::Mat& t, const cv::Mat &imDepth, cv::Mat &V) 
{
    auto it = std::find_if(CurpointRmDynamic.begin(), CurpointRmDynamic.end(), [&](const cv::Point2f& kp) {
        return kp == keypoint;
    });

    if (it != CurpointRmDynamic.end()) {
        int index = std::distance(CurpointRmDynamic.begin(), it);
        auto it_prec = PrepointRmDynamic.begin() + index;
        cv::Mat Ppre, Pz, Pe;

        ComputePz(*it_prec, imDepthPre, Ppre);
        //std::cout << "Ppre = " << Ppre << std::endl;
        if (!Ppre.empty()) {
            Pe = R * Ppre + t;
            ComputePz(keypoint, imDepth, Pz);
            //std::cout << "Pz = " << Pz << std::endl;
            if (!Pz.empty()) {
                V = Pe - Pz;
                //std::cout << "V = " << V << std::endl;
            }
            else{
                V = cv::Mat();
            }
        }
        else{
            V = cv::Mat();
        }
    }
    
    return V;
}
void Frame::ClusterConnectedPoints(const std::vector<cv::Point2f>& depthCluster, const std::vector<Connection>& Connection, std::vector<std::vector<cv::Point2f>>& ConnectCluster) {
    // 存储点的所属类别
    std::vector<int> pointLabels(depthCluster.size());

    for (int i = 0; i < depthCluster.size(); ++i) {
        pointLabels[i] = i;
    }
    for (const auto& connection : Connection) {

        int label1 = pointLabels[connection.point1];
        int label2 = pointLabels[connection.point2];

        if (label1 != label2) {
            int smallerLabel = std::min(label1, label2);
            int largerLabel = std::max(label1, label2);
            
            for (int i = 0; i < pointLabels.size(); ++i) {
                if (pointLabels[i] == largerLabel) {
                    pointLabels[i] = smallerLabel;
                }
            }
        }
    }

    std::map<int, std::vector<cv::Point2f>> clusteredPoints;

    for (int i = 0; i < pointLabels.size(); ++i) {
        clusteredPoints[pointLabels[i]].push_back(depthCluster[i]);
    }

    for (const auto& cluster : clusteredPoints) {
        ConnectCluster.push_back(cluster.second);
    }
}

float Frame::ComputeDistance(const cv::Point2f& kp1, const cv::Point2f& kp2) {
    float dx = kp1.x - kp2.x;
    float dy = kp1.y - kp2.y;
    return std::sqrt(dx * dx + dy * dy);
}


float Frame::ComputeAverageDepth(const vector<cv::Point2f>& depthCluster, const cv::Mat &imDepth) 
{
    float sumDepth = 0.0f;

    for (const auto& kp : depthCluster) {
        sumDepth += imDepth.at<float>(kp.y, kp.x);
    }

    float avgDepth = sumDepth / static_cast<float>(depthCluster.size());

    return avgDepth;
}
void Frame::ComputeRt(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{

    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0) 
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}
void Frame::CheckRt(cv::Mat &R1, cv::Mat &R2, cv::Mat &t_, cv::Mat &R, cv::Mat &t)
{
    cv::Mat t1=t_;
    cv::Mat t2=-t_;
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F); 
    cv::Mat P2_1 = (cv::Mat_<float>(3, 4)<<
        R1.at<float>(0, 0), R1.at<float>(0, 1), R1.at<float>(0, 2), t1.at<float>(0, 0),
        R1.at<float>(1, 0), R1.at<float>(1, 1), R1.at<float>(1, 2), t1.at<float>(1, 0),
        R1.at<float>(2, 0), R1.at<float>(2, 1), R1.at<float>(2, 2), t1.at<float>(2, 0));
    cv::Mat P2_2 = (cv::Mat_<float>(3, 4)<<
        R1.at<float>(0, 0), R1.at<float>(0, 1), R1.at<float>(0, 2), t2.at<float>(0, 0),
        R1.at<float>(1, 0), R1.at<float>(1, 1), R1.at<float>(1, 2), t2.at<float>(1, 0),
        R1.at<float>(2, 0), R1.at<float>(2, 1), R1.at<float>(2, 2), t2.at<float>(2, 0));
    cv::Mat P2_3 = (cv::Mat_<float>(3, 4)<<
        R2.at<float>(0, 0), R2.at<float>(0, 1), R2.at<float>(0, 2), t1.at<float>(0, 0),
        R2.at<float>(1, 0), R2.at<float>(1, 1), R2.at<float>(1, 2), t1.at<float>(1, 0),
        R2.at<float>(2, 0), R2.at<float>(2, 1), R2.at<float>(2, 2), t1.at<float>(2, 0));
    cv::Mat P2_4 = (cv::Mat_<float>(3, 4)<<
        R2.at<float>(0, 0), R2.at<float>(0, 1), R2.at<float>(0, 2), t2.at<float>(0, 0),
        R2.at<float>(1, 0), R2.at<float>(1, 1), R2.at<float>(1, 2), t2.at<float>(1, 0),
        R2.at<float>(2, 0), R2.at<float>(2, 1), R2.at<float>(2, 2), t2.at<float>(2, 0));

    std::vector<cv::Point2f>Curpoint_, Prepoint_;//匹配点的归一化坐标
    for(const auto& itc: Curpoint){
        Curpoint_.push_back(cv::Point2f((itc.x-mK.at<float>(0,2))/mK.at<float>(0,0),(itc.y-mK.at<float>(1,2))/mK.at<float>(1,1)));
    }
    for(const auto& itp: Prepoint){
        Prepoint_.push_back(cv::Point2f((itp.x-mK.at<float>(0,2))/mK.at<float>(0,0),(itp.y-mK.at<float>(1,2))/mK.at<float>(1,1)));
    }
    
    cv::Mat pts3d_1, pts3d_2, pts3d_3, pts3d_4;

    cv::triangulatePoints(P1, P2_1, Prepoint_, Curpoint_, pts3d_1);
    cv::triangulatePoints(P1, P2_2, Prepoint_, Curpoint_, pts3d_2);
    cv::triangulatePoints(P1, P2_3, Prepoint_, Curpoint_, pts3d_3);
    cv::triangulatePoints(P1, P2_4, Prepoint_, Curpoint_, pts3d_4);
    if (pts3d_1.empty()) {
        std::cout << "Error: triangulatePoints failed for P2_1" << std::endl;
    } else {
    }
    std::vector<cv::Point3f> point3D_1, point3D_2, point3D_3, point3D_4;
    
    for (int i = 0; i < pts3d_1.cols; i++) {
        cv::Mat x = pts3d_1.col(i).clone();
        x /= x.at<float>(3, 0);
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        point3D_1.push_back(p);
    }
    for (int i = 0; i < pts3d_2.cols; i++) {
        cv::Mat x = pts3d_2.col(i).clone();
        x /= x.at<float>(3, 0); 
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        point3D_2.push_back(p);
    }
    for (int i = 0; i < pts3d_3.cols; i++) {
        cv::Mat x = pts3d_3.col(i).clone();
        x /= x.at<float>(3, 0); 
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        point3D_3.push_back(p);
    }
    for (int i = 0; i < pts3d_4.cols; i++) {
        cv::Mat x = pts3d_4.col(i).clone();
        x /= x.at<float>(3, 0); 
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        point3D_4.push_back(p);
    }
    int positive_depth_count_1 = 0, positive_depth_count_2 = 0, positive_depth_count_3 = 0, positive_depth_count_4 = 0;
    for (const auto& pt : point3D_1) {
        if (pt.z > 0) {
            positive_depth_count_1++;
        }
    }
    for (const auto& pt : point3D_2) {
        if (pt.z > 0) {
            positive_depth_count_2++;
        }
    }
    for (const auto& pt : point3D_3) {
        if (pt.z > 0) {
            positive_depth_count_3++;
        }
    }
    for (const auto& pt : point3D_4) {
        if (pt.z > 0) {
            positive_depth_count_4++;
        }
    }
    R = cv::Mat();
    t = cv::Mat();

    int max_positive_depth_count = std::max({positive_depth_count_1, positive_depth_count_2, positive_depth_count_3, positive_depth_count_4});

    if (max_positive_depth_count == positive_depth_count_1) {
        R1.copyTo(R);
        t1.copyTo(t);
    } else if (max_positive_depth_count == positive_depth_count_2) {
        R1.copyTo(R);
        t2.copyTo(t);
    } else if (max_positive_depth_count == positive_depth_count_3) {
        R2.copyTo(R);
        t1.copyTo(t);
    } else if (max_positive_depth_count == positive_depth_count_4) {
        R2.copyTo(R);
        t2.copyTo(t);
    }

}

cv::Mat Frame::ComputePz(const cv::KeyPoint &kp, const cv::Mat &imDepth, cv::Mat &Pz)
{
	const float u = kp.pt.x;
    const float v = kp.pt.y;
    if (u < 0 || u >= imDepth.cols || v < 0 || v >= imDepth.rows) {
    std::cout << "uv = (" << u << "," << v << ")\n";
    
    }
    const float z = imDepth.at<float>(v,u);
    if(z>0)
    {
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;

        Pz = (cv::Mat_<float>(3,1) << x, y, z);
        return Pz;

    }
    else{
        Pz = cv::Mat();
        return Pz;
    }

}
cv::Mat Frame::ComputePz(const cv::Point2f &kp, const cv::Mat &imDepth, cv::Mat &Pz)
{
	const float u = kp.x;
    const float v = kp.y;
    if (u < 0 || u >= imDepth.cols || v < 0 || v >= imDepth.rows) {
    
    }
    const float z = imDepth.at<float>(v,u);
    if(z>0)
    {

        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Pz = (cv::Mat_<float>(3,1) << x, y, z);
        return Pz;
    }
    else{
        Pz = cv::Mat();
        return Pz;
    }
}


void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);
    
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;
        
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM