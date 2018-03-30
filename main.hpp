#ifndef __MAIN_HPP__
#define __MAIN_HPP__

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <list>
#include <time.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


/// x,y: translation; z:rotation(theta-radian); s:scale
struct Point4f  {
  float x,y,z,s;
};

/// Magnitude and Orientation struct for DOT 
struct MagOri {
  float mag;
  float ori;  
  cv::Point2f loc;
};

struct EdgeData {
  cv::Mat gray;
  cv::Mat Sx;
  cv::Mat Sy;
  cv::Mat mag;
  cv::Mat ang;
};

struct DOTparams {
  int k;     // max k number of gradients
  int n0;    // number of bin for angle digitization
  cv::Size Rsz;  // region size
  float t;   // min mag threshold
};

struct DOTdata {
  std::vector<std::vector<MagOri> > DOTvec;
  cv::Size size;
};

struct DOTresult {
  float score;
  cv::Point2f pose;
};

struct DOTresult3D {
  float score;
  cv::Point3f pose;
};

struct DOTresult4D {
  float score;
  Point4f pose;
};


class DOTObjectDetect {
  
public:
  
  DOTObjectDetect();
  
  ~DOTObjectDetect();
  
  int init(const cv::Mat& model_, const cv::Mat& image_, DOTparams mdl_dpr_, DOTparams img_dpr_);
  
  /// detect in 2d, 2 dimensions of translation
  int detect2D(DOTresult& dres_);
  
  /// detect in 3d, 3rd dimension is rotation
  int detect3D(DOTresult3D& dres_);
  
  /// detect in 4d, 4th dimension is scaling 
  int detect(DOTresult4D& dres_);
  
  /**
   * @brief     draw a rotated rectangle with the rotation center is at the center of the rectangle, the rotating axis is math xyz cs, not picture xyz cs
   * @param[in] rec_sz size of the rectangle to be drawn
   * @param[in] pose   (pose.x,pose.y) is the upper-left corner of the rectangle before it is rotated; pose.z is the rotating angle
   */
  int rotatedRectangle(cv::Mat& image, const cv::Size& rec_sz, const cv::Point3f& pose, const cv::Scalar& color);
  
  /**
   * @brief similar to rotatedRectangle, but add a scaling 
   */
  int ScaledRotatedRectangle(cv::Mat& image, const cv::Size& rec_sz, const Point4f& pose, const cv::Scalar& color); 


private:
  
  cv::Mat model, image;
  EdgeData mdl_edDat, img_edDat;
  DOTdata mdl_DOT, img_DOT;
  DOTparams mdl_dpr, img_dpr;
  DOTresult dres;
  
  void printCheckAng(int x, int y, const cv::Mat Sx, const cv::Mat Sy, const cv::Mat& mag, const cv::Mat& ang);

  int getEdgeData(const cv::Mat& image, EdgeData& edDat);
  
  /**
   * @brief extract the Dominant Orientation Template, which is basically max k number of gradients 
   */
  int extractDOT(const cv::Mat& image, const EdgeData& image_edDat, DOTparams dpr, DOTdata& mdl_DOT);
  
  /// log(N) complexity
  int bag_sort(DOTparams dpr, std::vector<MagOri>& bag); 

  /// performs the search of model throughout the template and returns the max score and pose
  int searchDOT(const DOTparams& dpr, const DOTdata& mdl_DOT, const DOTdata& img_DOT, DOTresult& dres);
  
  /// check whether "do" \in "DO" where "do" is a set of orientation (here 1 element), "DO" is another set of orientation (here 7 elements)
  int in_(const DOTdata& img_DOT, int i, int j, const DOTdata& mdl_DOT, int k);
  
  /// draw the max gradients on the detected image 
  int visualizeDOT(const cv::Mat& image, const DOTparams& dpr, const DOTdata& ddat);
};

#endif