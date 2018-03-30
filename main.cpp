#include "main.hpp"
#include <boost/concept_check.hpp>

using namespace std;
using namespace cv;

#define DEBUG_MODE 1 


/**
 * @brief Main function
 * 
 * @param argc ...
 * @param argv ...
 * @return int
 */
int main( int argc, char** argv )
{
    int ret = 0;
    
    cout << endl;
    cout << " usage: exec imag0 imag1" << endl;
    cout << " example: exec a.bmp b.bmp" << endl;
    cout << " Esc or q to quit" << endl;
    
    Mat model, image;
    if (argc < 2)
    {
      cout << "error! not enough argument!" << endl;
      ret = 1;
      return ret;
    } else
    {
      model = imread(argv[1]);
        if (model.empty())
        {
          cout << "error! model cannot retrieve" << endl;
          ret = 1;
          return ret;
        }
      
      image = imread(argv[2]);
        if (image.empty())
        {
          cout << "error! image cannot retrieve" << endl;
          ret = 1;
          return ret;
        }
       
    }
    
    imshow("model",model);
    waitKey(0);
    imshow("image",image);
    waitKey(0);  
    
    DOTObjectDetect dod;
    DOTparams mdl_dpr, img_dpr;
    mdl_dpr.k = 7; mdl_dpr.n0 = 36; mdl_dpr.Rsz = Size(10,10); mdl_dpr.t = 13.;
    img_dpr.k = 1; img_dpr.n0 = 36; img_dpr.Rsz = Size(10,10); img_dpr.t = 20.;
    dod.init(model,image,mdl_dpr,img_dpr);
    
    float total_time =0;
    clock_t start_time1 = clock();
    clock_t finish_time1;

    DOTresult4D dres;
    dod.detect(dres);
    
    finish_time1 = clock();
    total_time = (double)(finish_time1-start_time1)/CLOCKS_PER_SEC; 

    printf("Result is:\n");
    printf("Score: %.1f\n",dres.score);
    printf("Position: x=%.4f, y=%.4f, z=%.4f, s=%.2f\n",dres.pose.x,dres.pose.y,dres.pose.z,dres.pose.s);
    printf("Searching Time = %.2f ms\n",total_time*1000);
    
    dod.ScaledRotatedRectangle(image, Size(model.cols,model.rows), dres.pose, Scalar(255,0,255));
    imshow("Detection result",image);
    waitKey(0);
	
    return ret;
}


int DOTObjectDetect::ScaledRotatedRectangle(Mat& image, const Size& rec_sz, const Point4f& pose, const Scalar& color) {
  int ret = 0;
  
  float th = pose.z;
  float s = pose.s;
  
  Point2f A,B,C,D;       // A,B,C,D in upper-left corner coordinate system 
  Point2f A_,B_,C_,D_;   // A,B,C,D in rotating center coordinate system
  Point2f Ap,Bp,Cp,Dp;   // rotated A,B,C,D in rotating center cs
  Point2f App,Bpp,Cpp,Dpp; // rotated A,B,C,D in upper-left corner cs 
  
  A.x = pose.x; A.y = pose.y;
  B.x = pose.x +rec_sz.width*s;
  B.y = pose.y;
  C.x = pose.x +rec_sz.width*s; 
  C.y = pose.y +rec_sz.height*s;
  D.x = pose.x;
  D.y = pose.y +rec_sz.height*s;  
  
  Point2f rotC; // rotation center
  rotC.x = (A.x +C.x)/2;
  rotC.y = (A.y +C.y)/2;
  
  A_.x = rotC.x - A.x;
  A_.y = rotC.y - A.y;
  B_.x = rotC.x - B.x;
  B_.y = rotC.y - B.y;
  C_.x = rotC.x - C.x;
  C_.y = rotC.y - C.y;
  D_.x = rotC.x - D.x;
  D_.y = rotC.y - D.y;
  
  // rotation matrix
  // rotate by z-axis (x-y-z of the picture cs)
//   float r11 = cos(th); float r12 = -sin(th); float r21 = sin(th); float r22 = cos(th);
  // rotate by -z-axis (where -z-axis is the normal math cs's z-axis)
  float r11 = cos(-th); float r12 = -sin(-th); float r21 = sin(-th); float r22 = cos(-th);
   
  Ap.x = r11*A_.x +r12*A_.y; 
  Ap.y = r21*A_.x +r22*A_.y;
  Bp.x = r11*B_.x +r12*B_.y; 
  Bp.y = r21*B_.x +r22*B_.y;
  Cp.x = r11*C_.x +r12*C_.y; 
  Cp.y = r21*C_.x +r22*C_.y;
  Dp.x = r11*D_.x +r12*D_.y; 
  Dp.y = r21*D_.x +r22*D_.y;
  
  App.x = rotC.x +Ap.x;
  App.y = rotC.y +Ap.y;
  Bpp.x = rotC.x +Bp.x;
  Bpp.y = rotC.y +Bp.y;
  Cpp.x = rotC.x +Cp.x;
  Cpp.y = rotC.y +Cp.y;
  Dpp.x = rotC.x +Dp.x;
  Dpp.y = rotC.y +Dp.y;
  
  line(image,App,Bpp,color);
  line(image,Bpp,Cpp,color);
  line(image,Cpp,Dpp,color);
  line(image,Dpp,App,color);
  
  return ret;
}




int DOTObjectDetect::rotatedRectangle(Mat& image, const Size& rec_sz, const Point3f& pose, const Scalar& color) {
  int ret = 0;
  
  float th = pose.z;
  
  Point2f A,B,C,D;       // A,B,C,D in upper-left corner coordinate system 
  Point2f A_,B_,C_,D_;   // A,B,C,D in rotating center coordinate system
  Point2f Ap,Bp,Cp,Dp;   // rotated A,B,C,D in rotating center cs
  Point2f App,Bpp,Cpp,Dpp; // rotated A,B,C,D in upper-left corner cs 
  
  A.x = pose.x; A.y = pose.y;
  B.x = pose.x +rec_sz.width;
  B.y = pose.y;
  C.x = pose.x +rec_sz.width; 
  C.y = pose.y +rec_sz.height;
  D.x = pose.x;
  D.y = pose.y +rec_sz.height;  
  
  Point2f rotC; // rotation center
  rotC.x = (A.x +C.x)/2;
  rotC.y = (A.y +C.y)/2;
  
  A_.x = rotC.x - A.x;
  A_.y = rotC.y - A.y;
  B_.x = rotC.x - B.x;
  B_.y = rotC.y - B.y;
  C_.x = rotC.x - C.x;
  C_.y = rotC.y - C.y;
  D_.x = rotC.x - D.x;
  D_.y = rotC.y - D.y;
  
  // rotation matrix
  // rotate by z-axis (x-y-z of the picture cs)
//   float r11 = cos(th); float r12 = -sin(th); float r21 = sin(th); float r22 = cos(th);
  // rotate by -z-axis (where -z-axis is the normal math cs's z-axis)
  float r11 = cos(-th); float r12 = -sin(-th); float r21 = sin(-th); float r22 = cos(-th);
   
  Ap.x = r11*A_.x +r12*A_.y; 
  Ap.y = r21*A_.x +r22*A_.y;
  Bp.x = r11*B_.x +r12*B_.y; 
  Bp.y = r21*B_.x +r22*B_.y;
  Cp.x = r11*C_.x +r12*C_.y; 
  Cp.y = r21*C_.x +r22*C_.y;
  Dp.x = r11*D_.x +r12*D_.y; 
  Dp.y = r21*D_.x +r22*D_.y;
  
  App.x = rotC.x +Ap.x;
  App.y = rotC.y +Ap.y;
  Bpp.x = rotC.x +Bp.x;
  Bpp.y = rotC.y +Bp.y;
  Cpp.x = rotC.x +Cp.x;
  Cpp.y = rotC.y +Cp.y;
  Dpp.x = rotC.x +Dp.x;
  Dpp.y = rotC.y +Dp.y;
  
  line(image,App,Bpp,color);
  line(image,Bpp,Cpp,color);
  line(image,Cpp,Dpp,color);
  line(image,Dpp,App,color);
  
  return ret;
}


int DOTObjectDetect::detect(DOTresult4D& dres_)
{
  int ret = 0;
  
  dres_.score = 0;
  dres_.pose.x = 0;
  dres_.pose.y = 0;
  dres_.pose.z = 0;
  
  float th0, th1, dth;
  th0 = CV_PI/2.*3.; th1 = CV_PI*2.;
  dth = 10./180.*CV_PI;
  
  float s0, s1, ds;
  s0 = .9; s1 = 1.1;
  ds = .05;
  
  Mat model_,image_,model0; 
  model.copyTo(model0);
  Point2i rotC;
  rotC.x = model.cols/2; rotC.y = model.rows/2;
  
  Mat rotMat;
  
  DOTresult dres;
  for (float s = s0; s < s1; s = s+ds)
    for (float th = th0; th < th1; th = th+dth) {
    
      model0.copyTo(model_);
      image.copyTo(image_);
    
      // create rotation & scale matrix
      rotMat = getRotationMatrix2D(rotC, th/CV_PI*180., s);
      
      // perform transformation 
      warpAffine(model_,model_,rotMat,model_.size());
      
  //     imshow("transformed model",model_);
  //     waitKey(0);
      
      // feed transformed model and detect 2D
      init(model_,image,mdl_dpr,img_dpr);
      detect2D(dres);
      
      // collect if result improves
      if (dres.score > dres_.score) {
	dres_.pose.x = dres.pose.x;
	dres_.pose.y = dres.pose.y;
	dres_.pose.z = th;
	dres_.pose.s = s;
	dres_.score = dres.score;
	
	if (DEBUG_MODE){
	  printf("Result is:\n");
	  printf("Score: %.1f\n",dres_.score);
	  printf("Position: x=%.2f, y=%.2f, th=%.2f, s=%.2f\n",dres_.pose.x,dres_.pose.y,dres_.pose.z,dres_.pose.s);
	  
	  ScaledRotatedRectangle(image_, Size(model.cols,model.rows), dres_.pose, Scalar(255,0,255));
	  imshow("Detection result",image_);
	  waitKey(0);
	}
      }
  }

  return ret;
}



int DOTObjectDetect::detect3D(DOTresult3D& dres_)
{
  int ret = 0;
  
  dres_.score = 0;
  dres_.pose.x = 0;
  dres_.pose.y = 0;
  dres_.pose.z = 0;
  
  float th0, th1, dth;
  th0 = CV_PI/2.*3.; th1 = CV_PI*2.;
  dth = 10./180.*CV_PI;
  
  Mat model_,image_,model0; 
  model.copyTo(model0);
  Point2i rotC;
  rotC.x = model.cols/2; rotC.y = model.rows/2;
  
  Mat rotMat;
  float scale = 1.;
  
  DOTresult dres;
  for (float th = th0; th < th1; th = th+dth) {
    
    model0.copyTo(model_);
    image.copyTo(image_);
   
    // create rotation & scale matrix
    rotMat = getRotationMatrix2D(rotC, th/CV_PI*180., scale);
    
    // perform transformation 
    warpAffine(model_,model_,rotMat,model_.size());
    
//     imshow("transformed model",model_);
//     waitKey(0);
    
    // feed transformed model and detect 2D
    init(model_,image,mdl_dpr,img_dpr);
    detect2D(dres);
    
    // collect if result improves
    if (dres.score > dres_.score) {
      dres_.pose.x = dres.pose.x;
      dres_.pose.y = dres.pose.y;
      dres_.pose.z = th;
      dres_.score = dres.score;
      
      if (DEBUG_MODE){
	printf("Result is:\n");
	printf("Score: %.1f\n",dres_.score);
	printf("Position: x=%.2f, y=%.2f, th=%.2f\n",dres_.pose.x,dres_.pose.y,dres_.pose.z);
	
	rotatedRectangle(image_, Size(model.cols,model.rows), dres_.pose, Scalar(255,0,255));
	imshow("Detection result",image_);
	waitKey(0);
      }
    }
  }
  

  return ret;
}


/// draw the max gradients on the detected image 
int DOTObjectDetect::visualizeDOT(const Mat& image, const DOTparams& dpr, const DOTdata& ddat) {
  int ret = 0;
  
  Mat img;
  image.copyTo(img);
  
  float max_mag;
  
  for (int i=0; i<ddat.size.height; i++)
    for (int j=0; j<ddat.size.width; j++)
    {
      int idx = i * ddat.size.width + j;
//       printf("region (%d,%d) has size %d ||| ",i,j,ddat.DOTvec);
      
      max_mag = 0;
      for (int k=0; k<ddat.DOTvec[idx].size(); k++)
      {
	Point2i p;
	
	p.x = (int)ddat.DOTvec[idx][k].loc.x;
	p.y = (int)ddat.DOTvec[idx][k].loc.y;
	
	if (ddat.DOTvec[idx][k].mag > max_mag) {
	  max_mag = ddat.DOTvec[idx][k].mag;
	  printf("(%d,%d) mag %.2f |||",p.x,p.y,ddat.DOTvec[idx][k].mag);
	}
	circle(img,p,2,Scalar(255,0,255));
      }
      printf("\nMax mag is %.2f\n",max_mag);
    }
  
  imshow("the max gradients",img);
  waitKey(0);
  
  return ret;
}


int DOTObjectDetect::detect2D(DOTresult& dres_)
{
  int ret = 0;
  
  getEdgeData(model,mdl_edDat);
  getEdgeData(image,img_edDat);
  
  extractDOT(model,mdl_edDat,mdl_dpr,mdl_DOT);
  if (DEBUG_MODE)
    visualizeDOT(model,mdl_dpr,mdl_DOT);
  
  extractDOT(image,img_edDat,img_dpr,img_DOT);
  if (DEBUG_MODE)
    visualizeDOT(image,img_dpr,img_DOT);
  
  searchDOT(mdl_dpr,mdl_DOT, img_DOT, dres);
  
  dres_ = dres;

  return ret;
}



DOTObjectDetect::DOTObjectDetect()
{

}



DOTObjectDetect::~DOTObjectDetect()
{

}



int DOTObjectDetect::init(const cv::Mat& model_, const cv::Mat& image_, DOTparams mdl_dpr_, DOTparams img_dpr_)
{
  int ret = 0;
  
  model = model_;
  image = image_;
  
  mdl_dpr = mdl_dpr_;
  img_dpr = img_dpr_;
  
  return ret;
}



/// check whether "do" \in "DO" where "do" is a set of orientation (here 1 element), "DO" is another set of orientation (here 7 elements)
int DOTObjectDetect::in_(const DOTdata& img_DOT, int i, int j, const DOTdata& mdl_DOT, int k) {
  int ret = 0;
  
  Point2i DO_xy;  // coordinates of the region in model coordinate system
  Point2i do_xy;  // coordinates of the region in image coordinate system
  
  DO_xy.x = k % mdl_DOT.size.width;
  DO_xy.y = k / mdl_DOT.size.width;
  
  do_xy.x = DO_xy.x + j;
  do_xy.y = DO_xy.y + i;
  
  // 1D index of the region in image coordinate
  int idx = do_xy.y * img_DOT.size.width + do_xy.x;
  
  vector<MagOri> DO_, do_;
  DO_ = mdl_DOT.DOTvec[k];
  do_ = img_DOT.DOTvec[idx];
  
  for (int i=0; i<do_.size(); i++)
    for (int j=0; j<DO_.size(); j++)
      if (do_[i].ori == DO_[j].ori) {
	ret = 1;
	break;
      }
  
  return ret;
}



int DOTObjectDetect::searchDOT(const DOTparams& dpr, const DOTdata& mdl_DOT, const DOTdata& img_DOT, DOTresult& dres) {
  int ret = 0; 
  
  dres.score = 0;
  dres.pose.x = 0.;
  dres.pose.y = 0.;
  
  for (int i=0; i<img_DOT.size.height -mdl_DOT.size.height; i++)
    for (int j=0; j<img_DOT.size.width -mdl_DOT.size.width; j++)
    {
      float score = 0;
      
      for (int k=0; k<mdl_DOT.DOTvec.size(); k++)
       if (in_(img_DOT,i,j,mdl_DOT,k))
	 score++;
       
//        if (DEBUG_MODE) {
// 	printf("x=%d,y=%d score=%.1f\n",j*dpr.Rsz.width,i*dpr.Rsz.height,score);
// 	getchar();
//        }
	
      if (score > dres.score) {
	dres.score = score;
	dres.pose.x = j*dpr.Rsz.width;
	dres.pose.y = i*dpr.Rsz.height;  
      }
    }
  
  return ret;
}




int DOTObjectDetect::bag_sort(DOTparams dpr, vector<MagOri>& bag) {
  int ret = 0;
  
  MagOri tmp;
  for (int i=0; i<bag.size()-1; i++)
    if (bag[i].mag > bag.back().mag) {
      
      tmp = bag.back();
      bag.back() = bag[i];
      bag[i] = tmp;
    }
    
  if (bag.size() > dpr.k) 
    bag.erase(bag.begin(), bag.end()-dpr.k);
  
  if (DEBUG_MODE) {
//     for (int i=0;i<bag.size();i++)
//       printf("mag=%.2f, ori=%.2f\n",bag[i].mag,bag[i].ori);
  }
//   cout << endl;
//   getchar();
  
  return ret;
}


void DOTObjectDetect::printCheckAng(int x, int y, const Mat Sx, const Mat Sy, const Mat& mag, const Mat& ang){
  
  printf("Sx=%f, Sy=%f\n",Sx.at<float>(y,x),Sy.at<float>(y,x));
  // mag = sqrt(Sx^2+Sy^2)
  // ang = atan2(Sy,Sx)
  printf("mag=%f, ang=%f\n",mag.at<float>(y,x),ang.at<float>(y,x));
}


int DOTObjectDetect::getEdgeData(const Mat& image, EdgeData& edDat) {
    int ret = 0;
  
    Mat image_gr; 
    cvtColor(image,image_gr,CV_BGR2GRAY); 
    // **** filter

    GaussianBlur(image_gr, image_gr, Size(3,3), 0);
    
    Mat Sx, Sy;
    Sobel(image_gr, Sx, CV_16SC1, 1, 0, 3); 

    Sobel(image_gr, Sy, CV_16SC1, 0, 1, 3); 

    Sx.convertTo(Sx,CV_32FC1);
    Sy.convertTo(Sy,CV_32FC1);
    
    if (DEBUG_MODE){
      imshow("sb x", Sx);
      waitKey(0);
      // Sobel Y

      imshow("sb y", Sy);
      waitKey(0);
    }
    
    Mat mag, ang; 
    cartToPolar(Sx,Sy,mag,ang,true);

    image_gr.copyTo(edDat.gray);
    ang.copyTo(edDat.ang);
    mag.copyTo(edDat.mag);
    Sx.copyTo(edDat.Sx);
    Sy.copyTo(edDat.Sy);
    
    if (DEBUG_MODE) {
//       for (int i=0; i<Sx.rows; i++)
// 	for (int j=0; j<Sx.cols; j++)
// 	  if (Sx.at<float>(i,j) || Sy.at<float>(i,j) )
// 	{
// 	  printCheckAng(j,i,Sx,Sy,mag,ang);
// 	  getchar();
// 	}
    }

    return ret;
}



/**
 * @brief extract the Dominant Orientation Template, which is basically max k number of gradients 
 */
int DOTObjectDetect::extractDOT(const Mat& image, const EdgeData& image_edDat, DOTparams dpr, DOTdata& mdl_DOT) {
  int ret = 0;
  
  int x_num_steps = image.cols / dpr.Rsz.width;
  int y_num_steps = image.rows / dpr.Rsz.height;
  
  mdl_DOT.size = Size(x_num_steps,y_num_steps);
  
  mdl_DOT.DOTvec.clear(); 
  
  if (DEBUG_MODE) {
    printf("DOT size estimated = %d, %d\n",x_num_steps,y_num_steps);
  }
  
  vector<MagOri> bag;
  MagOri mo;
  
  float dth = 360. / (float)dpr.n0;
  
  
  for (int i=0; i<y_num_steps; i++) 
    for (int j=0; j<x_num_steps; j++)
    {
      // init a region
      bag.clear();
      
      for (int o=i*dpr.Rsz.height; o<(i+1)*dpr.Rsz.height; o++)
	for (int p=j*dpr.Rsz.width; p<(j+1)*dpr.Rsz.width; p++) {
	 
	  mo.mag = image_edDat.mag.at<float>(o,p);
	  mo.ori = ceil(image_edDat.ang.at<float>(o,p) / dth);    // ori = 0..n0-1
	  mo.loc.x = (float)p;
	  mo.loc.y = (float)o;
	  if (abs(mo.ori-36.00) < 0.001) mo.ori = 0;
	  
	  if (mo.mag < dpr.t) continue;
	  
	  bag.push_back(mo);     // don't allow bag size to grow bigger than dpr.k elements 
	  
	  bag_sort(dpr,bag);
	  
// 	  if (mo.ori) {
// 	    printf("ori = %.2f -> %.2f\n",image_edDat.ang.at<float>(o,p) / dth,mo.ori);
// 	    getchar();
// 	  }
	}
	
      // collect the region
      mdl_DOT.DOTvec.push_back(bag);  
    }
    
  if (DEBUG_MODE) {
//     printf("mdl_DOT size = %d,%d\n",mdl_DOT.size.height,mdl_DOT.size.width);
//     
//     for (int i=0; i<mdl_DOT.DOTvec.size(); i++)
//       printf("DOT %d have %d elements\n",i,mdl_DOT.DOTvec[i].size());
  }
  
  return ret;
}






