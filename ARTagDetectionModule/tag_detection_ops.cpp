
#include "tag_detection_ops.h"

#include "opencv2\core\core.hpp"
#include "opencv2\core\core_c.h"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\imgproc\imgproc_c.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\highgui\highgui_c.h"

#include <iostream>
#include <math.h>
#include <string.h>


#include "OpenCVHelperLibrary\cv_helper_lib.h"


using namespace cv;
using namespace std;


namespace tag_detection_module {

  

CvFont font;

#ifndef thresh_S              //define the threshold values we selected according to the color info we want to hangle,this example we use pure RGB for example
#define thresh_S  245
#define thresh_V  245
#define thresh_HR_low    5
#define thresh_HR_hight  250
#define thresh_HG_low    55
#define thresh_HG_hight  65
#define thresh_HB_low    115
#define thresh_HB_hight  125
#define threshold_Canny  50
#endif


/*
//   This method is to detect the rgb region from the source image ,the detail way  is to change color space  from rgb to hsv in order to reduce 
//   the influence from the  illumination .Then we need   to create a mask from  s,v channel adapt a special threshold and use it  to  cope  H channel.
//   Next we need to segment the  H channel result up to the range of values on different color  and transfor it to binary image   
*/
std::vector<IplImage*> dectetColor(IplImage* img){

std::vector<IplImage*>* Result=new std::vector<IplImage*>();//the final result after coped
IplImage*     src=cvCreateImage(cvGetSize(img),8,3);
IplImage*     hsv=cvCreateImage(cvGetSize(src),8,3);//the last pare means the num of channel
IplImage*       H=cvCreateImage(cvGetSize(src),8,1);
IplImage*       S=cvCreateImage(cvGetSize(src),8,1);
IplImage*       V=cvCreateImage(cvGetSize(src),8,1);
IplImage*  copedH=cvCreateImage(cvGetSize(src),8,1);//we need cope the H channel before we handle it adapt to the values of rgb 
//define the final coped result
IplImage* SResult=cvCreateImage(cvGetSize(src),8,1);
IplImage* VResult=cvCreateImage(cvGetSize(src),8,1);
IplImage* SVMask =cvCreateImage(cvGetSize(src),8,1); //the mask from the coped info about s,v channel
IplImage* HResult=cvCreateImage(cvGetSize(src),8,1);

//define the result when hanle H chanel
IplImage* HrResult=cvCreateImage(cvGetSize(src),8,1);
IplImage* HgResult=cvCreateImage(cvGetSize(src),8,1);
IplImage* HbResult=cvCreateImage(cvGetSize(src),8,1);

try{
src=cvCloneImage(img);               //clone source image to this functin tempt image
cvSmooth( src,src, CV_GAUSSIAN,5,5 );//smooth the source image to resuce noise

//transfor color sapce from bgr to hsv and get each chanel in hsv color space and show .
//the transfor formulate :
//                      V <- max(R,G,B)
//                      S <- (V-min(R,G,B))/V   if V¡Ù0, 0 otherwise
//
//                           (G - B)*60/S,  if V=R
//                      H <- 180+(B - R)*60/S,  if V=G
//                           240+(R - G)*60/S,  if V=B
//
//if H<0 then H<-H+360
//On output 0¡ÜV¡Ü1, 0¡ÜS¡Ü1, 0¡ÜH¡Ü360.
//The values are then converted to the destination data type:
//    8-bit images:
//        V <- V*255, S <- S*255, H <- H/2 (to fit to 0..255)
//    16-bit images (currently not supported):
//        V <- V*65535, S <- S*65535, H <- H
//    32-bit images:
//        H, S, V are left as is
//That means pure r to hsv is: 255,255,0
//           pure g to hsv is: 255,255,60
//		   pure b to hsv is: 255,255,120
//
cvCvtColor(src,hsv,CV_BGR2HSV);
cvSplit(hsv,H,S,V,0);

//handle S,V channel 
cvThreshold(V,VResult,thresh_V,255,CV_THRESH_BINARY);
cvThreshold(S,SResult,thresh_S,255,CV_THRESH_BINARY);
//at first wo get the mask from the info of S,V channel and cope the H  before handle H channel
cvAnd(VResult,SResult,SVMask,0);                                             // creat  the s,v and mask
cvCopy(H,copedH,SVMask);

cvThreshold(copedH,HbResult,thresh_HB_hight,255,CV_THRESH_TOZERO_INV);       // larger than thrshold is 0 and smaller is  not changed
cvThreshold(HbResult,HbResult,thresh_HB_low,255,CV_THRESH_BINARY);           // larger than threshold is 255 ang smaller is 0

cvThreshold(copedH,HgResult,thresh_HG_hight,255,CV_THRESH_TOZERO_INV);       // larger than thrshold is 0 and smaller is  not changed
cvThreshold(HgResult,HgResult,thresh_HG_low,255,CV_THRESH_BINARY);           // larger than threshold is 255 ang smaller is 0

cvThreshold(copedH,HrResult,thresh_HR_hight,255,CV_THRESH_TOZERO_INV);       // larger than thrshold is 0 and smaller is  not changed
cvThreshold(HrResult,HrResult,thresh_HR_low,255,CV_THRESH_BINARY);           // larger than threshold is 255 ang smaller is 0
cvThreshold(HrResult,HrResult,254,255,CV_THRESH_BINARY_INV);                 //the source img is binary image and need reserve to get red channel
//reverse the binary imgage in order to make the  selected region black
//cvNot(HrResult,HrResult);
//cvNot(HgResult,HgResult);
//cvNot(HbResult,HbResult);

//create the vector of result on grb channel
Result->push_back(HrResult);
Result->push_back(HgResult);
Result->push_back(HbResult);

//Release the memory at last
cvReleaseImage(&HResult);
cvReleaseImage(&SResult);
cvReleaseImage(&VResult);
cvReleaseImage(&SVMask);
cvReleaseImage(&copedH);
cvReleaseImage(&H);
cvReleaseImage(&S);
cvReleaseImage(&V);
cvReleaseImage(&hsv);

}catch(cv::Exception e){
 cout<<"Exception:"<<e.msg<<endl;
 exit(0);
}

return *Result;
}

/*
//  The input para is the binary image as result of color segment result .First we need to get the single region 
*/
IplImage* handleBinaryImg(IplImage* src){
	//reduce the noise 
IplImage* Result=cvCreateImage(cvGetSize(src),8,1);
IplImage* Tempt =cvCreateImage(cvGetSize(src),8,1);
IplConvKernel* element=cvCreateStructuringElementEx(5,5,2,2,CV_SHAPE_RECT,0);
cvMorphologyEx(src,Tempt,0,element,CV_MOP_OPEN,1);
cvMorphologyEx(Tempt,Result,0,element,CV_MOP_CLOSE,1);
//cvErode(Result,Tempt,element,1);
//cvDilate(Tempt,Result,element,1);

return  Result;
} 

/* 
// This function is to finds a cosine of angle between vectorsfrom pt0->pt1 and from pt0->pt2
*/
double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 ){
    double dx1 = pt1->x  -  pt0->x;
    double dy1 = pt1->y  -  pt0->y;
    double dx2 = pt2->x  -  pt0->x;
    double dy2 = pt2->y  -  pt0->y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/*
// This meathod is to get the contours of squrares we need to  detect
*/
CvSeq* findSquares4( IplImage* img, CvMemStorage* storage ){  
	
    CvSeq* contours; //use cvFindContours to hold contours find from source image
    CvSeq* result;   //use cvApproxPoly to get the info each contour 
	CvSeq* squares=cvCreateSeq(  0,  sizeof( CvSeq),  sizeof( CvPoint),  storage );// hold the result 

	//canny test ,but we needn't this handling beacuse our input image is a binary image at first
	//cvNamedWindow("sourceImg",1);
	//cvShowImage("sourceImg",img);
	//IplImage* temp=cvCreateImage(cvGetSize(img),8,1);
	//cvCanny(img,temp, 0,threshold_Canny, 5);
	//cvNamedWindow("tempImg",1);
	//cvShowImage("tempImg",temp);

	cvFindContours(img, storage, &contours, sizeof(CvContour),    CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	while(contours)
    {
        result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
     if(result->total==4 /*&& fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 20*/)//this the condition we used to select goal rectangle
        {   
			 double maxCosine = 0;

             for( int j = 2; j < 5; j++ )
                   {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(( CvPoint*)cvGetSeqElem(  result,  j%4), ( CvPoint*)cvGetSeqElem(  result,  j-2 ), ( CvPoint*)cvGetSeqElem(  result,  j-1 )));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        for(  int i  =  0; i < 4; i++ )
					          cvSeqPush(  squares,  ( CvPoint*)cvGetSeqElem(  result,  i ));

        }
          contours = contours->h_next;
    }
 
    return squares;
}

/*
// the function draws all the squares in the image
*/
IplImage* drawSquares( IplImage* src, CvSeq* squares ){
	int  total = squares->total;
	IplImage* img=cvCreateImage(cvGetSize(src),8,3);
	CvSeqReader reader;
	cvStartReadSeq(squares,&reader,0);
	for(int i=0;i<total;i+=4){
		    int xSum=0;
		    int ySum=0;
		    CvPoint *pt[4];
            for(int j=i;j<i+4;j++){
                pt[j%4] = (CvPoint*)cvGetSeqElem(squares, j);
				xSum+=pt[j%4]->x;
				ySum+=pt[j%4]->y;
			}
            cvLine(img, *pt[0], *pt[1], cvScalar(255));
            cvLine(img, *pt[1], *pt[2], cvScalar(255));
            cvLine(img, *pt[2], *pt[3], cvScalar(255));
            cvLine(img, *pt[3], *pt[0], cvScalar(255));	
			cvPutText(img, "_+_", cvPoint(xSum/4,ySum/4), &font, cvScalar(0, 0, 0, 0));
			cout<<"x:"<<xSum/4<<",  y:"<<ySum/4<<endl;
		
	}
	 return img;
}



/*
//  We  go on handling the binary image according to the special shape info ,we use  rectangle for  example.  
//
*/
IplImage* detectShape(IplImage* src){

	IplImage* Result=cvCreateImage(cvGetSize(src),8,3);
	//IplImage* Temp=cvCreateImage(cvGetSize(src),8,3);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* squares=findSquares4(src,storage);
	Result=drawSquares(src,squares);
	//cvCvtColor(Temp,Result,CV_GRAY2RGB);
	return  Result;
}
/*
// This method is to  get the center point of special shape binary region .
*/
std::vector<cvCenterPoint> getCenterPoint(IplImage* src,cvTagType type){

	std::vector<cvCenterPoint>* centerPoints=new std::vector<cvCenterPoint>();
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* squares=findSquares4(src,storage);
	int  total = squares->total;
	CvSeqReader reader;
	cvStartReadSeq(squares,&reader,0);
	for(int i=0;i<total;i+=4){
		    int xSum=0;
		    int ySum=0;
		    CvPoint *pt[4];
            for(int j=i;j<i+4;j++){
                pt[j%4] = (CvPoint*)cvGetSeqElem(squares, j);
				xSum+=pt[j%4]->x;
				ySum+=pt[j%4]->y;
			}
            int pointX=xSum/4;
			int pointY=ySum/4;
			cvCenterPoint cpSet;
			cpSet.center_Point=cvPoint(pointX,pointY);
			cpSet.tag_Type= type ;
			centerPoints->push_back(cpSet);
		
	}

	return  *centerPoints;

 } 

}

