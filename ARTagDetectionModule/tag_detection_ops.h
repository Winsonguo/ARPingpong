#ifndef TAG_DETECTION_OPS_H_
#define TAG_DETECTION_OPS_H_

#ifdef _EXPORTINGARTDM
  #define ARTDM_CLASS_DECLSPEC    __declspec(dllexport)
#else
  #define ARTDM_CLASS_DECLSPEC    __declspec(dllimport)
#endif


//forward declaration
namespace cv {
  class Mat;
}

namespace tag_detection_module{
/*define the  color tag  type*/
enum  cvTagType{
    CV_TAG_RED=1,     //red tag 
	CV_TAG_GREEN=2,   //green tag
	CV_TAG_BLUE=3     //blue tag
};
/*define the special struct to hold the center point info*/
struct cvCenterPoint{
	CvPoint    center_Point;  //center point info 
    cvTagType  tag_Type;      //tag type info 
};
/*define common operation dure catch tag info*/
  class ARTDM_CLASS_DECLSPEC TagDetectionOp {
 public:
	std::vector<IplImage*> dectetColor(IplImage* src);       //get the region we need by the color information 
    IplImage* handleBinaryImg(IplImage* src);                //handle the binary image to prominent the ROI 
    IplImage* detectShape(IplImage* src);                    //get the region we need by the special shape information on the base of binary image 
    std::vector<cvCenterPoint> getCenterPoint(IplImage* src,cvTagType type);//get the center point  of the region we have selected on the bianry image
    double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 );               //get the angle of the coner in order to judge the shape
    CvSeq* findSquares4( IplImage* img, CvMemStorage* storage );            //get the countor of a  binary image 
    IplImage* drawSquares( IplImage* img, CvSeq* squares );                 //draw the detect shape result
  private:


  };
}//ns tag_detection_module

#endif //TAG_DETECTION_OPS_H_