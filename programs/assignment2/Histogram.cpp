/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to use the function calcHist
 * @author
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <sys/timeb.h>

using namespace std;
using namespace cv;

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/**
 * @function calculate a histogram
 */
void calculate_histogram(Mat image, Mat hist) {
    int k;
    for(int i = 0; i < image.rows; i++) {
        uchar *ip = image.ptr<uchar>(i);
        uchar *hp = hist.ptr<uchar>(0);
        for(int j = 0; j < image.cols; j++) {
            k = ip[j];
            hp[k] = hp[k] + 1;
        }
    }
}

/**
 * @function main
 */
int main(int argc, char** argv)
{

  Mat src, dst;
  char* filename;

  if (argc > 1) {
      filename = argv[1];
  } else {
      filename = "../data/lena.jpg";
  }

  src = imread( filename, IMREAD_COLOR );

  if( src.empty() )
    { return -1; }

  // Separate the image in 3 places ( B, G and R )
  vector<Mat> bgr_planes;
  split( src, bgr_planes );

  // Establish the number of bins
  int histSize = 256;

  // initialize histograms
  Mat b_hist = Mat(1, histSize, CV_8UC1, Scalar(0));
  Mat g_hist = Mat(1, histSize, CV_8UC1, Scalar(0));
  Mat r_hist = Mat(1, histSize, CV_8UC1, Scalar(0));

  /// Compute the histograms:
  double hist_time = read_timer();
  calculate_histogram( bgr_planes[0], b_hist);
  calculate_histogram( bgr_planes[1], g_hist);
  calculate_histogram( bgr_planes[2], r_hist);
  hist_time = read_timer() - hist_time;

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  /// Draw for each channel

  for( int i = 0; i < histSize; i++ ) {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<uchar>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<uchar>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<uchar>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<uchar>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<uchar>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<uchar>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Print the times
  printf("======================================================================================================\n");
  printf("\tHistogram for %s\n", filename);
  printf("------------------------------------------------------------------------------------------------------\n");
  printf("Performance: \t\tRuntime (ms)\t MFLOPS \n");
  printf("------------------------------------------------------------------------------------------------------\n");
  printf("Histogram calculation: \t\t%4f\t%4f\n",  hist_time * 1.0e3, 3 * src.rows * src.cols / (1.0e6 *  hist_time));

  /// Display
  namedWindow("calcHist Demo", WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );

  waitKey(0);

  return 0;
}
