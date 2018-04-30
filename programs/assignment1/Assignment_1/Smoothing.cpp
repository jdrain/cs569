/**
 * file Smoothing.cpp
 * brief Sample code for simple filters
 * author OpenCV team
 */

#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <sys/timeb.h>

using namespace std;
using namespace cv;

Mat src;
Mat cp1;
Mat cp2;
Mat cp3;

char window_name[] = "Smoothing Demo";

/// Function headers
int display_caption( const char* caption );
int display_dst( int delay );

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

// smooth image
void smoothImage(Mat image, short filter[3][3]) {
    int rows = image.rows;
    int cols = image.cols;
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                int sum = 0;
                for (int a = -1; a < 2; a++) {
                    for (int b = -1; b < 2; b++) {
                        sum += filter[1 + a][1 + b]*image.at<Vec3b>(i+a, j+b)[k];
                    }
                }
                if (sum < 0) sum = 0;
                image.at<Vec3b>(i, j)[k] = sum/9;
            }
        }
    }
}

// define different filters
short lpf_filter_6[3][3] =
   { {0, 1, 0},
     {1, 2, 1},
     {0, 1, 0}};

short lpf_filter_10[3][3] =
   { {1, 1, 1},
     {1, 2, 1},
     {1, 1, 1}};

short lpf_filter_16[3][3] =
   { {1, 2, 1},
     {2, 4, 2},
     {1, 2, 1}};

short lpf_filter_32[3][3] =
   { {1,  4, 1},
     {4, 12, 4},
     {1,  4, 1}};

short hpf_filter_1[3][3] =
   { { 0, -1,  0},
     {-1,  5, -1},
     { 0, -1,  0}};

short hpf_filter_2[3][3] =
   { {-1, -1, -1},
     {-1,  9, -1},
     {-1, -1, -1}};

short hpf_filter_3[3][3] =
   { { 1, -2,  1},
     {-2,  5, -2},
     { 1, -2,  1}};

/**
 * function main
 */
int main( int argc, char ** argv )
{
    // namedWindow( window_name, WINDOW_AUTOSIZE );

    /// Load the source image
    const char* filename; "../data/lena.jpg";
    short* filter;

    if (argc == 1) {
        filename = "../data/lena.jpg";
    } else {
        filename = argv[1];
    }

    src = imread( filename, IMREAD_COLOR );
    cp1 = src.clone();
    cp2 = src.clone();
    cp3 = src.clone();

    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Usage: ./Smoothing [image_name -- default ../data/lena.jpg] \n");
        return -1;
    }

    double t1 = read_timer();
    smoothImage(cp1, lpf_filter_32);
    t1 = read_timer() - t1;

    double t2 = read_timer();
    smoothImage(cp2, lpf_filter_6);
    t2 = read_timer() - t2;

    double t3 = read_timer();
    smoothImage(cp3, lpf_filter_10);
    t3 = read_timer() - t3;

    /// Print the times
    printf("======================================================================================================\n");
    printf("\tSmoothing for %s\n", filename);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance: \t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Filter 1: \t\t%4f\t%4f\n",  t1 * 1.0e3, 9 * cp1.rows * cp1.cols / (1.0e6 *  t1));
    printf("Filter 2: \t\t%4f\t%4f\n",  t2 * 1.0e3, 9 * cp2.rows * cp2.cols / (1.0e6 *  t2));
    printf("Filter 3: \t\t%4f\t%4f\n",  t3 * 1.0e3, 9 * cp3.rows * cp3.cols / (1.0e6 *  t3));

    imshow("Smoothing Demo -- Blurred 1", cp1);
    imshow("Smoothing Demo -- Blurred 2", cp2);
    imshow("Smoothing Demo -- Blurred 3", cp3);
    waitKey(0);

    return 0;
}
