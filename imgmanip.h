#ifndef IMGMANIP_H
#define IMGMANIP_H

#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QImage>

using namespace std;
using namespace cv;

class ImgManip
{
public:
    ImgManip();
    static QPixmap img_out;
    static double trans_time;

    static QPixmap getPixmap(cv::Mat image, QImage::Format image_type);
    static Mat customThreshold(const Mat& gray, int thresholdValue);
    static Mat boxFilterAux(const Mat& src, int kernelSize);
    static Mat minFilter(const Mat& image, int kernel_size);

    static QPixmap get_histogram(string path_to_img);
    static QPixmap box_filter(string path_to_img, int kernel_size);
    static QPixmap min_filter(string path_to_img, int kernel_size);
    static QPixmap sobel_filter(string path_to_img);
    static QPixmap canny_threshold(string path_to_img);
    static QPixmap equalization(string path_to_img);
    static QPixmap laplacian(string path_to_img, bool enhanced);
    static QPixmap segmentation(string path_to_img, int threshold_value);
    static double transmission_time(string path_to_img, int baud_rate);
};

#endif // IMGMANIP_H
