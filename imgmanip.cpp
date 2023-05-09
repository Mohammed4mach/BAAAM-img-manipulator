#include "imgmanip.h"
#include <bits/stdc++.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QImage>

using namespace std;
using namespace cv;

ImgManip::ImgManip()
{ }

QPixmap ImgManip::getPixmap(Mat image, QImage::Format image_type)
{
    // Convert the OpenCV Mat to a Qt QImage
    // QImage qImage(image.data, image.cols, image.rows, static_cast<int>(image.step), QImage::Format_RGB888);
    QImage qImage = QImage((unsigned char*)image.data, image.cols, image.rows, image_type).rgbSwapped();

    // Convert the QImage to a QPixmap
    QPixmap pixmap = QPixmap::fromImage(qImage);

    return pixmap;
}

Mat ImgManip::customThreshold(const Mat& gray, int thresholdValue)
{
        Mat binary(gray.size(), CV_8UC1);
        for (int i = 0; i < gray.rows; i++) {
            for (int j = 0; j < gray.cols; j++) {
                if (gray.at<uchar>(i, j) > thresholdValue) {
                    binary.at<uchar>(i, j) = 255;
                }
                else {
                    binary.at<uchar>(i, j) = 0;
                }
            }
        }

        return binary;
}

Mat ImgManip::boxFilterAux(const Mat& src, int kernelSize) {
    // Create a new image to store the filtered result
    Mat filteredImage = Mat::zeros(src.rows, src.cols, src.type());

    // Calculate the size of the kernel (should be an odd number)
    int kernelRadius = kernelSize / 2;

    // Loop over each pixel in the source image
    for (int y = kernelRadius; y < src.rows - kernelRadius; y++)
    {
        for (int x = kernelRadius; x < src.cols - kernelRadius; x++)
        {
            // Compute the sum of pixel values in the kernel
            float sum = 0;
            for (int i = -kernelRadius; i <= kernelRadius; i++)
            {
                for (int j = -kernelRadius; j <= kernelRadius; j++)
                {
                    sum += src.at<uchar>(y + i, x + j);
                }
            }

            // Compute the mean of pixel values in the kernel and store in the filtered image
            filteredImage.at<uchar>(y, x) = sum / (kernelSize * kernelSize);
        }
    }

    return filteredImage;
}

Mat ImgManip::minFilter(const Mat& image, int kernel_size) {
    int offset = kernel_size / 2;
    Mat filtered_image = Mat::zeros(image.rows, image.cols, image.type());

    for (int i = offset; i < image.rows - offset; i++) {
        for (int j = offset; j < image.cols - offset; j++) {
            uchar min_val = 255;
            for (int ki = -offset; ki <= offset; ki++) {
                for (int kj = -offset; kj <= offset; kj++) {
                    uchar pixel_value = image.at<uchar>(i + ki, j + kj);
                    if (pixel_value < min_val) {
                        min_val = pixel_value;
                    }
                }
            }
            filtered_image.at<uchar>(i, j) = min_val;
        }
    }

    return filtered_image;
}

/** For Canny Threshold **/
// Sobel operators for computing gradient
int Gx[3][3] = { {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1} };

int Gy[3][3] = { {-1, -2, -1},
                {0, 0, 0},
                {1, 2, 1} };

// Gaussian kernel for smoothing
double G[5][5] = { {2, 4, 5, 4, 2},
                  {4, 9, 12, 9, 4},
                  {5, 12, 15, 12, 5},
                  {4, 9, 12, 9, 4},
                  {2, 4, 5, 4, 2} };

// Function to convolve a kernel with an image
void convolve(Mat img, int kernel[][3], Mat result) {
    int kh = 3, kw = 3;
    int y, x, i, j, sum;
    for (y = 1; y < img.rows - 1; y++) {
        for (x = 1; x < img.cols - 1; x++) {
            sum = 0;
            for (i = 0; i < kh; i++) {
                for (j = 0; j < kw; j++) {
                    sum += img.at<uchar>(y + i - 1, x + j - 1) * kernel[i][j];
                }
            }
            result.at<uchar>(y, x) = sum;
        }
    }
}

// Function to apply Gaussian smoothing
void smooth(Mat img, Mat result) {
    int kh = 5, kw = 5;
    int y, x, i, j;
    double sum, weight, val;
    for (y = 2; y < img.rows - 2; y++) {
        for (x = 2; x < img.cols - 2; x++) {
            sum = 0;
            weight = 0;
            for (i = 0; i < kh; i++) {
                for (j = 0; j < kw; j++) {
                    val = img.at<uchar>(y + i - 2, x + j - 2);
                    sum += val * G[i][j];
                    weight += G[i][j];
                }
            }
            result.at<uchar>(y, x) = sum / weight;
        }
    }
}
/** For Canny Threshold **/

QPixmap ImgManip::get_histogram(string path_to_img)
{
    // Load the image
    Mat image = imread(path_to_img, IMREAD_GRAYSCALE);

    // Compute the histogram
    int histogram[256] = { 0 };
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int pixelValue = image.at<uchar>(i, j);
            histogram[pixelValue]++;
        }
    }

    // Display the histogram
    int histWidth = 512, histHeight = 400;
    int binWidth = cvRound((double)histWidth / 256);
    Mat histImage(histHeight, histWidth, CV_8UC1, Scalar(0));
    int maxCount = 0;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > maxCount) {
            maxCount = histogram[i];
        }
    }
    for (int i = 0; i < 256; i++) {
        int binHeight = cvRound((double)histogram[i] / maxCount * histHeight);
        rectangle(histImage, Point(i * binWidth, histHeight - binHeight), Point((i + 1) * binWidth - 1, histHeight - 1), Scalar(255), FILLED);
    }

    return getPixmap(histImage, QImage::Format_Indexed8);
}

QPixmap ImgManip::segmentation(string path_to_img, int threshold_value)
{
    // Load the image
    Mat image = imread(path_to_img, IMREAD_GRAYSCALE);

    // Apply thresholding to segment the image
    Mat thresholded_image = customThreshold(image, threshold_value);

    // Display the original and segmented images

    return getPixmap(thresholded_image, QImage::Format_Indexed8);
}

QPixmap ImgManip::box_filter(string path_to_img, int kernel_size)
{
    // Read the image from file
    Mat image = imread(path_to_img, IMREAD_GRAYSCALE);

    // Check if the image was loaded successfully
    if (image.empty())
    {
        printf("Could not open or find the image\n");
    }

    // Apply a box filter with a kernel size of 5x5
    Mat filteredImage = boxFilterAux(image, kernel_size);

    return getPixmap(filteredImage, QImage::Format_Indexed8);
}

QPixmap ImgManip::min_filter(string path_to_img, int kernel_size)
{
    // Load image data
    Mat image = imread(path_to_img, IMREAD_GRAYSCALE);

    Mat filtered_image = minFilter(image, kernel_size);

    return getPixmap(filtered_image, QImage::Format_Indexed8);
}

QPixmap ImgManip::canny_threshold(string path_to_img)
{
    Mat img = imread(path_to_img, IMREAD_GRAYSCALE);

    Mat Gx_img(img.size(), CV_8UC1);
    Mat Gy_img(img.size(), CV_8UC1);
    Mat mag(img.size(), CV_8UC1);
    Mat dir(img.size(), CV_8UC1);
    Mat smoothed(img.size(), CV_8UC1);
    Mat edges(img.size(), CV_8UC1);
    convolve(img, Gx, Gx_img);
    convolve(img, Gy, Gy_img);
    int x, y;
    double dx, dy;
    for (y = 1; y < img.rows - 1; y++) {
        for (x = 1; x < img.cols - 1; x++) {
            dx = Gx_img.at<uchar>(y, x + 1) - Gx_img.at<uchar>(y, x - 1);
            dy = Gy_img.at<uchar>(y + 1, x) - Gy_img.at<uchar>(y - 1, x);
            mag.at<uchar>(y, x) = sqrt(dx * dx + dy * dy);
            dir.at<uchar>(y, x) = atan2(dy, dx) * 180 / M_PI;
        }
    }
    smooth(mag, smoothed);
    for (y = 1; y < img.rows - 1; y++) {
        for (x = 1; x < img.cols - 1; x++) {
            if (smoothed.at<uchar>(y, x) > 50) {
                if ((dir.at<uchar>(y, x) >= -22.5 && dir.at<uchar>(y, x) < 22.5) ||
                    (dir.at<uchar>(y, x) >= 157.5 && dir.at<uchar>(y, x) < -157.5)) {
                    edges.at<uchar>(y, x) = 255;
                }
                if ((dir.at<uchar>(y, x) >= 22.5 && dir.at<uchar>(y, x) < 67.5) ||
                    (dir.at<uchar>(y, x) >= -157.5 && dir.at<uchar>(y, x) < -112.5)) {
                    edges.at<uchar>(y, x) = 255;
                }
                if ((dir.at<uchar>(y, x) >= 67.5 && dir.at<uchar>(y, x) < 112.5) ||
                    (dir.at<uchar>(y, x) >= -112.5 && dir.at<uchar>(y, x) < -67.5)) {
                    edges.at<uchar>(y, x) = 255;
                }
                if ((dir.at<uchar>(y, x) >= 112.5 && dir.at<uchar>(y, x) < 157.5) ||
                    (dir.at<uchar>(y, x) >= -67.5 && dir.at<uchar>(y, x) < -22.5)) {
                    edges.at<uchar>(y, x) = 255;
                }
            }
        }
    }

    imshow("Output Image", edges);

    return getPixmap(edges, QImage::Format_Indexed8);
}

QPixmap ImgManip::equalization(string path_to_img)
{
    // Load the image
    Mat image = imread(path_to_img, IMREAD_GRAYSCALE);

    // Compute the histogram
    int histogram[256] = { 0 };
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int pixelValue = image.at<uchar>(i, j);
            histogram[pixelValue]++;
        }
    }

    // Compute the cumulative distribution function
    int cdf[256] = { 0 };
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Compute the equalized histogram
    int equalizedHistogram[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        equalizedHistogram[i] = cvRound((double)(cdf[i] - cdf[0]) / ((image.rows * image.cols) - cdf[0]) * 255);
    }

    // Apply histogram equalization
    Mat equalized(image.size(), CV_8UC1);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int pixelValue = image.at<uchar>(i, j);
            equalized.at<uchar>(i, j) = equalizedHistogram[pixelValue];
        }
    }

    return getPixmap(equalized, QImage::Format_Indexed8);
}

QPixmap ImgManip::laplacian(string path_to_img, bool enhanced)
{
    // Read in the input image
    Mat input_image = imread(path_to_img, IMREAD_GRAYSCALE);

    // Define the Laplacian filter
    vector<vector<int>> laplacian_filter = { {0, -1, 0}, {-1, 4, -1}, {0, -1, 0} };

    if (enhanced) laplacian_filter[1][1]++;

    // Define the output image
    Mat output_image(input_image.size(), CV_8UC1);

    // Apply the Laplacian filter
    for (int i = 1; i < input_image.rows - 1; i++) {
        for (int j = 1; j < input_image.cols - 1; j++) {
            int sum = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum += laplacian_filter[k + 1][l + 1] * input_image.at<uchar>(i + k, j + l);
                }
            }
            output_image.at<uchar>(i, j) = saturate_cast<uchar>(sum);
        }
    }


    // add the Laplacian output from the input image to produce a sharpened image
    Mat sharpened_image = input_image + output_image;

    return getPixmap(sharpened_image, QImage::Format_Indexed8);
}

QPixmap ImgManip::sobel_filter(string path_to_img)
{
    String imglo = path_to_img;
    Mat img = imread(imglo, 0);
    Mat newimg = Mat::zeros(img.size(), img.type());
    if (newimg.type() == img.type())
    {
        cout << "YESH" << endl;
    }

    int sobel_x[3][3] = { {-1,0,1},
                          {-2,0,2},
                          {-1,0,1} };

    int sobel_y[3][3] = { {-1,-2,-1},
                          { 0, 0, 0},
                          { 1, 2, 1} };

    for (int j = 0; j < img.rows - 2; j++)
    {
        for (int i = 0; i < img.cols - 2; i++)
        {
            int pixval_x =
                (sobel_x[0][0] * (int)img.at<uchar>(j, i)) + (sobel_x[0][1] * (int)img.at<uchar>(j + 1, i)) + (sobel_x[0][2] * (int)img.at<uchar>(j + 2, i)) +
                (sobel_x[1][0] * (int)img.at<uchar>(j, i + 1)) + (sobel_x[1][1] * (int)img.at<uchar>(j + 1, i + 1)) + (sobel_x[1][2] * (int)img.at<uchar>(j + 2, i + 1)) +
                (sobel_x[2][0] * (int)img.at<uchar>(j, i + 2)) + (sobel_x[2][1] * (int)img.at<uchar>(j + 1, i + 2)) + (sobel_x[2][2] * (int)img.at<uchar>(j + 2, i + 2));

            int pixval_y =
                (sobel_y[0][0] * (int)newimg.at<uchar>(j, i)) + (sobel_y[0][1] * (int)newimg.at<uchar>(j + 1, i)) + (sobel_y[0][2] * (int)newimg.at<uchar>(j + 2, i)) +
                (sobel_y[1][0] * (int)newimg.at<uchar>(j, i + 1)) + (sobel_y[1][1] * (int)newimg.at<uchar>(j + 1, i + 1)) + (sobel_y[1][2] * (int)newimg.at<uchar>(j + 2, i + 1)) +
                (sobel_y[2][0] * (int)newimg.at<uchar>(j, i + 2)) + (sobel_y[2][1] * (int)newimg.at<uchar>(j + 1, i + 2)) + (sobel_y[2][2] * (int)newimg.at<uchar>(j + 2, i + 2));

            int sum = abs(pixval_x) + abs(pixval_y);
            if (sum > 255)
            {
                sum = 255;
            }
            // cout << sum << endl;
            newimg.at<uchar>(j, i) = sum;
        }
    }

    return getPixmap(newimg, QImage::Format_Indexed8);
}

double ImgManip::transmission_time(string path_to_img, int baud_rate)
{
    // Read the input image
    Mat image = imread(path_to_img, IMREAD_GRAYSCALE);

    // Calculate the number of bits in each packet of the image
    int bits_per_pixel = 10; // including start and stop bits
    int num_bits = image.rows * image.cols * bits_per_pixel;

    // Calculate the transmission time
    double transmission_time = (double)num_bits / baud_rate;

    // Print the transmission time to the console
    return transmission_time;
}

