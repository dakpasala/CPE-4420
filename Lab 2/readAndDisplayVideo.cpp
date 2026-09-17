/************************************************************
* File: readAndDisplayVideo.cpp
*
* Description: Reads a video file frame by frame and applies
* manual grayscale conversion and Sobel edge detection,
* comparing execution time against OpenCV's built-in
* equivalents.
*
* Author: Dakshesh Pasala, Nickaan Jahadi
*
* Revisions:
*
************************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>

/*-----------------------------------------------------
* Function: grayscale
*
* Description: Converts a BGR frame to grayscale using
* floating-point BT.709 luma coefficients, accessed via
* row pointers for speed.
*
* param frame: cv::Mat: input BGR frame
*
* return: cv::Mat: single-channel (CV_8UC1) grayscale image
*-----------------------------------------------------*/
cv::Mat grayscale(cv::Mat frame) {
    int m = frame.rows;
    int n = frame.cols;

    cv::Mat newImage(m, n, CV_8UC1); // 8-bit unsigned and C1 = 1 channel (grayscale)

    for (int i = 0; i < m; i++) {
        cv::Vec3b* row = frame.ptr<cv::Vec3b>(i);
        unsigned char* newRow = newImage.ptr<unsigned char>(i);

        for (int j = 0; j < n; j++) {
            cv::Vec3b& pixel = row[j];
            unsigned char blue = pixel[0];
            unsigned char green = pixel[1];
            unsigned char red = pixel[2];

            newRow[j] = cv::saturate_cast<unsigned char>(0.2126 * red + 0.7152 * green + 0.0722 * blue);
            // unsigned char will round to nearest integer, clamp anything below 0 to 0, and clamp anything above 255 to 255
            // necessary to average the values and ensure we are not out of the color pixel range
        }
    }

    return newImage;
}

/*-----------------------------------------------------
* Function: sobel
*
* Description: Computes Sobel edge magnitude using the
* exact Euclidean formula (sqrt(Gx^2 + Gy^2)) from a
* grayscale frame, accessed via row pointers for speed.
*
* param frame: cv::Mat: single-channel grayscale input
*
* return: cv::Mat: single-channel Sobel edge magnitude
* image (2 rows/cols smaller than input)
*-----------------------------------------------------*/
cv::Mat sobel(cv::Mat frame) {
    int m = frame.rows;
    int n = frame.cols;

    cv::Mat newImage(m - 2, n - 2, CV_8UC1, cv::Scalar(0));

    // issue was everytime we did frame.at<uchar>(i, j) openCV has to recompute data + i * step + j * elementSize,
    // then run a runtime check that T actually matches the Mat's real type
    // and then returns a reference to the byte

    for (int i = 1; i < m - 1; i++) {
        uchar* topRow    = frame.ptr<uchar>(i - 1); // this will just compute it once per row, then it's just array indexing
        uchar* middleRow = frame.ptr<uchar>(i);
        uchar* bottomRow = frame.ptr<uchar>(i + 1);
        uchar* newRow    = newImage.ptr<uchar>(i - 1);

        for (int j = 1; j < n - 1; j++) {
            int topLeft     = topRow[j - 1]; // plain pointer arithmetic
            int topMiddle   = topRow[j];
            int topRight    = topRow[j + 1];

            int middleLeft  = middleRow[j - 1];
            int middleRight = middleRow[j + 1];

            int bottomLeft   = bottomRow[j - 1];
            int bottomMiddle = bottomRow[j];
            int bottomRight  = bottomRow[j + 1];

            int gx = -topLeft + topRight -2 * middleLeft + 2 * middleRight - bottomLeft + bottomRight;
            int gy = -topLeft - 2 * topMiddle - topRight + bottomLeft + 2 * bottomMiddle + bottomRight;

            double magnitude = std::sqrt(gx * gx + gy * gy);

            newRow[j - 1] = cv::saturate_cast<uchar>(magnitude);
        }
    }

    return newImage;
}

/*-----------------------------------------------------
* Function: main
*
* Description: Opens the video, runs manual vs OpenCV
* grayscale and Sobel timing comparisons per frame, then
* prints the averages.
*
* return: int: 0 on success, 1 if the video failed to open
*-----------------------------------------------------*/
int main() {
    cv::VideoCapture video("video.mp4");

    if (!video.isOpened()) {
        std::cerr << "no vid\n";
        return 1;
    }

    cv::Mat frame;

    double totalManualGray = 0.0;
    double totalManualSobel = 0.0;
    double totalOpenCVGray = 0.0;
    double totalOpenCVSobel = 0.0;

    long long frameCount = 0;

    while (true) {
        video >> frame;
        if (frame.empty()) break;

        // grayscale timing
        auto startManualGray = std::chrono::high_resolution_clock::now();
        cv::Mat grayFrame = grayscale(frame);
        auto endManualGray = std::chrono::high_resolution_clock::now();

        // sobel timing
        auto startManualSobel = std::chrono::high_resolution_clock::now();
        cv::Mat sobelFrame = sobel(grayFrame);
        auto endManualSobel = std::chrono::high_resolution_clock::now();

        // openCV grayscale timing
        cv::Mat openCVGrayFrame;
        auto startOpenCVGray = std::chrono::high_resolution_clock::now();
        cv::cvtColor(frame, openCVGrayFrame, cv::COLOR_BGR2GRAY);
        auto endOpenCVGray = std::chrono::high_resolution_clock::now();

        // openCV sobel timing
        cv::Mat gx;
        cv::Mat gy;
        cv::Mat magnitude;
        cv::Mat openCVSobelFrame;

        auto startOpenCVSobel = std::chrono::high_resolution_clock::now();
        cv::Sobel(openCVGrayFrame, gx, CV_32F, 1, 0, 3);
        cv::Sobel(openCVGrayFrame, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, magnitude);
        magnitude.convertTo(openCVSobelFrame, CV_8UC1);
        auto endOpenCVSobel = std::chrono::high_resolution_clock::now();

        // millisecond conversion
        double manualGrayTime = std::chrono::duration<double, std::milli>(endManualGray - startManualGray).count();
        double manualSobelTime = std::chrono::duration<double, std::milli>(endManualSobel - startManualSobel).count();

        double openCVGrayTime = std::chrono::duration<double, std::milli>(endOpenCVGray - startOpenCVGray).count();
        double openCVSobelTime = std::chrono::duration<double, std::milli>(endOpenCVSobel - startOpenCVSobel).count();

        totalManualGray += manualGrayTime;
        totalManualSobel += manualSobelTime;
        totalOpenCVGray += openCVGrayTime;
        totalOpenCVSobel += openCVSobelTime;

        frameCount++;

        std::cout << std::fixed << std::setprecision(3)
                  << "Frame " << frameCount << " | "
                  << "Manual gray: " << manualGrayTime << " ms | "
                  << "OpenCV gray: " << openCVGrayTime << " ms | "
                  << "Manual Sobel: " << manualSobelTime << " ms | "
                  << "OpenCV Sobel: " << openCVSobelTime << " ms\n";

        cv::imshow("Video", frame);
        cv::imshow("Grayscale", grayFrame);
        cv::imshow("Sobel", sobelFrame);

        int key = cv::waitKey(5);
        if (key == 'q') break;
    }

    video.release();
    cv::destroyAllWindows();

    // ai generated below

    if (frameCount > 0) {
        std::cout << "\nAverage processing times\n"
                  << "------------------------\n"
                  << std::fixed << std::setprecision(3)
                  << "Manual grayscale: "
                  << totalManualGray / frameCount << " ms\n"
                  << "OpenCV grayscale: "
                  << totalOpenCVGray / frameCount << " ms\n"
                  << "Manual Sobel:     "
                  << totalManualSobel / frameCount << " ms\n"
                  << "OpenCV Sobel:     "
                  << totalOpenCVSobel / frameCount << " ms\n";
    }

    return 0;
}