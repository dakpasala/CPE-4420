#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>

cv::Mat grayscale(cv::Mat frame) {
    int m = frame.rows;
    int n = frame.cols;

    cv::Mat newImage(m, n, CV_8UC1);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cv::Vec3b& pixel = frame.at<cv::Vec3b>(i, j);
            unsigned char blue = pixel[0];
            unsigned char green = pixel[1];
            unsigned char red = pixel[2];

            newImage.at<unsigned char>(i, j) = cv::saturate_cast<unsigned char>(0.2126 * red + 0.7152 * green + 0.0722 * blue);
        }
    }

    return newImage;
}

cv::Mat sobel(cv::Mat frame) {
    int m = frame.rows;
    int n = frame.cols;

    cv::Mat newImage(m - 2, n - 2, CV_8UC1, cv::Scalar(0));

    for (int i = 1; i < m - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            int topLeft     = frame.at<uchar>(i - 1, j - 1);
            int topMiddle   = frame.at<uchar>(i - 1, j);
            int topRight    = frame.at<uchar>(i - 1, j + 1);

            int middleLeft  = frame.at<uchar>(i, j - 1);
            int middleRight = frame.at<uchar>(i, j + 1);

            int bottomLeft   = frame.at<uchar>(i + 1, j - 1);
            int bottomMiddle = frame.at<uchar>(i + 1, j);
            int bottomRight  = frame.at<uchar>(i + 1, j + 1);

            int gx = -topLeft + topRight -2 * middleLeft + 2 * middleRight - bottomLeft + bottomRight;
            int gy = -topLeft - 2 * topMiddle - topRight + bottomLeft + 2 * bottomMiddle + bottomRight;

            double magnitude = std::sqrt(gx * gx + gy * gy);

            newImage.at<unsigned char>(i - 1, j - 1) = cv::saturate_cast<uchar>(magnitude);
        }
    }

    return newImage;
}

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