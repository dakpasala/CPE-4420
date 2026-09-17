/************************************************************
* File: lab3.cpp
*
* Description: Extends readAndDisplayVideo.cpp with a
* fixed-point (integer-only) grayscale conversion and an
* approximate (L1 norm) Sobel gradient magnitude, measuring
* speed and accuracy tradeoffs against the floating-point
* baseline.
*
* Author: Dakshesh
*
* Revisions:
*
************************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>

/*-----------------------------------------------------
* Function: grayscale
*
* Description: Converts a BGR frame to grayscale using
* floating-point BT.709 luma coefficients, accessed via
* row pointers for speed. Writes into a caller-provided
* output buffer instead of allocating one, so the same
* buffer can be allocated once and reused every frame.
*
* param frame: const cv::Mat&: input BGR frame
* param output: cv::Mat&: pre-allocated (m x n, CV_8UC1)
* destination for the grayscale result
*
* return: void
*-----------------------------------------------------*/
void grayscale(const cv::Mat& frame, cv::Mat& output) {
    int m = frame.rows;
    int n = frame.cols;

    for (int i = 0; i < m; i++) {
        const cv::Vec3b* row = frame.ptr<cv::Vec3b>(i);
        unsigned char* newRow = output.ptr<unsigned char>(i);

        for (int j = 0; j < n; j++) {
            const cv::Vec3b& pixel = row[j];
            unsigned char blue = pixel[0];
            unsigned char green = pixel[1];
            unsigned char red = pixel[2];

            newRow[j] = cv::saturate_cast<unsigned char>(0.2126 * red + 0.7152 * green + 0.0722 * blue);
            // unsigned char will round to nearest integer, clamp anything below 0 to 0, and clamp anything above 255 to 255
            // necessary to average the values and ensure we are not out of the color pixel range
        }
    }
}

/*-----------------------------------------------------
* Function: grayscaleFixedPoint
*
* Description: Converts a BGR frame to grayscale using a
* fixed-point (integer-only) approximation of the BT.709
* luma coefficients, scaled by 256 (8 fractional bits) with
* rounding to the nearest integer. Writes into a
* caller-provided output buffer instead of allocating one.
*
* param frame: const cv::Mat&: input BGR frame
* param output: cv::Mat&: pre-allocated (m x n, CV_8UC1)
* destination for the grayscale result
*
* return: void
*-----------------------------------------------------*/
void grayscaleFixedPoint(const cv::Mat& frame, cv::Mat& output) {
    int m = frame.rows;
    int n = frame.cols;

    const int redC = 54;                  // 0.2126 * 256
    const int greenC = 184;               // 0.7152 * 256 and then + 1 to = 256
    const int blueC = 18;                 // 0.0722 * 256
    const int round = 1 << 7;             // 128

    for (int i = 0; i < m; i++) {
        const cv::Vec3b* row = frame.ptr<cv::Vec3b>(i);
        unsigned char* newRow = output.ptr<unsigned char>(i);

        for (int j = 0; j < n; j++) {
            const cv::Vec3b& pixel = row[j];
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];

            int y = (redC * red + greenC * green + blueC * blue + round) >> 8;
            // add the round so the sum is pushed over the next multiple of 256

            newRow[j] = cv::saturate_cast<unsigned char>(y);
        }
    }
}

/*-----------------------------------------------------
* Function: sobel
*
* Description: Computes Sobel edge magnitude using the
* exact Euclidean formula (sqrt(Gx^2 + Gy^2)) from a
* grayscale frame, accessed via row pointers for speed.
* Writes into a caller-provided output buffer instead of
* allocating one.
*
* param frame: const cv::Mat&: single-channel grayscale input
* param output: cv::Mat&: pre-allocated ((m-2) x (n-2),
* CV_8UC1) destination for the edge magnitude result
*
* return: void
*-----------------------------------------------------*/
void sobel(const cv::Mat& frame, cv::Mat& output) {
    int m = frame.rows;
    int n = frame.cols;

    // issue was everytime we did frame.at<uchar>(i, j) openCV has to recompute data + i * step + j * elementSize,
    // then run a runtime check that T actually matches the Mat's real type
    // and then returns a reference to the byte

    for (int i = 1; i < m - 1; i++) {
        const uchar* topRow    = frame.ptr<uchar>(i - 1); // this will just compute it once per row, then it's just array indexing
        const uchar* middleRow = frame.ptr<uchar>(i);
        const uchar* bottomRow = frame.ptr<uchar>(i + 1);
        uchar* newRow           = output.ptr<uchar>(i - 1);

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
}

/*-----------------------------------------------------
* Function: sobelApprox
*
* Description: Computes Sobel edge magnitude using the
* L1-norm approximation (|Gx| + |Gy|) instead of the exact
* Euclidean magnitude, avoiding the sqrt call. Writes into
* a caller-provided output buffer instead of allocating one.
*
* param frame: const cv::Mat&: single-channel grayscale input
* param output: cv::Mat&: pre-allocated ((m-2) x (n-2),
* CV_8UC1) destination for the edge magnitude result
*
* return: void
*-----------------------------------------------------*/
void sobelApprox(const cv::Mat& frame, cv::Mat& output) {
    int m = frame.rows;
    int n = frame.cols;

    for (int i = 1; i < m - 1; i++) {
        const uchar* topRow    = frame.ptr<uchar>(i - 1);
        const uchar* middleRow = frame.ptr<uchar>(i);
        const uchar* bottomRow = frame.ptr<uchar>(i + 1);
        uchar* newRow           = output.ptr<uchar>(i - 1);

        for (int j = 1; j < n - 1; j++) {
            int topLeft     = topRow[j - 1];
            int topMiddle   = topRow[j];
            int topRight    = topRow[j + 1];

            int middleLeft  = middleRow[j - 1];
            int middleRight = middleRow[j + 1];

            int bottomLeft   = bottomRow[j - 1];
            int bottomMiddle = bottomRow[j];
            int bottomRight  = bottomRow[j + 1];

            int gx = -topLeft + topRight -2 * middleLeft + 2 * middleRight - bottomLeft + bottomRight;
            int gy = -topLeft - 2 * topMiddle - topRight + bottomLeft + 2 * bottomMiddle + bottomRight;

            int magnitude = std::abs(gx) + std::abs(gy); // change right here thats all

            newRow[j - 1] = cv::saturate_cast<uchar>(magnitude);
        }
    }
}

/*-----------------------------------------------------
* Function: main
*
* Description: Opens the video, runs baseline vs
* fixed-point grayscale and exact vs approximate Sobel
* timing/error comparisons per frame, then prints the
* averages and error metrics. All image buffers are
* allocated once, before the frame loop, and reused for
* every frame rather than reallocated each iteration.
*
* return: int: 0 on success, 1 if the video failed to open
*-----------------------------------------------------*/
int main() {
    cv::VideoCapture video("video.mp4");

    if (!video.isOpened()) {
        std::cerr << "no vid\n";
        return 1;
    }

    int frameWidth = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    // reusable output buffers, allocated once before the timed loop below;
    // every frame's results get written into these same buffers instead of
    // allocating a new cv::Mat per frame
    cv::Mat grayFrame(frameHeight, frameWidth, CV_8UC1);
    cv::Mat grayFixedFrame(frameHeight, frameWidth, CV_8UC1);
    cv::Mat sobelFrame(frameHeight - 2, frameWidth - 2, CV_8UC1, cv::Scalar(0));
    cv::Mat sobelApproxFrame(frameHeight - 2, frameWidth - 2, CV_8UC1, cv::Scalar(0));
    cv::Mat sobelCombinedFrame(frameHeight - 2, frameWidth - 2, CV_8UC1, cv::Scalar(0));

    cv::Mat frame;

    double totalBaselineGray = 0.0;
    double totalFixedGray = 0.0;
    double totalBaselineSobel = 0.0;
    double totalApproxSobel = 0.0;
    double totalCombinedSobel = 0.0;

    double maxGrayError = 0.0;
    double sumGrayError = 0.0;

    double maxSobelError = 0.0;
    double sumSobelError = 0.0;

    long long frameCount = 0;

    while (true) {
        video >> frame;
        if (frame.empty()) break;

        // baseline grayscale (floating point BT.709)
        auto startBaselineGray = std::chrono::high_resolution_clock::now();
        grayscale(frame, grayFrame);
        auto endBaselineGray = std::chrono::high_resolution_clock::now();

        // fixed-point grayscale
        auto startFixedGray = std::chrono::high_resolution_clock::now();
        grayscaleFixedPoint(frame, grayFixedFrame);
        auto endFixedGray = std::chrono::high_resolution_clock::now();

        // baseline sobel: exact Euclidean magnitude, from baseline grayscale
        auto startBaselineSobel = std::chrono::high_resolution_clock::now();
        sobel(grayFrame, sobelFrame);
        auto endBaselineSobel = std::chrono::high_resolution_clock::now();

        // approximate magnitude, from the SAME baseline grayscale -> isolates the magnitude approximation alone
        auto startApproxSobel = std::chrono::high_resolution_clock::now();
        sobelApprox(grayFrame, sobelApproxFrame);
        auto endApproxSobel = std::chrono::high_resolution_clock::now();

        // combined optimized pipeline: fixed-point grayscale -> approximate magnitude
        auto startCombinedSobel = std::chrono::high_resolution_clock::now();
        sobelApprox(grayFixedFrame, sobelCombinedFrame);
        auto endCombinedSobel = std::chrono::high_resolution_clock::now();

        // millisecond conversion
        double baselineGrayTime = std::chrono::duration<double, std::milli>(endBaselineGray - startBaselineGray).count();
        double fixedGrayTime = std::chrono::duration<double, std::milli>(endFixedGray - startFixedGray).count();
        double baselineSobelTime = std::chrono::duration<double, std::milli>(endBaselineSobel - startBaselineSobel).count();
        double approxSobelTime = std::chrono::duration<double, std::milli>(endApproxSobel - startApproxSobel).count();
        double combinedSobelTime = std::chrono::duration<double, std::milli>(endCombinedSobel - startCombinedSobel).count();

        totalBaselineGray += baselineGrayTime;
        totalFixedGray += fixedGrayTime;
        totalBaselineSobel += baselineSobelTime;
        totalApproxSobel += approxSobelTime;
        totalCombinedSobel += combinedSobelTime;

        // grayscale error: fixed-point vs baseline (float)
        cv::Mat grayDiff;
        cv::absdiff(grayFrame, grayFixedFrame, grayDiff);
        double frameMaxGrayError = 0.0;
        cv::minMaxLoc(grayDiff, nullptr, &frameMaxGrayError);
        double frameMeanGrayError = cv::mean(grayDiff)[0];

        maxGrayError = std::max(maxGrayError, frameMaxGrayError);
        sumGrayError += frameMeanGrayError;

        // sobel magnitude error: approx vs exact, both derived from baseline grayscale
        cv::Mat sobelDiff;
        cv::absdiff(sobelFrame, sobelApproxFrame, sobelDiff);
        double frameMaxSobelError = 0.0;
        cv::minMaxLoc(sobelDiff, nullptr, &frameMaxSobelError);
        double frameMeanSobelError = cv::mean(sobelDiff)[0];

        maxSobelError = std::max(maxSobelError, frameMaxSobelError);
        sumSobelError += frameMeanSobelError;

        frameCount++;

        std::cout << std::fixed << std::setprecision(3)
                  << "Frame " << frameCount << " | "
                  << "Baseline gray: " << baselineGrayTime << " ms | "
                  << "Fixed gray: " << fixedGrayTime << " ms | "
                  << "Baseline Sobel: " << baselineSobelTime << " ms | "
                  << "Approx Sobel: " << approxSobelTime << " ms | "
                  << "Combined Sobel: " << combinedSobelTime << " ms\n";

        cv::imshow("Video", frame);
        cv::imshow("Grayscale (float)", grayFrame);
        cv::imshow("Grayscale (fixed-point)", grayFixedFrame);
        cv::imshow("Sobel (exact)", sobelFrame);
        cv::imshow("Sobel (combined optimized)", sobelCombinedFrame);

        int key = cv::waitKey(5);
        if (key == 'q') break;
    }

    video.release();
    cv::destroyAllWindows();

    if (frameCount > 0) {
        double avgBaselineGray = totalBaselineGray / frameCount;
        double avgFixedGray = totalFixedGray / frameCount;
        double avgBaselineSobel = totalBaselineSobel / frameCount;
        double avgApproxSobel = totalApproxSobel / frameCount;
        double avgCombinedSobel = totalCombinedSobel / frameCount;

        double avgBaselineTotal = avgBaselineGray + avgBaselineSobel;
        double avgCombinedTotal = avgFixedGray + avgCombinedSobel;

        std::cout << "\nAverage processing times (per frame, over " << frameCount << " frames)\n"
                  << "-------------------------------------------------------\n"
                  << std::fixed << std::setprecision(3)
                  << "Baseline grayscale (float):          " << avgBaselineGray << " ms\n"
                  << "Fixed-point grayscale:               " << avgFixedGray << " ms\n"
                  << "Baseline Sobel (exact magnitude):    " << avgBaselineSobel << " ms\n"
                  << "Approximate-gradient Sobel:          " << avgApproxSobel << " ms\n"
                  << "Combined optimized Sobel step:       " << avgCombinedSobel << " ms\n\n"
                  << "Baseline execution time (gray + sobel):            " << avgBaselineTotal << " ms\n"
                  << "Combined optimized execution time (gray + sobel):  " << avgCombinedTotal << " ms\n"
                  << "Speedup (baseline / combined):                     " << (avgBaselineTotal / avgCombinedTotal) << "x\n\n"
                  << "Grayscale error, fixed-point vs float:\n"
                  << "  Max pixel error:  " << maxGrayError << "\n"
                  << "  Mean pixel error: " << (sumGrayError / frameCount) << "\n\n"
                  << "Sobel magnitude error, approx vs exact:\n"
                  << "  Max pixel error:  " << maxSobelError << "\n"
                  << "  Mean pixel error: " << (sumSobelError / frameCount) << "\n";
    }

    return 0;
}