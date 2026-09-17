/************************************************************
* File: lab4.cpp
*
* Description: Lab 4 experiments.
*
* Part 1: cache-locality experiment. Measures how row-major
* vs column-major traversal order affects performance on a
* 2D array across a small, medium, and large working set, to
* demonstrate spatial locality and cache-line effects before
* that reasoning gets applied to the image-processing pipeline.
*
* Part 2: Sobel pixel-access experiment. Reuses the exact
* Sobel arithmetic and border policy from lab3.cpp's
* row-pointer sobel(), and compares it against an otherwise
* identical cv::Mat::at()-based implementation, to isolate
* the cost of the pixel-access mechanism itself.
*
* Author: Dakshesh Pasala, Nickaan Jahadi
*
* Revisions:
*
************************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>

/*-----------------------------------------------------
* Function: applyOperation
*
* Description: The per-element operation performed during
* both traversal orders. Kept identical between row-major and
* column-major passes so any timing difference comes purely
* from memory access order, not from different work.
*
* param value: int: the current element value
*
* return: int: the transformed value
*-----------------------------------------------------*/
int applyOperation(int value) {
    return value * 3 + 7;
}

/*-----------------------------------------------------
* Function: rowMajorPass
*
* Description: Visits every element one row at a time,
* advancing across columns before moving to the next row.
* Consecutive accesses land on adjacent memory addresses
* (stride = 1 element), matching the physical row-major
* layout of the backing array.
*
* param data: std::vector<int>&: flat backing array, size rows*cols
* param rows: int: number of rows
* param cols: int: number of columns
*
* return: long long: checksum of all transformed values
*-----------------------------------------------------*/
long long rowMajorPass(std::vector<int>& data, int rows, int cols) {
    long long checksum = 0;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            data[idx] = applyOperation(data[idx]);
            checksum += data[idx];
        }
    }

    return checksum;
}

/*-----------------------------------------------------
* Function: columnMajorPass
*
* Description: Visits every element one column at a time,
* advancing down rows before moving to the next column.
* Consecutive accesses jump by cols elements (stride = cols),
* which crosses cache lines far more often than row-major.
*
* param data: std::vector<int>&: flat backing array, size rows*cols
* param rows: int: number of rows
* param cols: int: number of columns
*
* return: long long: checksum of all transformed values
*-----------------------------------------------------*/
long long columnMajorPass(std::vector<int>& data, int rows, int cols) {
    long long checksum = 0;

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            int idx = row * cols + col;
            data[idx] = applyOperation(data[idx]);
            checksum += data[idx];
        }
    }

    return checksum;
}

/*-----------------------------------------------------
* Function: buildInitialData
*
* Description: Fills a flat rows*cols array with a
* deterministic pattern before any timed traversal begins, so
* every trial (row-major or column-major) starts from
* identical data.
*
* param rows: int: number of rows
* param cols: int: number of columns
*
* return: std::vector<int>: initialized flat array
*-----------------------------------------------------*/
std::vector<int> buildInitialData(int rows, int cols) {
    std::vector<int> data(static_cast<size_t>(rows) * cols);

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            data[row * cols + col] = (row * 31 + col * 17 + 3) % 1000;
        }
    }

    return data;
}

/*-----------------------------------------------------
* Function: timeTraversal
*
* Description: Runs a traversal function repeats times, each
* time starting from a fresh copy of initialData so results
* are directly comparable to the other traversal order. The
* copy happens outside the timed region so only the traversal
* itself is measured.
*
* param initialData: const std::vector<int>&: starting values
* param rows: int: number of rows
* param cols: int: number of columns
* param repeats: int: number of trials to average over
* param traversal: long long(*)(std::vector<int>&, int, int): traversal function to time
* param outChecksum: long long&: receives the checksum from the last trial
*
* return: double: average time per traversal, in milliseconds
*-----------------------------------------------------*/
double timeTraversal(const std::vector<int>& initialData, int rows, int cols, int repeats,
                      long long (*traversal)(std::vector<int>&, int, int), long long& outChecksum) {
    double totalMs = 0.0;

    for (int trial = 0; trial < repeats; trial++) {
        std::vector<int> workingData = initialData;

        auto start = std::chrono::high_resolution_clock::now();
        outChecksum = traversal(workingData, rows, cols);
        auto end = std::chrono::high_resolution_clock::now();

        totalMs += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return totalMs / repeats;
}

/*-----------------------------------------------------
* Function: runExperiment
*
* Description: Runs the row-major vs column-major comparison
* for one array size, then prints the dimensions, timings,
* time per element, speedup, and checksum match.
*
* param label: std::string: name of this working-set size
* param rows: int: number of rows
* param cols: int: number of columns
* param repeats: int: trials to average over (tuned per size
* so very fast small-array passes still get a stable average)
*
* return: void
*-----------------------------------------------------*/
void runExperiment(const std::string& label, int rows, int cols, int repeats) {
    long long totalElements = static_cast<long long>(rows) * cols;
    long long totalBytes = totalElements * static_cast<long long>(sizeof(int));

    std::vector<int> initialData = buildInitialData(rows, cols);

    long long rowChecksum = 0;
    long long colChecksum = 0;

    double avgRowMs = timeTraversal(initialData, rows, cols, repeats, rowMajorPass, rowChecksum);
    double avgColMs = timeTraversal(initialData, rows, cols, repeats, columnMajorPass, colChecksum);

    double nsPerElementRow = (avgRowMs * 1e6) / totalElements;
    double nsPerElementCol = (avgColMs * 1e6) / totalElements;

    double speedup = avgColMs / avgRowMs;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n" << label << "\n";
    std::cout << std::string(label.size(), '-') << "\n";
    std::cout << "Dimensions:            " << rows << " x " << cols << "\n";
    std::cout << "Total elements:        " << totalElements << "\n";
    std::cout << "Total size:            " << totalBytes << " bytes (" << (totalBytes / 1024.0) << " KB)\n";
    std::cout << "Trials averaged:       " << repeats << "\n";
    std::cout << "Row-major time:        " << avgRowMs << " ms  (" << nsPerElementRow << " ns/element)\n";
    std::cout << "Column-major time:     " << avgColMs << " ms  (" << nsPerElementCol << " ns/element)\n";
    std::cout << "Row-major speedup:     " << speedup << "x"
              << (speedup >= 1.0 ? " (row-major faster)" : " (row-major slower)") << "\n";
    std::cout << "Checksum match:        " << (rowChecksum == colChecksum ? "MATCH" : "MISMATCH")
              << " (row=" << rowChecksum << ", col=" << colChecksum << ")\n";
}

/*-----------------------------------------------------
* Function: sobelAt
*
* Description: Computes Sobel edge magnitude using
* cv::Mat::at<uint8_t>() for every pixel access. Same
* arithmetic, row-major traversal order, and border-handling
* policy (skip the outermost ring, output is (m-2) x (n-2))
* as sobelPtr, so the only difference between the two
* implementations is the access mechanism itself. Writes into
* a caller-provided output buffer instead of allocating one.
*
* param frame: const cv::Mat&: single-channel grayscale input
* param output: cv::Mat&: pre-allocated ((m-2) x (n-2),
* CV_8UC1) destination for the edge magnitude result
*
* return: void
*-----------------------------------------------------*/
void sobelAt(const cv::Mat& frame, cv::Mat& output) {
    int m = frame.rows;
    int n = frame.cols;

    for (int row = 1; row < m - 1; row++) {
        for (int col = 1; col < n - 1; col++) {
            int topLeft     = frame.at<uint8_t>(row - 1, col - 1);
            int topMiddle   = frame.at<uint8_t>(row - 1, col);
            int topRight    = frame.at<uint8_t>(row - 1, col + 1);

            int middleLeft  = frame.at<uint8_t>(row, col - 1);
            int middleRight = frame.at<uint8_t>(row, col + 1);

            int bottomLeft   = frame.at<uint8_t>(row + 1, col - 1);
            int bottomMiddle = frame.at<uint8_t>(row + 1, col);
            int bottomRight  = frame.at<uint8_t>(row + 1, col + 1);

            int gx = -topLeft + topRight - 2 * middleLeft + 2 * middleRight - bottomLeft + bottomRight;
            int gy = -topLeft - 2 * topMiddle - topRight + bottomLeft + 2 * bottomMiddle + bottomRight;

            double magnitude = std::sqrt(gx * gx + gy * gy);

            output.at<uint8_t>(row - 1, col - 1) = cv::saturate_cast<uint8_t>(magnitude);
        }
    }
}

/*-----------------------------------------------------
* Function: sobelPtr
*
* Description: Computes Sobel edge magnitude using a row
* pointer obtained once per row via cv::Mat::ptr<>(), then
* indexed with plain pointer arithmetic. Same arithmetic,
* row-major traversal order, and border-handling policy as
* sobelAt (this is the same implementation as sobel() in
* lab3.cpp, carried over with the same buffer-reuse signature).
* Writes into a caller-provided output buffer instead of
* allocating one.
*
* param frame: const cv::Mat&: single-channel grayscale input
* param output: cv::Mat&: pre-allocated ((m-2) x (n-2),
* CV_8UC1) destination for the edge magnitude result
*
* return: void
*-----------------------------------------------------*/
void sobelPtr(const cv::Mat& frame, cv::Mat& output) {
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

            double magnitude = std::sqrt(gx * gx + gy * gy);

            newRow[j - 1] = cv::saturate_cast<uchar>(magnitude);
        }
    }
}

/*-----------------------------------------------------
* Function: buildTestImage
*
* Description: Fills a rows x cols single-channel image with
* a deterministic pattern before any timed Sobel pass begins,
* so both access-mechanism trials run on identical input.
*
* param rows: int: image height
* param cols: int: image width
*
* return: cv::Mat: initialized CV_8UC1 test image
*-----------------------------------------------------*/
cv::Mat buildTestImage(int rows, int cols) {
    cv::Mat image(rows, cols, CV_8UC1);

    for (int row = 0; row < rows; row++) {
        uchar* imageRow = image.ptr<uchar>(row);

        for (int col = 0; col < cols; col++) {
            imageRow[col] = static_cast<uchar>((row * 31 + col * 17 + 3) % 256);
        }
    }

    return image;
}

/*-----------------------------------------------------
* Function: timeSobel
*
* Description: Runs a Sobel implementation repeats times over
* the same test image and averages the elapsed time. Image
* creation and outResult's allocation both happen before this
* function is called (and outResult is reused across every
* trial), so only the Sobel stage itself is timed.
*
* param testImage: const cv::Mat&: shared input image
* param repeats: int: number of trials to average over
* param sobelFn: void(*)(const cv::Mat&, cv::Mat&): Sobel implementation to time
* param outResult: cv::Mat&: pre-allocated buffer reused for every trial
*
* return: double: average time per call, in milliseconds
*-----------------------------------------------------*/
double timeSobel(const cv::Mat& testImage, int repeats,
                  void (*sobelFn)(const cv::Mat&, cv::Mat&), cv::Mat& outResult) {
    double totalMs = 0.0;

    for (int trial = 0; trial < repeats; trial++) {
        auto start = std::chrono::high_resolution_clock::now();
        sobelFn(testImage, outResult);
        auto end = std::chrono::high_resolution_clock::now();

        totalMs += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return totalMs / repeats;
}

/*-----------------------------------------------------
* Function: runSobelAccessExperiment
*
* Description: Runs the .at() vs row-pointer Sobel comparison
* at one image resolution, then prints the timing table and
* the correctness check (differing pixels, max absolute
* difference, bit-for-bit match).
*
* param label: std::string: name of this resolution
* param rows: int: image height
* param cols: int: image width
* param repeats: int: trials to average over (tuned per
* resolution so large images still finish in reasonable time)
*
* return: void
*-----------------------------------------------------*/
void runSobelAccessExperiment(const std::string& label, int rows, int cols, int repeats) {
    cv::Mat testImage = buildTestImage(rows, cols);
    long long totalPixels = static_cast<long long>(rows - 2) * (cols - 2);

    // allocated once, before any timed call, and reused across every trial
    cv::Mat atResult(rows - 2, cols - 2, CV_8UC1, cv::Scalar(0));
    cv::Mat ptrResult(rows - 2, cols - 2, CV_8UC1, cv::Scalar(0));

    double avgAtMs = timeSobel(testImage, repeats, sobelAt, atResult);
    double avgPtrMs = timeSobel(testImage, repeats, sobelPtr, ptrResult);

    double nsPerPixelAt = (avgAtMs * 1e6) / totalPixels;
    double nsPerPixelPtr = (avgPtrMs * 1e6) / totalPixels;

    double relativePerformance = avgAtMs / avgPtrMs;

    cv::Mat diff;
    cv::absdiff(atResult, ptrResult, diff);
    int differingPixels = cv::countNonZero(diff);
    double maxDiff = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxDiff);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n" << label << "\n";
    std::cout << std::string(label.size(), '-') << "\n";
    std::cout << "Image size:              " << cols << " x " << rows << "  (" << totalPixels << " output pixels)\n";
    std::cout << "Trials averaged:         " << repeats << "\n";
    std::cout << "cv::Mat::at() time:      " << avgAtMs << " ms  (" << nsPerPixelAt << " ns/pixel)  [1.0000x]\n";
    std::cout << "Row-pointer time:        " << avgPtrMs << " ms  (" << nsPerPixelPtr << " ns/pixel)  ["
              << relativePerformance << "x]\n";
    std::cout << "Differing pixels:        " << differingPixels << " / " << totalPixels << "\n";
    std::cout << "Max absolute difference: " << maxDiff << "\n";
    std::cout << "Bit-for-bit identical:   " << (differingPixels == 0 ? "YES" : "NO") << "\n";
}

/*-----------------------------------------------------
* Function: grayscaleBaseline
*
* Description: Same BT.709 grayscale arithmetic as
* grayscale() in lab3.cpp, materializing a complete
* single-channel output frame. Serves as the ground truth
* that sobelStreaming's fused output is checked against.
* Writes into a caller-provided output buffer instead of
* allocating one.
*
* param frame: const cv::Mat&: BGR input image
* param output: cv::Mat&: pre-allocated (m x n, CV_8UC1)
* destination for the grayscale result
*
* return: void
*-----------------------------------------------------*/
void grayscaleBaseline(const cv::Mat& frame, cv::Mat& output) {
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
        }
    }
}

/*-----------------------------------------------------
* Function: convertRowToGrayscale
*
* Description: Converts one row of a BGR image to grayscale,
* using the same arithmetic as grayscaleBaseline, but writes
* into a caller-provided row buffer instead of a full-frame
* Mat.
*
* param frame: const cv::Mat&: BGR input image
* param rowIndex: int: which row of frame to convert
* param outRow: uchar*: destination buffer, must hold cols bytes
* param cols: int: number of columns
*
* return: void
*-----------------------------------------------------*/
void convertRowToGrayscale(const cv::Mat& frame, int rowIndex, uchar* outRow, int cols) {
    const cv::Vec3b* colorRow = frame.ptr<cv::Vec3b>(rowIndex);

    for (int col = 0; col < cols; col++) {
        const cv::Vec3b& pixel = colorRow[col];
        unsigned char blue = pixel[0];
        unsigned char green = pixel[1];
        unsigned char red = pixel[2];

        outRow[col] = cv::saturate_cast<unsigned char>(0.2126 * red + 0.7152 * green + 0.0722 * blue);
    }
}

/*-----------------------------------------------------
* Function: sobelStreaming
*
* Description: Streaming Sobel implementation that never
* materializes a full grayscale frame. Only three grayscale
* rows are kept at once, held in three fixed row buffers that
* rotate logically (buffer index = rowIndex % 3) rather than
* being copied. Same arithmetic, row-major order, and border
* policy as grayscaleBaseline + sobelPtr, so the output must
* match theirs exactly. Both the output image and the three
* row buffers are caller-provided so nothing is allocated
* inside this function.
*
* param frame: const cv::Mat&: BGR input image
* param output: cv::Mat&: pre-allocated ((m-2) x (n-2),
* CV_8UC1) destination for the edge magnitude result
* param rowBuffer: std::vector<uchar>[3]: three pre-sized
* (cols bytes each) row buffers, reused across every call
*
* return: void
*-----------------------------------------------------*/
void sobelStreaming(const cv::Mat& frame, cv::Mat& output, std::vector<uchar> rowBuffer[3]) {
    int m = frame.rows;
    int n = frame.cols;

    // prime the window: the loop below converts row (i+1) fresh on every
    // pass, so rows 0 and 1 need to already be sitting in their buffers
    // before the first iteration
    convertRowToGrayscale(frame, 0, rowBuffer[0].data(), n);
    convertRowToGrayscale(frame, 1, rowBuffer[1].data(), n);

    for (int i = 1; i < m - 1; i++) {
        int topIndex    = (i - 1) % 3;
        int middleIndex = i % 3;
        int bottomIndex = (i + 1) % 3;

        // bottomIndex's slot last held row (i+1)-3 = i-2, which finished
        // being used as topRow at the end of the previous iteration, so
        // it is safe to overwrite with row i+1 here
        convertRowToGrayscale(frame, i + 1, rowBuffer[bottomIndex].data(), n);

        uchar* topRow    = rowBuffer[topIndex].data();
        uchar* middleRow = rowBuffer[middleIndex].data();
        uchar* bottomRow = rowBuffer[bottomIndex].data();
        uchar* newRow    = output.ptr<uchar>(i - 1);

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

            double magnitude = std::sqrt(gx * gx + gy * gy);

            newRow[j - 1] = cv::saturate_cast<uchar>(magnitude);
        }
    }
}

/*-----------------------------------------------------
* Function: buildColorTestImage
*
* Description: Fills a rows x cols BGR image with a
* deterministic (but non-constant per channel) pattern, used
* as the shared input for the baseline vs streaming Sobel
* comparison.
*
* param rows: int: image height
* param cols: int: image width
*
* return: cv::Mat: initialized CV_8UC3 test image
*-----------------------------------------------------*/
cv::Mat buildColorTestImage(int rows, int cols) {
    cv::Mat image(rows, cols, CV_8UC3);

    for (int row = 0; row < rows; row++) {
        cv::Vec3b* imageRow = image.ptr<cv::Vec3b>(row);

        for (int col = 0; col < cols; col++) {
            uchar blue  = static_cast<uchar>((row * 31 + col * 17 + 3) % 256);
            uchar green = static_cast<uchar>((row * 13 + col * 29 + 7) % 256);
            uchar red   = static_cast<uchar>((row * 19 + col * 23 + 11) % 256);

            imageRow[col] = cv::Vec3b(blue, green, red);
        }
    }

    return image;
}

/*-----------------------------------------------------
* Function: runStreamingExperiment
*
* Description: Runs the baseline (full grayscale frame, then
* sobelPtr) pipeline against the streaming (three rotating
* row buffers) pipeline at one resolution, then prints timing
* and the correctness check (differing pixels, max absolute
* difference, bit-for-bit match).
*
* param label: std::string: name of this resolution
* param rows: int: image height
* param cols: int: image width
* param repeats: int: trials to average over
*
* return: void
*-----------------------------------------------------*/
void runStreamingExperiment(const std::string& label, int rows, int cols, int repeats) {
    cv::Mat colorImage = buildColorTestImage(rows, cols);
    long long totalPixels = static_cast<long long>(rows - 2) * (cols - 2);

    // every buffer below is allocated once, before any timed call, and
    // reused across every trial in the loop -- nothing is allocated inside
    // grayscaleBaseline, sobelPtr, or sobelStreaming anymore
    cv::Mat baselineGray(rows, cols, CV_8UC1);
    cv::Mat baselineSobel(rows - 2, cols - 2, CV_8UC1, cv::Scalar(0));
    cv::Mat streamingResult(rows - 2, cols - 2, CV_8UC1, cv::Scalar(0));

    std::vector<uchar> rowBuffer[3];
    rowBuffer[0].resize(cols);
    rowBuffer[1].resize(cols);
    rowBuffer[2].resize(cols);

    double totalBaselineMs = 0.0;
    double totalStreamingMs = 0.0;

    for (int trial = 0; trial < repeats; trial++) {
        auto startBaseline = std::chrono::high_resolution_clock::now();
        grayscaleBaseline(colorImage, baselineGray);
        sobelPtr(baselineGray, baselineSobel);
        auto endBaseline = std::chrono::high_resolution_clock::now();
        totalBaselineMs += std::chrono::duration<double, std::milli>(endBaseline - startBaseline).count();

        auto startStreaming = std::chrono::high_resolution_clock::now();
        sobelStreaming(colorImage, streamingResult, rowBuffer);
        auto endStreaming = std::chrono::high_resolution_clock::now();
        totalStreamingMs += std::chrono::duration<double, std::milli>(endStreaming - startStreaming).count();
    }

    double avgBaselineMs = totalBaselineMs / repeats;
    double avgStreamingMs = totalStreamingMs / repeats;

    double nsPerPixelBaseline = (avgBaselineMs * 1e6) / totalPixels;
    double nsPerPixelStreaming = (avgStreamingMs * 1e6) / totalPixels;

    double streamingSpeedup = avgBaselineMs / avgStreamingMs;

    cv::Mat diff;
    cv::absdiff(baselineSobel, streamingResult, diff);
    int differingPixels = cv::countNonZero(diff);
    double maxDiff = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxDiff);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nResults table for streaming grayscale -- " << label
               << "  (" << cols << " x " << rows << ", " << totalPixels << " output pixels, "
               << repeats << " trials averaged)\n";

    std::cout << std::left
               << std::setw(18) << "Implementation"
               << std::setw(28) << "Intermediate storage"
               << std::right
               << std::setw(16) << "Execution time"
               << std::setw(18) << "Time per pixel"
               << std::setw(12) << "Speedup" << "\n";

    std::cout << std::left
               << std::setw(18) << "Baseline"
               << std::setw(28) << "Complete grayscale frame"
               << std::right
               << std::setw(13) << avgBaselineMs << " ms"
               << std::setw(15) << nsPerPixelBaseline << " ns"
               << std::setw(11) << 1.0 << "x" << "\n";

    std::cout << std::left
               << std::setw(18) << "Streaming"
               << std::setw(28) << "Three grayscale rows"
               << std::right
               << std::setw(13) << avgStreamingMs << " ms"
               << std::setw(15) << nsPerPixelStreaming << " ns"
               << std::setw(11) << streamingSpeedup << "x" << "\n";

    std::cout << "Differing pixels:        " << differingPixels << " / " << totalPixels << "\n";
    std::cout << "Max absolute difference: " << maxDiff << "\n";
    std::cout << "Bit-for-bit identical:   " << (differingPixels == 0 ? "YES" : "NO") << "\n";
}

/*-----------------------------------------------------
* Function: main
*
* Description: Runs the Part 1 row-major vs column-major
* cache locality experiment, the Part 2 .at() vs row-pointer
* Sobel pixel-access experiment, and the Part 3 full-frame vs
* streaming (3-row-buffer) Sobel experiment, each across a
* small, medium, and large working set / resolution.
*
* return: int: always 0
*-----------------------------------------------------*/
int main() {
    std::cout << "=== Part 1: Array Traversal Order Experiment ===\n";

    // small: comfortably fits inside L1 cache on essentially any modern processor,
    // so row-major and column-major should perform about the same here
    runExperiment("Small (16 x 16, 256 elements)", 16, 16, 100000);

    // medium: sized to sit around the L1/L2 boundary, where column-major should
    // start showing a measurable but not extreme slowdown
    runExperiment("Medium (100 x 100, 10,000 elements)", 100, 100, 2000);

    // large: far exceeds L2 and typically L3 too, so column-major should be
    // clearly and consistently slower due to cache misses on almost every access
    runExperiment("Large (4096 x 4096, 16,777,216 elements)", 4096, 4096, 5);

    std::cout << "\n\n=== Part 2: Sobel Pixel-Access Method Experiment ===\n";

    // 320x240 (QVGA): small enough that per-call overhead dominates over cache effects
    runSobelAccessExperiment("320 x 240 (QVGA)", 240, 320, 100);

    // 1280x720 (HD): mid-size frame, representative of a typical webcam/video resolution
    runSobelAccessExperiment("1280 x 720 (HD)", 720, 1280, 20);

    // 3840x2160 (4K): large enough to stress both access overhead and cache behavior
    runSobelAccessExperiment("3840 x 2160 (4K)", 2160, 3840, 5);

    std::cout << "\n\n=== Part 3: Full-Frame vs Streaming (3-Row) Sobel Experiment ===\n";

    runStreamingExperiment("320 x 240 (QVGA)", 240, 320, 100);
    runStreamingExperiment("1280 x 720 (HD)", 720, 1280, 20);
    runStreamingExperiment("3840 x 2160 (4K)", 2160, 3840, 5);

    return 0;
}
