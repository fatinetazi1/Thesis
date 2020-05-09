#include "DGM.h"
#include "DGM/timer.h"

#include "PFM.h"

using namespace DirectGraphicalModels;

void print_help(char *argv0) {
    printf("Usage: %s left_image right_image min_disparity max_disparity right_image_groundtruth output_disparity\n", argv0);
}

float meanAbs(Mat solution, Mat gt, float scaleFactor) {
    float sum = 0;
    for (int y = 0; y < solution.rows; y++) {
        const byte *pM1 = solution.ptr<byte>(y);
        const byte *pM2 = gt.ptr<byte>(y);
        for (int x = 0; x < solution.cols; x++){
            float percentage = pM2[x] ? abs(pM1[x] - pM2[x])/pM2[x] : abs(pM1[x] - pM2[x]);
            sum += percentage;
        }
    }
    return sum / (solution.rows*solution.cols);
}

float badPixel(Mat solution, Mat gt, float scaleFactor) {
    float threshold = 1;
    float sum = 0;
    for (int y = 0; y < solution.rows; y++) {
        const byte *pM1 = solution.ptr<byte>(y);
        const byte *pM2 = gt.ptr<byte>(y);
        for (int x = 0; x < solution.cols; x++){
            float difference = abs(pM1[x] - (pM2[x]));
            if (difference >= threshold) sum++;
        }
    }
    return sum / (solution.rows*solution.cols);
}

int main(int argc, char *argv[]) {
    
    if (argc != 7) {
        print_help(argv[0]);
        return 0;
    }
    
    const int gtScaleFactor = 6;

    // Reading parameters and images
    Mat imgL = imread(argv[1], 0);          if (imgL.empty()) printf("Can't open %s\n", argv[1]);
    resize(imgL, imgL, Size(imgL.cols / gtScaleFactor, imgL.rows / gtScaleFactor));
    
    Mat imgR = imread(argv[2], 0);          if (imgR.empty()) printf("Can't open %s\n", argv[2]);
    int rows = imgR.rows;
    int cols = imgR.cols;
    resize(imgR, imgR, Size(imgR.cols / gtScaleFactor, imgR.rows / gtScaleFactor));
    
    
    int       minDisparity  = atoi(argv[3]);
    int       maxDisparity  = atoi(argv[4]);
    int       width         = imgL.cols;
    int       height        = imgL.rows;
    unsigned int nStates    = maxDisparity - minDisparity;
    
    const int assertScaleFactor   = 255/nStates;
    
    PFM imgR_pfm;
    float* pimgR_pfm = imgR_pfm.read_pfm<float>(argv[5]);
    Mat imgR_gt = Mat(rows, cols, CV_32FC1, pimgR_pfm);
    resize(imgR_gt, imgR_gt, Size(imgR_gt.cols / gtScaleFactor, imgR_gt.rows / gtScaleFactor));
    
    CGraphPairwiseKit graphKit(nStates, INFER::TRW);
//    CGraphDenseKit graphKit(nStates); // Uncomment for Dense graph model, comment line above.
    
    graphKit.getGraphExt().buildGraph(imgL.size());
    graphKit.getGraphExt().addDefaultEdgesModel(1.175f);
    
    // ==================== Filling the nodes of the graph ====================
    Mat nodePot(nStates, 1, CV_32FC1);                                      // node Potential (column-vector)
    size_t idx = 0;
    for (int y = 0; y < height; y++) {
        byte * pImgL    = imgL.ptr<byte>(y);
        byte * pImgR    = imgR.ptr<byte>(y);
        for (int x = 0; x < width; x++) {
            float imgL_value = static_cast<float>(pImgL[x]);
            for (unsigned int s = 0; s < nStates; s++) {                    // state
                int disparity = minDisparity + s;
                float imgR_value = (x + disparity < width) ? static_cast<float>(pImgR[x + disparity]) : imgL_value;
                float p = 1.0f - fabs(imgL_value - imgR_value) / 255.0f;
                nodePot.at<float>(s, 0) = p * p;
            }
            graphKit.getGraph().setNode(idx++, nodePot);
        } // x
    } // y
    
    // =============================== Decoding ===============================
    Timer::start("Decoding... ");
    vec_byte_t optimalDecoding = graphKit.getInfer().decode(10);
    Timer::stop();
    
    // ============================ Evaluation =============================
    Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
    disparity = (disparity + minDisparity) * (256 / maxDisparity);
    medianBlur(disparity, disparity, 3);
    
    float meanError = meanAbs(disparity, imgR_gt, assertScaleFactor);
    float badError = badPixel(disparity, imgR_gt,  assertScaleFactor);
    
    // ============================ Visualization =============================
    char error_str[255];
    sprintf(error_str, "Mean abs error: %.2f | Bad pixel: %.2f", meanError, badError);
    putText(disparity, error_str, Point(width - 300, height - 25), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1, cv::LineTypes::LINE_AA);
    
    char disp_str[255];
    sprintf(disp_str, "Min-disparity: %d | Max-disparity: %d", minDisparity, maxDisparity);
    putText(disparity, disp_str, Point(width - 290, height - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1, cv::LineTypes::LINE_AA);
    
    imwrite(argv[6], disparity);
    imshow("Disparity", disparity);
    waitKey();

    return 0;
}
