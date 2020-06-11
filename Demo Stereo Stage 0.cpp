#include "DGM.h"
#include "DGM/timer.h"
#include "PFM.h"

using namespace DirectGraphicalModels;

void print_help(char* argv0) {
    printf("Usage: %s right_image left_image min_disparity max_disparity right_image_groundtruth output_disparity\n", argv0);
}

float meanAbs(Mat solution, Mat gt) {
    float sum = 0;
    for (int y = 0; y < solution.rows; y++) {
        const byte* pSolution = solution.ptr<byte>(y);
        const byte* pGT = gt.ptr<byte>(y);
        for (int x = 0; x < solution.cols; x++) {
            float solution_val = static_cast<float>(pSolution[x]);
            float gt_val = static_cast<float>(pGT[x]);
            float percentage = (solution_val - gt_val) * (solution_val - gt_val);
            sum += percentage;
        }
    }
    return sum / (solution.rows * solution.cols);
}

float badPixel(const Mat& solution, const Mat& gt) {
    Mat clone = solution.clone();
    cv::cvtColor(clone, clone, cv::COLOR_GRAY2BGR);

    const float threshold = 1;
    float sum = 0;
    int pixels = 0;
    for (int y = 0; y < solution.rows; y++) {
        const byte* pSolution = solution.ptr<byte>(y);
        const byte* pGT = gt.ptr<byte>(y);
        for (int x = 0; x < solution.cols; x++) {
            if (pGT[x] == 0) continue;  // if the gt has no data -> do not consider it
            float solution_val = static_cast<float>(pSolution[x]);
            float gt_val = static_cast<float>(pGT[x]);
            float difference = fabs(solution_val - gt_val);
            if (difference >= threshold) {
                Vec3b error = clone.at<Vec3b>(Point(x, y));
                error[0] = 0;
                error[1] = 0;
                error[2] = 255;
                clone.at<Vec3b>(Point(x, y)) = error;
                sum++;
            }
            pixels++;
        }
    }
    imshow("Solution", solution);
    waitKey();
    imshow("Error", clone);
    waitKey();

    return 100 * sum / pixels;
}

void compare_images(const Mat& img1, const Mat& img2)
{
    for (size_t i = 0; ; i++) {
        imshow("comparing", i % 2 == 0 ? img1 : img2);
        int key = waitKey();
        if (key == 'q') break;
    }
}

int main(int argc, char* argv[]) {

    if (argc != 7) {
        print_help(argv[0]);
        return 0;
    }

    // Reading parameters and images
    Mat imgR = imread(argv[1], 0);          if (imgR.empty()) printf("Can't open %s\n", argv[1]);
    Mat imgL = imread(argv[2], 0);          if (imgL.empty()) printf("Can't open %s\n", argv[2]);
    
    int       minDisparity = atoi(argv[3]);
    int       maxDisparity = atoi(argv[4]);
    unsigned int nStates = maxDisparity - minDisparity;

    PFM imgR_pfm;
    float* pimgR_pfm = imgR_pfm.read_pfm<float>(argv[5]);
    Mat imgR_gt = Mat(imgR.size(), CV_32FC1, pimgR_pfm);
    
    // Scaling
    const int gtScaleFactor = 6;
    resize(imgL, imgL, Size(imgL.cols / gtScaleFactor, imgL.rows / gtScaleFactor)); //byte
    resize(imgR, imgR, Size(imgR.cols / gtScaleFactor, imgR.rows / gtScaleFactor)); //byte
    resize(imgR_gt, imgR_gt, Size(imgR_gt.cols / gtScaleFactor, imgR_gt.rows / gtScaleFactor)); //floating pont
    
    int       height        = imgL.rows;
    int       width         = imgL.cols;

    CGraphPairwiseKit graphKit(nStates, INFER::TRW);

    graphKit.getGraphExt().buildGraph(imgL.size());
    graphKit.getGraphExt().addDefaultEdgesModel(1.175f);

    // ==================== Filling the nodes of the graph ====================
    Mat nodePot(nStates, 1, CV_32FC1);                                        // node Potential (column-vector)
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
    Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());  // the values are [0; nStates] = [0; maxDisp - minDisp]
    disparity = disparity + minDisparity;
    Mat gt;                                                        // the values are [minDisp; maxDisp]
    imgR_gt.convertTo(gt, CV_8UC1, 1.0 / gtScaleFactor);

    float meanError = meanAbs(disparity, gt);
    float badError = badPixel(disparity, gt);

    // ============================ Visualization =============================
    disparity = disparity * (256 / maxDisparity);
    //disparity = disparity * gtScaleFactor;
    medianBlur(disparity, disparity, 3);

    char error_str[255];
    sprintf(error_str, "MSE: %.2f | Bad pixel percentage: %.2f %%", meanError, badError);
    putText(disparity, error_str, Point(width - 320, height - 25), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1, cv::LineTypes::LINE_AA);

    char disp_str[255];
    sprintf(disp_str, "Min-disparity: %d | Max-disparity: %d", minDisparity, maxDisparity);
    putText(disparity, disp_str, Point(width - 290, height - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1, cv::LineTypes::LINE_AA);

    imshow("Disparity", disparity);
    waitKey();
    imwrite(argv[6], disparity);

    return 0;
}
