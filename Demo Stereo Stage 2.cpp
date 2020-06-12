#include "DGM.h"
#include "DGM/timer.h"
#include "PFM.h"
#include "FEX.h"

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
    
    const int useColor = 1;

    if (argc != 7) {
        print_help(argv[0]);
        return 0;
    }

    // Reading parameters and images
    Mat imgR = imread(argv[1], useColor);          if (imgR.empty()) printf("Can't open %s\n", argv[1]);
    Mat imgL = imread(argv[2], useColor);          if (imgL.empty()) printf("Can't open %s\n", argv[2]);
    
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
    
    // Extracted SIFT Features
    Mat siftR = fex::CSIFT::get(imgR);
    Mat siftL = fex::CSIFT::get(imgL);

    vec_mat_t fexR;
    vec_mat_t fexL;
    split(siftR, fexR);
    split(siftL, fexL);
    
    int       height        = imgL.rows;
    int       width         = imgL.cols;
    int       rgbChannels   = imgL.channels();
    int       siftChannels  = siftL.channels();
    const float rgbW        = 0.88f;
    const float siftW       = 1.00f - rgbW;
    float     edgeParam     = 1.175f;
    
    // Preparing parameters for parameter estimation
    vec_float_t   vParams = {edgeParam, rgbW, siftW};
    vec_float_t   vDeltas = {0.05f,  0.01f, 0.01f};
    
    // Initializing Powell search class and parameters
    CPowell powell(vParams.size());
    powell.setInitParams(vParams);
    powell.setDeltas(vDeltas);
    
    CGraphPairwiseKit graphKit(nStates, INFER::TRW);
    vec_byte_t optimalDecoding;
    
    // Main loop of parameters optimization
    for (int i = 1; ; i++) {
        // ==================== Graph building ====================
        graphKit.getGraphExt().buildGraph(imgL.size());
        graphKit.getGraphExt().addDefaultEdgesModel(vParams[0]);
        
        // ==================== Filling the nodes of the graph ====================
        Mat nodePot(nStates, 1, CV_32FC1);                                      // node Potential (column-vector)
        size_t idx = 0;
        for (int y = 0; y < height; y++) {
            byte* pImgR = imgR.ptr<byte>(y);
            byte* pImgL = imgL.ptr<byte>(y);
            byte* pSiftR = siftR.ptr<byte>(y);
            byte* pSiftL = siftL.ptr<byte>(y);
            for (int x = 0; x < width; x++) {
                // -------------------- RGB data --------------------
                vec_float_t sum_rgb(nStates, 0);
                for (int ch = 0; ch < rgbChannels; ch++) {                      // channel
                    float imgR_value = static_cast<float>(pImgR[rgbChannels * x + ch]);
                    for (unsigned int s = 0; s < nStates; s++) {                // state
                        int disparity = minDisparity + s;
                        float imgL_value = (x - disparity >= 0) ? static_cast<float>(pImgL[rgbChannels * (x - disparity) + ch]) : static_cast<float>(pImgL[rgbChannels * x + ch]);
                        sum_rgb[s] += fabs(imgL_value - imgR_value) / 255.0f;
                    } // s
                } // ch
                
                // -------------------- SIFT data --------------------
                vec_float_t sum_sift(nStates, 0);
                for (int ch = 0; ch < siftChannels; ch++) {                      // channel
                    float siftR_value = static_cast<float>(pSiftR[siftChannels * x + ch]);
                    for (unsigned int s = 0; s < nStates; s++) {                 // state
                        int disparity = minDisparity + s;
                        float siftL_value = (x - disparity >= 0) ? static_cast<float>(pSiftL[siftChannels * (x - disparity) + ch]) : static_cast<float>(pSiftL[siftChannels * x + ch]);
                        sum_sift[s] += fabs(siftL_value - siftR_value) / 255.0f;
                    } // s
                } // ch
                
                // -------------------- Potential calculation --------------------
                for (unsigned int s = 0; s < nStates; s++) {
                    float p_rgb  = 1.0f - sum_rgb[s] / rgbChannels;
                    float p_sift = 1.0f - sum_sift[s] / siftChannels;
                    float p = (vParams[1] * p_rgb) + (vParams[2] * p_sift);
                    nodePot.at<float>(s, 0) = p * p;
                }
                graphKit.getGraph().setNode(idx++, nodePot);
            } // x
        } // y

        // =============================== Decoding ===============================
        Timer::start("Decoding... ");
        optimalDecoding = graphKit.getInfer().decode(10);
        Timer::stop();
        
        // ====================== Powell Evaluation =======================
        Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());  // the values are [0; nStates] = [0; maxDisp - minDisp]
        disparity = disparity + minDisparity;
        
        Mat gt;                                                        // the values are [minDisp; maxDisp]
        imgR_gt.convertTo(gt, CV_8UC1, 1.0 / gtScaleFactor);
        
        float badError = badPixel(disparity, gt);
        float accuracy = 100 - badError;
        
        printf("Iteration: %d, parameters: { ", i);
        for (const float& param : vParams) printf("%.1f ", param);
        printf("}, accuracy: %.2f%%\n", accuracy);

        if (powell.isConverged()) break;
        vParams = powell.getParams(accuracy);
    }
    

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

