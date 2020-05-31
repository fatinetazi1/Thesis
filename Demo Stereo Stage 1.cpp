#include "DGM.h"
#include "DGM/timer.h"
#include "PFM.h"

using namespace DirectGraphicalModels;

void print_help(char* argv0) {
    printf("Usage: %s left_image right_image min_disparity max_disparity node_norm_function right_image_groundtruth image_features_folder output_disparity\n", argv0);
    
    printf("\nNode norm function:\n");
    printf("0: Absolute value norm\n");
    printf("1: Manhattan norm\n");
    printf("2: Euclidean norm\n");
    printf("3: P-norm\n");
}

float absolute_norm(Vec3b imgL_values, Vec3b imgR_values) {
    float blue = abs(imgL_values[0] - imgR_values[0]);
    float green = abs(imgL_values[1] - imgR_values[1]);
    float red = abs(imgL_values[2] - imgR_values[2]);
    float sum = blue + green + red;
    return sum/3;
}

float manhattan_norm(Vec3b imgL_values, Vec3b imgR_values) {
    float blue = abs(imgL_values[0] - imgR_values[0]);
    float green = abs(imgL_values[1] - imgR_values[1]);
    float red = abs(imgL_values[2] - imgR_values[2]);
    float sum = blue + green + red;
    return sum;
}

float euclidean_norm(Vec3b imgL_values, Vec3b imgR_values) {
    float blue = pow(abs(imgL_values[0] - imgR_values[0]), 2);
    float green = pow(abs(imgL_values[1] - imgR_values[1]), 2);
    float red = pow(abs(imgL_values[2] - imgR_values[2]), 2);
    float sum = blue + green + red;
    return sqrt(sum);
}

float p_norm(Vec3b imgL_values, Vec3b imgR_values, int p) {
    float blue = pow(abs(imgL_values[0] - imgR_values[0]), p);
    float green = pow(abs(imgL_values[1] - imgR_values[1]), p);
    float red = pow(abs(imgL_values[2] - imgR_values[2]), p);
    float sum = blue + green + red;
    return pow(sum, 1/p);
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

float badPixel(Mat solution, Mat gt) {
    const float threshold = 1;
    float sum = 0;
    for (int y = 0; y < solution.rows; y++) {
        const byte* pSolution = solution.ptr<byte>(y);
        const byte* pGT = gt.ptr<byte>(y);
        for (int x = 0; x < solution.cols; x++) {
            float solution_val = static_cast<float>(pSolution[x]);
            float gt_val = static_cast<float>(pGT[x]);
            float difference = fabs(solution_val - gt_val);
            if (difference >= threshold) sum++;
        }
    }
    return 100 * sum / (solution.rows * solution.cols);
}

int main(int argc, char* argv[]) {

    if (argc != 9) {
        print_help(argv[0]);
        return 0;
    }

    // Reading parameters and images
    Mat imgL = imread(argv[1], 0);          if (imgL.empty()) printf("Can't open %s\n", argv[1]);
    Mat imgR = imread(argv[2], 0);          if (imgR.empty()) printf("Can't open %s\n", argv[2]);
    
    int       minDisparity = atoi(argv[3]);
    int       maxDisparity = atoi(argv[4]);
    int       nodeNorm     = atoi(argv[5]);
    unsigned int nStates = maxDisparity - minDisparity;

    PFM imgR_pfm;
    float* pimgR_pfm = imgR_pfm.read_pfm<float>(argv[6]);
    Mat imgR_gt = Mat(imgR.size(), CV_32FC1, pimgR_pfm);
    
    // Scaling
    const int gtScaleFactor = 6;
    resize(imgL, imgL, Size(imgL.cols / gtScaleFactor, imgL.rows / gtScaleFactor)); //byte
    resize(imgR, imgR, Size(imgR.cols / gtScaleFactor, imgR.rows / gtScaleFactor)); //byte
    resize(imgR_gt, imgR_gt, Size(imgR_gt.cols / gtScaleFactor, imgR_gt.rows / gtScaleFactor)); //floating pont

    int       height = imgL.rows;
    int       width = imgL.cols;
    
    // Extracted SIFT Features
    const int features = 128;
    Mat fexL[features];
    Mat fexR[features];
    for (size_t ch = 0; ch < features; ch++) {
        fexL[ch] = imread(std::string(argv[7]) + std::string("fexL") + std::to_string(ch) +  std::string(".png"));
        fexR[ch] = imread(std::string(argv[7]) + std::string("fexR") + std::to_string(ch) +  std::string(".png"));
    }

    CGraphPairwiseKit graphKit(nStates, INFER::TRW);

    graphKit.getGraphExt().buildGraph(imgL.size());
    graphKit.getGraphExt().addDefaultEdgesModel(1.175f);

    // ==================== Filling the nodes of the graph ====================
    Mat nodePot(nStates, 1, CV_32FC1);                                      // node Potential (column-vector)
    size_t idx = 0;
    float p, p1, p2;
    int p_value = 0;                                                        // user input
    if (nodeNorm == 3) {
        printf( "Enter a p-value : ");
        scanf("%d", &p_value);
    }
    for (int y = 0; y < height; y++) {
        byte* pImgL = imgL.ptr<byte>(y);
        byte* pImgR = imgR.ptr<byte>(y);
        for (int x = 0; x < width; x++) {
            Vec3b imgL_values = static_cast<Vec3b>(pImgL[x]);
            for (unsigned int s = 0; s < nStates; s++) {                    // state
                int disparity = minDisparity + s;
                Vec3b imgR_values = (x + disparity < width) ? static_cast<Vec3b>(pImgR[x + disparity]) : imgL_values;
                switch (nodeNorm) {
                    case 0:
                        p1 = 1.0f - absolute_norm(imgL_values, imgR_values) / 255.0f;
                        break;
                    case 1:
                        p1 = 1.0f - manhattan_norm(imgL_values, imgR_values) / 255.0f;
                        break;
                    case 2:
                        p1 = 1.0f - euclidean_norm(imgL_values, imgR_values) / 255.0f;
                        break;
                    case 3:
                        p1 = 1.0f - p_norm(imgL_values, imgR_values, p_value) / 255.0f;
                        break;
                    default:
                        p1 = 1.0f - absolute_norm(imgL_values, imgR_values) / 255.0f;
                        break;
                }
                float sum = 0;
                for (size_t ch = 0; ch < features; ch++) {
                    byte* pFexL = fexL[ch].ptr<byte>(y);
                    byte* pFexR = fexR[ch].ptr<byte>(y);
                    float val_l = static_cast<float>(pFexL[x]);
                    float val_r = static_cast<float>(pFexR[x]);
                    sum += fabs(val_l - val_r) / 255;
                }
                p2 = (1.0f - sum / features);
                p = p1 * p2;
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

    imwrite(argv[8], disparity);
    imshow("Disparity", disparity);
    waitKey();

    return 0;
}
