#include "DGM.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;

void print_help(char *argv0) {
    printf("Usage: %s left_image right_image min_disparity max_disparity node_norm_function output_disparity\n", argv0);
    
    printf("\nNode norm function:\n");
    printf("0: Zero norm\n");
    printf("1: Manhattan norm\n");
    printf("2: Euclidean norm\n");
    printf("3: P-norm\n");
}

float zero_norm(Vec3b imgL_values, Vec3b imgR_values) {
    float blue = abs(imgL_values[0] - imgR_values[0]);
    float green = abs(imgL_values[1] - imgR_values[1]);
    float red = abs(imgL_values[2] - imgR_values[2]);
    float avg = (blue + green + red) / 3;
    return (pow(2, -1)*avg) / (1+avg);
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

int main(int argc, char *argv[]) {
    
    if (argc != 7) {
        print_help(argv[0]);
        return 0;
    }
    
    int gtScaleFactor = 6;

    // Reading parameters and images
    Mat imgL = imread(argv[1], 0);          if (imgL.empty()) printf("Can't open %s\n", argv[1]);
    resize(imgL, imgL, Size(imgL.cols / gtScaleFactor, imgL.rows / gtScaleFactor));
    
    Mat imgR = imread(argv[2], 0);          if (imgR.empty()) printf("Can't open %s\n", argv[2]);
    resize(imgR, imgR, Size(imgR.cols / gtScaleFactor, imgR.rows / gtScaleFactor));
    int             minDisparity    = atoi(argv[3]);
    int             maxDisparity    = atoi(argv[4]);
    int             nodeNorm        = atoi(argv[5]);
    int             width           = imgL.cols;
    int             height          = imgL.rows;
    unsigned int    nStates         = maxDisparity - minDisparity;
    
    CGraphPairwiseKit graphKit(nStates, INFER::TRW);
//    CGraphDenseKit graphKit(nStates); // Uncomment for Dense graph model, comment line above.
    
    graphKit.getGraphExt().buildGraph(imgL.size());
    graphKit.getGraphExt().addDefaultEdgesModel(1.175f);
    
    // ==================== Filling the nodes of the graph ====================
    Mat nodePot(nStates, 1, CV_32FC1);                                      // node potential (column-vector)
    float p;                                                                // potential
    int p_value = 0;                                                        // user input
    if (nodeNorm == 3) {
        printf( "Enter a p-value : ");
        scanf("%d", &p_value);
    }
    size_t idx = 0;
    for (int y = 0; y < height; y++) {
        byte * pImgL    = imgL.ptr<byte>(y);
        byte * pImgR    = imgR.ptr<byte>(y);
        for (int x = 0; x < width; x++) {
            Vec3b imgL_values = static_cast<Vec3b>(pImgL[x]);
            for (unsigned int s = 0; s < nStates; s++) {                    // state
                int disparity = minDisparity + s;
                Vec3b imgR_values = (x + disparity < width) ? static_cast<Vec3b>(pImgR[x + disparity]) : imgL_values;
                switch (nodeNorm) {
                    case 0:
                        p = 1.0f - zero_norm(imgL_values, imgR_values) / 3*255.0f;
                        break;
                    case 1:
                        p = 1.0f - manhattan_norm(imgL_values, imgR_values) / 3*255.0f;
                        break;
                    case 2:
                        p = 1.0f - euclidean_norm(imgL_values, imgR_values) / 3*255.0f;
                        break;
                    case 3:
                        p = 1.0f - p_norm(imgL_values, imgR_values, p_value) / 3*255.0f;
                        break;
                    default:
                        p = 1.0f - manhattan_norm(imgL_values, imgR_values) / 3*255.0f;
                        break;
                }
                nodePot.at<float>(s, 0) = p * p;
            }
            graphKit.getGraph().setNode(idx++, nodePot);
        } // x
    } // y
    
    // =============================== Decoding ===============================
    Timer::start("Decoding... ");
    vec_byte_t optimalDecoding = graphKit.getInfer().decode(100);
    Timer::stop();
    
    // ============================ Visualization =============================
    Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
    disparity = (disparity + minDisparity) * (256 / maxDisparity);
    medianBlur(disparity, disparity, 3);
    
    char dispStr[255];
    sprintf(dispStr, "Min-disparity: %d / Max-disparity: %d", minDisparity, maxDisparity);
    putText(disparity, dispStr, Point(width - 290, height - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1, cv::LineTypes::LINE_AA);
    
    if (nodeNorm == 3) {
        char pStr[255];
        sprintf(pStr, "P-value: %d", p_value);
        putText(disparity, pStr, Point(width - 380, height - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1, cv::LineTypes::LINE_AA);
    }
    
    imwrite(argv[6], disparity);
    imshow("Disparity", disparity);
    waitKey();

    return 0;
}
