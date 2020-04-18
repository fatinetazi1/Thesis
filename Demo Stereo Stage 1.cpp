// Example Disparity estimation with CRFs
#include "DGM.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;

void print_help(char *argv0)
{
    printf("Usage: %s left_image right_image min_disparity max_disparity node_norm_function output_disparity\n", argv0);
    
    printf("\nNode norm function:\n");
    printf("0: Manhattan norm\n");
    printf("1: Euclidean norm\n");
    printf("2: P-norm\n");
    printf("3: Zero norm\n");
}

float manhattan_norm(float i) {
    return abs(i);
}

float euclidean_norm(float i) {
    return sqrt(pow(abs(i), 2));
}

float p_norm(float i, int p) {
    return pow(pow(abs(i), p), 1/p);
}

float zero_norm(float i) {
    return (pow(2, -1)*i) / (1+i);
}

int main(int argc, char *argv[])
{
    if (argc != 7) {
        print_help(argv[0]);
        return 0;
    }

    // Reading parameters and images
    Mat                imgL            = imread(argv[1], 0);    if (imgL.empty()) printf("Can't open %s\n", argv[1]);
    Mat                imgR            = imread(argv[2], 0);    if (imgR.empty()) printf("Can't open %s\n", argv[2]);
    int                minDisparity    = atoi(argv[3]);
    int                maxDisparity    = atoi(argv[4]);
    int             nodeNorm        = atoi(argv[5]);
    int                width            = imgL.cols;
    int                height            = imgL.rows;
    unsigned int nStates    = maxDisparity - minDisparity;
    
    CGraphPairwiseKit graphKit(nStates, INFER::TRW);
//    CGraphDenseKit graphKit(nStates); // Uncomment for Dense graph model, comment line above.
    
    graphKit.getGraphExt().buildGraph(imgL.size());
    graphKit.getGraphExt().addDefaultEdgesModel(1.175f);
    
    // ==================== Filling the nodes of the graph ====================
    Mat nodePot(nStates, 1, CV_32FC1);                                      // node Potential (column-vector)
    float p;
    int p_value = 0;
    if (nodeNorm == 2) {
        printf( "Enter a p-value : ");
        scanf("%d", &p_value);
    }
    size_t idx = 0;
    for (int y = 0; y < height; y++) {
        byte * pImgL    = imgL.ptr<byte>(y);
        byte * pImgR    = imgR.ptr<byte>(y);
        for (int x = 0; x < width; x++) {
            float imgL_value = static_cast<float>(pImgL[x]);
            for (unsigned int s = 0; s < nStates; s++) {                    // state
                int disparity = minDisparity + s;
                float imgR_value = (x + disparity < width) ? static_cast<float>(pImgR[x + disparity]) : imgL_value;
                switch (nodeNorm) {
                    case 0:
                        p = 1.0f - manhattan_norm(imgL_value - imgR_value) / 255.0f;
                        break;
                    case 1:
                        p = 1.0f - euclidean_norm(imgL_value - imgR_value) / 255.0f;
                        break;
                    case 2:
                        p = 1.0f - p_norm(imgL_value - imgR_value, p_value) / 255.0f;
                        break;
                    case 3:
                        p = 1.0f - zero_norm(imgL_value - imgR_value) / 255.0f;
                        break;
                    default:
                        p = 1.0f - manhattan_norm(imgL_value - imgR_value) / 255.0f;
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
    if (nodeNorm == 2) {
        char pStr[255];
        sprintf(pStr, "P-value: %d", p_value);
        putText(disparity, pStr, Point(width - 380, height - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1, cv::LineTypes::LINE_AA);
    }
    
    imwrite(argv[6], disparity);

    imshow("Disparity", disparity);

    waitKey();

    return 0;
}
