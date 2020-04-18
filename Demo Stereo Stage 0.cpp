// Example Disparity estimation with CRFs
#include "DGM.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;

void print_help(char *argv0)
{
    printf("Usage: %s left_image right_image min_disparity max_disparity node_norm_function edge_training_model output_disparity\n", argv0);
    
    printf("\nNode norm function:\n");
    printf("0: Manhattan norm\n");
    printf("1: Euclidean norm\n");
    printf("2: P-norm\n");
    printf("3: Zero norm\n");
    
    printf("\nEdge training models:\n");
    printf("0: Potts Model\n");
    printf("1: Contrast-Sensitive Potts Model\n");
    printf("2: Contrast-Sensitive Potts Model with Prior\n");
    printf("3: Concatenated Model\n");
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
    if (argc != 8) {
        print_help(argv[0]);
        return 0;
    }

    // Reading parameters and images
    Mat                imgL            = imread(argv[1], 0);    if (imgL.empty()) printf("Can't open %s\n", argv[1]);
    Mat                imgR            = imread(argv[2], 0);    if (imgR.empty()) printf("Can't open %s\n", argv[2]);
    int                minDisparity    = atoi(argv[3]);
    int                maxDisparity    = atoi(argv[4]);
    int             nodeNorm        = atoi(argv[5]);
    int             edgeModel       = atoi(argv[6]);
    int                width            = imgL.cols;
    int                height            = imgL.rows;
    unsigned int    nStates            = maxDisparity - minDisparity;
    const word      nFeatures       = 3;
    
    // Preparing parameters for edge trainers
    vec_float_t            vParams = {100, 0.01f};
    if (edgeModel == 0 || edgeModel == 3) vParams.pop_back(); // Potts and Concat models need ony 1 parameter
    
    auto                edgeTrainer = CTrainEdge::create(edgeModel, nStates, nFeatures);
    CGraphPairwise      graph(nStates);
    CGraphPairwiseExt   graphExt(graph);
    CInferLBP           decoder(graph);
    CCMat               confMat(nStates);
    
    // ==================== Building the graph ====================
    Timer::start("Building the Graph... ");
    graphExt.buildGraph(imgL.size());
    Timer::stop();

    // ==================== Node Potentials & Edge Training ====================
    Timer::start("Setting the Node Potentials & Edge Training... ");
    
    // Node Potentials
    Mat nodePot(nStates, 1, CV_32FC(nStates));
    float p;
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
                        int p_value;
                        printf( "Enter a p-value :");
                        p_value = getchar( );
                        p = 1.0f - p_norm(imgL_value - imgR_value, p_value) / 255.0f;
                        break;
                    case 3:
                        p = 1.0f - zero_norm(imgL_value - imgR_value) / 255.0f;
                        break;
                    default:
                        p = 1.0f - manhattan_norm(imgL_value - imgR_value) / 255.0f;
                        break;
                }
                
                nodePot.at<float>(s, s) = p * p;
            }
        } // x
    } // y
    
    // Edge Training
    Mat featureVector1(nFeatures, 1, CV_8UC1);
    Mat featureVector2(nFeatures, 1, CV_8UC1);
    for (int y = 1; y < height; y++) {
        byte *pimgL1 = imgL.ptr<byte>(y);
        byte *pimgL2 = imgL.ptr<byte>(y - 1);
        byte *pimgR1 = imgR.ptr<byte>(y);
        byte *pimgR2 = imgR.ptr<byte>(y - 1);
        for (int x = 1; x < width; x++) {
            for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pimgL1[nFeatures * x + f];       // featureVector1 = fv[x][y]
            for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pimgL1[nFeatures * (x - 1) + f]; // featureVector2 = fv[x-1][y]
            edgeTrainer->addFeatureVecs(featureVector1, pimgR1[x], featureVector2, pimgR1[x-1]);
            edgeTrainer->addFeatureVecs(featureVector2, pimgR1[x-1], featureVector1, pimgR1[x]);
            for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pimgL2[nFeatures * x + f];       // featureVector2 = fv[x][y-1]
            edgeTrainer->addFeatureVecs(featureVector1, pimgR1[x], featureVector2, pimgR2[x]);
            edgeTrainer->addFeatureVecs(featureVector2, pimgR2[x], featureVector1, pimgR1[x]);
        } // x
    } // y
    
    edgeTrainer->train();
    Timer::stop();
    
    // ==================== STAGE 3: Filling the Graph =====================
    Timer::start("Filling the Graph... ");
    graphExt.setGraph(nodePot);                         // Filling-in the graph nodes
    graphExt.fillEdges(*edgeTrainer, imgR, vParams);    // Filling-in the graph edges with pairwise potentials
    Timer::stop();
    
    // =============================== Decoding ===============================
    Timer::start("Decoding... ");
    vec_byte_t optimalDecoding = decoder.decode(10);
    Timer::stop();
    
    // ============================ Visualization =============================
    Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
    disparity = (disparity + minDisparity) * (256 / maxDisparity);
    medianBlur(disparity, disparity, 3);
    
    char str[255];
    sprintf(str, "Min-disparity: %d / Max-disparity: %d", minDisparity, maxDisparity);
    putText(disparity, str, Point(width - 290, height - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 2, cv::LineTypes::LINE_AA);
    
    imwrite(argv[7], disparity);

    imshow("Disparity", disparity);

    waitKey();

    return 0;
}
