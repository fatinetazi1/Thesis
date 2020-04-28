// Example Disparity estimation with CRFs
#include "DGM.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;

void print_help(char *argv0) {
    printf("Usage: %s left_image right_image min_disparity max_disparity node_norm_function edge_training_model left_image_groundtruth right_image_features output_disparity\n", argv0);
    
    printf("\nNode norm function:\n");
    printf("0: Zero norm\n");
    printf("1: Manhattan norm\n");
    printf("2: Euclidean norm\n");
    printf("3: P-norm\n");
    
    printf("\nEdge training models:\n");
    printf("0: Potts Model\n");
    printf("1: Contrast-Sensitive Potts Model\n");
    printf("2: Contrast-Sensitive Potts Model with Prior\n");
    printf("3: Concatenated Model\n");
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

float zero_norm(Vec3b imgL_values, Vec3b imgR_values) {
    float blue = abs(imgL_values[0] - imgR_values[0]);
    float green = abs(imgL_values[1] - imgR_values[1]);
    float red = abs(imgL_values[2] - imgR_values[2]);
    float avg = (blue + green + red) / 3;
    return (pow(2, -1)*avg) / (1+avg);
}

float meanSqrDist(Mat im1, Mat im2, float scaleFactor = 1) {
    float sum = 0;
    for (int y = 0; y < im1.rows; y++) {
        const byte *pM1 = im1.ptr<byte>(y);
        const byte *pM2 = im2.ptr<byte>(y);
        for (int x = 0; x < im1.cols; x++){
            float difference = pM1[x] - (pM2[x]/scaleFactor);
            sum += pow(abs(difference), 2);
        }
    }
    return sum / (im1.rows*im1.cols);
}

int main(int argc, char *argv[]) {
    if (argc != 11) {
        print_help(argv[0]);
        return 0;
    }

    // Reading parameters and images
    Mat imgL            = imread(argv[1], 1);   if (imgL.empty()) printf("Can't open %s\n", argv[1]);
    Mat imgR            = imread(argv[2], 1);   if (imgR.empty()) printf("Can't open %s\n", argv[2]);
    int minDisparity    = atoi(argv[3]);
    int maxDisparity    = atoi(argv[4]);
    int nodeNorm        = atoi(argv[5]);
    int edgeModel       = atoi(argv[6]);
    
    Mat train_gt = imread(argv[7], -1); if (train_gt.empty()) printf("Can't open %s\n", argv[7]);
    resize(train_gt, train_gt, imgL.size(), 0, 0, INTER_NEAREST);        // groundtruth for training (imgL)
    
    Mat test_fv = imread(argv[8], 1);   if (test_fv.empty()) printf("Can't open %s\n", argv[8]);
    resize(test_fv,  test_fv,  imgL.size(), 0, 0, INTER_LANCZOS4);    // testing image feature vector (imgR)
    
    Mat test_gt = imread(argv[9], -1);   if (test_gt.empty()) printf("Can't open %s\n", argv[9]);
    resize(test_gt,  test_gt,  imgL.size(), 0, 0, INTER_NEAREST);    // groundtruth for evaluation (imgR)
    
    const int           width       = imgL.cols;
    const int           height      = imgL.rows;
    const unsigned int  nStates     = maxDisparity - minDisparity;
    const word          nFeatures   = 3;
    
    test_gt.convertTo(test_gt, CV_8UC1);
    
    // Preparing parameters for edge trainers
    vec_float_t            vParams = {100, 0.01f};
    if (edgeModel == 0 || edgeModel == 3) vParams.pop_back(); // Potts and Concat models need ony 1 parameter
    
    auto                edgeTrainer = CTrainEdge::create(edgeModel, nStates, nFeatures);
    CGraphPairwise      graph(nStates);
    CGraphPairwiseExt   graphExt(graph);
    CInferLBP           decoder(graph);
    
    // Initializing Powell search class and parameters
    const vec_float_t vInitParams  = { 100.0f, 300.0f, 3.0f, 10.0f };
    const vec_float_t vInitDeltas  = {  10.0f,  10.0f, 1.0f,  1.0f };
    vec_float_t vEstParams = vInitParams;                                    // Actual model parameters

    CPowell powell(vEstParams.size());
    powell.setInitParams(vInitParams);
    powell.setDeltas(vInitDeltas);
    
    // ==================== Building the graph ====================
    Timer::start("Building the Graph... ");
    graphExt.buildGraph(imgL.size());
    Timer::stop();

    // ==================== Node Potentials & Edge Training ====================
    Timer::start("Setting the Node Potentials & Edge Training... ");
    
    // Node Potentials
    float p;
    int p_value = 0;
    if (nodeNorm == 2) {
        printf( "Enter a p-value : ");
        scanf("%d", &p_value);
    }
    
    Mat nodePot(imgL.size(), CV_32FC(nStates));
    Mat pot(nStates, 1, CV_32FC1);
    for (int y = 0; y < height; y++) {
        byte * pImgL        = imgL.ptr<byte>(y);
        byte * pImgR        = imgR.ptr<byte>(y);
        float * pNodePot    = nodePot.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            //float imgL_value = static_cast<float>(pImgL[x]);
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
                pot.at<float>(s, 0) = p * p;
            }
            for (int s = 0; s < nStates; s++) pNodePot[nStates * x + s] = pot.at<float>(s, 0);
        } // x
    } // y
    
    // Edge Training
    Mat featureVector1(nFeatures, 1, CV_8UC1);
    Mat featureVector2(nFeatures, 1, CV_8UC1);
    for (int y = 1; y < height; y++) {
        byte *pimgL1 = imgL.ptr<byte>(y);
        byte *pimgL2 = imgL.ptr<byte>(y - 1);
        byte *pimgLGT1 = train_gt.ptr<byte>(y);
        byte *pimgLGT2 = train_gt.ptr<byte>(y - 1);
        for (int x = 1; x < width; x++) {
            for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pimgL1[nFeatures * x + f];       // featureVector1 = fv[x][y]
            for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pimgL1[nFeatures * (x - 1) + f]; // featureVector2 = fv[x-1][y]
            edgeTrainer->addFeatureVecs(featureVector1, pimgLGT1[x], featureVector2, pimgLGT1[x-1]);
            edgeTrainer->addFeatureVecs(featureVector2, pimgLGT1[x-1], featureVector1, pimgLGT1[x]);
            for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pimgL2[nFeatures * x + f];       // featureVector2 = fv[x][y-1]
            edgeTrainer->addFeatureVecs(featureVector1, pimgLGT1[x], featureVector2, pimgLGT2[x]);
            edgeTrainer->addFeatureVecs(featureVector2, pimgLGT2[x], featureVector1, pimgLGT1[x]);
        } // x
    } // y
    
    edgeTrainer->train();
    Timer::stop();
    
    vec_byte_t optimalDecoding;
    
    // Main loop of parameters optimization
    for (int i = 1; ; i++) {
        // ==================== STAGE 3: Filling the Graph =====================
        Timer::start("Filling the Graph... ");
        // Filling in the node potentials
        byte nStatesBase = static_cast<byte>(nodePot.channels());
        Mat nPotBase(nStates, 1, CV_32FC1, Scalar(0.0f));
        for (int y = 0; y < height; y++) {
            const float *pPotBase = nodePot.ptr<float>(y);
            for (int x = 0; x < width; x++) {
                size_t idx = (y * width + x);
                for (byte s = 0; s < nStatesBase; s++)
                    nPotBase.at<float>(s, 0) = pPotBase[nStatesBase * x + s];
                graph.setNode(idx, nPotBase);
             } // x
         } // y
        graphExt.fillEdges(*edgeTrainer, test_fv, vParams);    // Filling-in the graph edges with pairwise potentials
        Timer::stop();
        
        // =============================== Decoding ===============================
        Timer::start("Decoding... ");
        optimalDecoding = decoder.decode(10);
        Timer::stop();
        
        // ====================== Evaluation =======================
        Mat solution(imgL.size(), CV_8UC1, optimalDecoding.data());
        float val = meanSqrDist(solution, test_gt, 8);              // compare solution with the groundtruth
        
        printf("Iteration: %d, parameters: { ", i);
        for (const float& param : vEstParams) printf("%.1f ", param);
        printf("}, accuracy: %.2f%%\n", val);

        if (powell.isConverged()) break;
        vEstParams = powell.getParams(val);
    }
    
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
    
    imwrite(argv[10], disparity);
    imshow("Disparity", disparity);
    waitKey();

    return 0;
}
