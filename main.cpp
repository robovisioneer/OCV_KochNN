#include <opencv2/opencv.hpp>
#include <memory>
#include <iostream>
#include <opencv2/ml.hpp>
#include <cmath>
//#include <random>
#include <fstream>

using namespace cv;
using namespace cv::ml;

class point{
private:
    double x;
    double y;
public:
    void set_x(double x_val) {
        x = x_val;
    }
    void set_y(double y_val) {
        y = y_val;
    }
    double get_x() {
        return x;
    }
    double get_y() {
        return y;
    }
};

class arm{
    point pt_Anfang;
    point pt_Ende;
public:
    void set_Anfang(double x, double y) {
        pt_Anfang.set_x(x);
        pt_Anfang.set_y(y);
    }
    void set_Ende(double x, double y) {
        pt_Ende.set_x(x);
        pt_Ende.set_y(y);
    }
    point get_Anfang() {
        return pt_Anfang;
    }
    point get_Ende() {
        return pt_Ende;     
    }
};

class Koch_Kurve {
private:
    arm Arm1;
    arm Arm2;
    arm Arm_neue_Spitze1;
    arm Arm_neue_Spitze2;
    std::vector<point> Punkte;
public:
    Koch_Kurve() {
        Arm1.set_Anfang(0,0);
        Arm1.set_Ende(3,0);
    }

    Koch_Kurve(int s, arm A, std::vector<point>& vRef_Punkte) {
        double x1 = A.get_Anfang().get_x();
        double y1 = A.get_Anfang().get_y();
        double x2 = A.get_Ende().get_x();
        double y2 = A.get_Ende().get_y();
        double x_1_3 = (x2-x1)*1/3 + x1;
        double y_1_3 = (y2-y1)*1/3 + y1;
        double x_2_3 = (x2-x1)*2/3 + x1;
        double y_2_3 = (y2-y1)*2/3 + y1;
        double x_1_2 = (x2-x1)*1/2 + x1;
        double y_1_2 = (y2-y1)*1/2 + y1;
        double x_spitze = x_1_2 - (sqrt(3)/6)*(y2 - y1);
        double y_spitze = y_1_2 + (sqrt(3)/6)*(x2 - x1);
        Arm1.set_Anfang(x1, y1);
        Arm1.set_Ende(x_1_3, y_1_3);
        Arm2.set_Anfang(x_2_3, y_2_3);
        Arm2.set_Ende(x2, y2);
        Arm_neue_Spitze1.set_Anfang(x_1_3, y_1_3);
        Arm_neue_Spitze1.set_Ende(x_spitze, y_spitze);
        Arm_neue_Spitze2.set_Anfang(x_spitze, y_spitze);
        Arm_neue_Spitze2.set_Ende(x_2_3, y_2_3);
        
        if(s > 0) {

            Koch_Kurve K1(s-1, Arm1, vRef_Punkte);
            Koch_Kurve K2(s-1, Arm_neue_Spitze1, vRef_Punkte);
            Koch_Kurve K3(s-1, Arm_neue_Spitze2,vRef_Punkte);
            Koch_Kurve K4(s-1, Arm2, vRef_Punkte);

        }
        else {
            vRef_Punkte.emplace_back(A.get_Anfang());
            vRef_Punkte.emplace_back(Arm_neue_Spitze1.get_Anfang());
            vRef_Punkte.emplace_back(Arm_neue_Spitze1.get_Ende());
            vRef_Punkte.emplace_back(Arm_neue_Spitze2.get_Ende());
        }
    }
    void substituieren(int s) {
        Koch_Kurve K1(s-1, Arm1, Punkte);
        Punkte.emplace_back(Arm1.get_Ende());
        std::cout << "Anzahl Punkte: " << Punkte.size() << "\n";
    }
    double get_point_x(int i) {
        return Punkte[i].get_x();
    }
    double get_point_y(int i) {
        return Punkte[i].get_y();
    }
    
};

void saveMatToCSV(const cv::Mat& mat, std::string filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // Nutze den OpenCV Formatter für CSV-Stil
        file << cv::format(mat, cv::Formatter::FMT_CSV) << std::endl;
        file.close();
    }
}

int main() {

    // ********************************
    // * Daten eingeben und skalieren *
    // ********************************
    Koch_Kurve KK;
    KK.substituieren(5);
    Mat data_input_x_s(513, 1, CV_32FC1);
    Mat data_output_x_s(513, 1, CV_32FC1);
    Mat data_input_s_y(513, 1, CV_32FC1);
    Mat data_output_s_y(513, 1, CV_32FC1);
    for (int i=0; i<513; ++i) {
        data_input_s_y.at<float>(i, 0) = static_cast<float>(i/512.0f);
        data_output_s_y.at<float>(i, 0) = static_cast<float>((KK.get_point_y(i))/sqrt(3)*2.0f);
    }
    for (int i=0; i<513; ++i) {
        data_output_x_s.at<float>(i, 0) = static_cast<float>(i/512.0f);
        data_input_x_s.at<float>(i, 0) = static_cast<float>((KK.get_point_x(i))/1.5f);
    }
    
    // ****************************
    // * Neuronales Netz aufbauen *
    // ****************************
    auto mlp_x_s = cv::ml::ANN_MLP::create();
    auto mlp_s_y = cv::ml::ANN_MLP::create();

    // ZUERST DIE SCHICHTEN
    cv::Mat layers_ = (cv::Mat_<int>(10, 1) << 1, 12, 16, 24, 32, 32, 24, 16, 10, 1);
    cv::Mat layers0 = (cv::Mat_<int>(10, 1) << 1, 12, 16, 24, 32, 32, 24, 16, 10, 1);
    mlp_x_s->setLayerSizes(layers_);    
    mlp_s_y->setLayerSizes(layers0);

    // ************************************************
    // * Parameter fuer das neuronale Netz einstellen *
    // ************************************************
    mlp_x_s->setTrainMethod(cv::ml::ANN_MLP::RPROP, 1e-6); // Sehr kleine DW0
    mlp_s_y->setTrainMethod(cv::ml::ANN_MLP::RPROP, 1e-6); // Sehr kleine DW0
    //mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0.5, 1.0);
    mlp_x_s->setActivationFunction(cv::ml::ANN_MLP::LEAKYRELU, 0.05, 0.01);
    mlp_s_y->setActivationFunction(cv::ml::ANN_MLP::LEAKYRELU, 0.05, 0.01);
    mlp_x_s->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5000, 1e-7));
    mlp_s_y->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5000, 1e-7));
    mlp_x_s->setRpropDW0(0.00001);  // Winziger Start
    mlp_s_y->setRpropDW0(0.00001);  // Winziger Start
    mlp_x_s->setRpropDWMax(0.01);   // Harter Deckel gegen Explosion
    mlp_s_y->setRpropDWMax(0.01);   // Harter Deckel gegen Explosion
    mlp_x_s->setRpropDWMin(1e-7);
    mlp_s_y->setRpropDWMin(1e-7);
    // Dummy-Daten
    cv::Mat null_data_input = cv::Mat::zeros(4, 1, CV_32F);
    cv::Mat null_data_output = cv::Mat::zeros(4, 1, CV_32F);
    null_data_input.at<float>(0, 0) = 0.0f;
    null_data_input.at<float>(1, 0) = 0.33f;
    null_data_input.at<float>(2, 0) = 0.67f;
    null_data_input.at<float>(3, 0) = 1.0f;
    mlp_x_s->train(null_data_input,ROW_SAMPLE, null_data_output);
    mlp_s_y->train(null_data_input,ROW_SAMPLE, null_data_output);
    mlp_x_s->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-7));
    mlp_s_y->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-7));
    cv::Ptr<cv::ml::TrainData> trainingObject_x_s = cv::ml::TrainData::create(
        data_input_x_s, 
        cv::ml::ROW_SAMPLE, 
        data_output_x_s);
    cv::Ptr<cv::ml::TrainData> trainingObject_s_y = cv::ml::TrainData::create(
        data_input_s_y, 
        cv::ml::ROW_SAMPLE, 
        data_output_s_y);
    mlp_x_s->train(trainingObject_x_s,cv::ml::ANN_MLP::UPDATE_WEIGHTS |
         cv::ml::ANN_MLP::NO_INPUT_SCALE |
          cv::ml::ANN_MLP::NO_OUTPUT_SCALE);
    mlp_s_y->train(trainingObject_s_y,cv::ml::ANN_MLP::UPDATE_WEIGHTS |
         cv::ml::ANN_MLP::NO_INPUT_SCALE |
          cv::ml::ANN_MLP::NO_OUTPUT_SCALE);
    Mat nn_result(1025, 1, CV_32F);      
    mlp_x_s->predict(data_input_x_s, nn_result);
    mlp_s_y->predict(nn_result, nn_result);
    Mat final_results(1025, 1, CV_32F);
    // Symmetrie nutzen
    for(int i=0; i<513; ++i) {
        final_results.at<float>(i,0) = nn_result.at<float>(i,0);
        final_results.at<float>(1024-i,0) = nn_result.at<float>(i,0);
    }

    // Plot in eine Matrix rendern und anzeigen
    saveMatToCSV(final_results, "ergebnisse.csv");

    for (int i=0; i<513; ++i) {
        std::cout << "Eingabe: " << data_input_x_s.at<float>(i,0) << " Vorhersage: " << nn_result.at<float>(i,0) << " Erwartet: " << data_output_x_s.at<float>(i,0) << "\n";
    }    
    return 0;
}
