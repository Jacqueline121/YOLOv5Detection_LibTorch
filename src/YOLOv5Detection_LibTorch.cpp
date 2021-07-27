#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>

std::vector<torch::Tensor> non_max_suppression(torch::Tensor outputs, float score_thresh=0.5, float iou_thresh=0.5){
    std::vector<torch::Tensor> preds;
    for (size_t i=0; i < outputs.sizes()[0]; ++i){
        torch::Tensor pred = outputs.select(0, i); // [15120, 85]
        // scores: conf * cls: [15120]
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1)); //slice (dim, star, end, step)
        // select predicted bboxs with scores that are greater than score_thresh.
        // pred: [select_num, 85]
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;

        //transfer (cx, cy, w, h) to (x1, y1, x2, y2)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        //compute scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor dets = pred.slice(1, 0, 6);
        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0){
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // compute IoU
            torch::Tensor minX = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor minY = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor maxX = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor maxY = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor width = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor height = torch::empty(indexes.sizes()[0] - 1);

            for (size_t i=0; i<indexes.sizes()[0]-1; ++i){
                minX[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i+1]][0].item().toFloat());
                minY[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i+1]][1].item().toFloat());
                maxX[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i+1]][2].item().toFloat());
                maxY[i] = std::max(dets[indexes[0]][3].item().toFloat(), dets[indexes[i+1]][3].item().toFloat());
                width[i] = std::max(float(0), maxX[i].item().toFloat() - minX[i].item().toFloat());
                height[i] = std::max(float(0), maxY[i].item().toFloat() - minY[i].item().toFloat());
            }
            torch::Tensor intersection_areas = width * height;

            torch::Tensor ious = intersection_areas / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - intersection_areas);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        preds.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return preds;
}

int main(){
    // load model using libtorch in c++
    torch::jit::script::Module module = torch::jit::load("../yolov5s.torchscript.pt");

    // load class name of dataset
    std::vector<std::string> classnames;
    std::ifstream f("../coco.names");
    std::string name = "";
    while (std::getline(f, name)){
        classnames.push_back(name);
    }

    // load test image
    cv::Mat ori, img;
    std::string file_path = "../test.jpg";
    ori = cv::imread(file_path);

    if(!ori.empty()){
        cv::resize(ori, img, cv::Size(640, 384));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // transfer image to tensor
        torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
        imgTensor = imgTensor.permute({2, 0, 1});
        imgTensor = imgTensor.toType(torch::kFloat);
        imgTensor = imgTensor.div(255);
        imgTensor = imgTensor.unsqueeze(0);

        //prediction：outputs: [B, 15120, 85]
        torch::Tensor outputs = module.forward({imgTensor}).toTuple()->elements()[0].toTensor();
        std::vector<torch::Tensor> dets = non_max_suppression(outputs, 0.4, 0.5);
        if (dets.size() > 0){
            for (size_t i = 0; i < dets[0].sizes()[0]; ++i){
                float x1 = dets[0][i][0].item().toFloat() * ori.cols / 640;
                float y1 = dets[0][i][1].item().toFloat() * ori.rows / 384;
                float x2 = dets[0][i][2].item().toFloat() * ori.cols / 640;
                float y2 = dets[0][i][3].item().toFloat() * ori.rows / 384;
                float score = dets[0][i][4].item().toFloat();
                float classID = dets[0][i][5].item().toInt();

                cv::rectangle(ori, cv::Rect(x1, y1, (x2 - x1), (y2 - y1)), cv::Scalar(255, 0, 0), 2);
                cv::putText(ori, classnames[classID] + ":" + cv::format("%.2f", score), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            }
        }
        cv::imshow("", ori);
        cv::waitKey(0);

    }else{
        std::cerr<<"loding image error!\n";
    }
    return 0;
}