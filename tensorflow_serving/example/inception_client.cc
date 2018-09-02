/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>


#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using grpc::CompletionQueue;
using grpc::ClientAsyncResponseReader;


using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

struct AsyncClientCall {
    PredictResponse reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<PredictResponse>> response_reader;
};

void parse_response(Status status, PredictResponse response) {
    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is " << response.outputs_size() << std::endl;
      OutMap& map_outputs = *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;

      for (iter = map_outputs.begin(); iter != map_outputs.end(); ++iter) {
        tensorflow::TensorProto& result_tensor_proto = iter->second;
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(result_tensor_proto);
        if (converted) {
          std::cout << "the result tensor[" << output_index
                    << "] is:" << std::endl
                    << tensor.SummarizeValue(10) << std::endl;
        } else {
          std::cout << "the result tensor[" << output_index
                    << "] convert failed." << std::endl;
        }
        ++output_index;
      }
      std::cout << "Done." << std::endl;
    } else {
      std::cout << "gRPC call return code: " << status.error_code() << ": "
                << status.error_message() << std::endl;
      std::cout <<  "gRPC failed." << std::endl;
    }
}

std::tuple<char*, long> read_image(const tensorflow::string& file_path) {
    std::ifstream imageFile(file_path, std::ios::binary);

    // if (!imageFile.is_open()) {
    //   std::cout << "Failed to open " << file_path << std::endl;
    //   return "";
    // }

    std::filebuf* pbuf = imageFile.rdbuf();
    auto fileSize = pbuf->pubseekoff(0, std::ios::end, std::ios::in);

    char* image = new char[fileSize]();

    pbuf->pubseekpos(0, std::ios::in);
    pbuf->sgetn(image, fileSize);
    imageFile.close();
    return std::make_tuple(image, fileSize);
}

class ServingClient {
 public:
  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

  tensorflow::string callPredict(const tensorflow::string& model_name,
                                 const tensorflow::string& model_signature_name,
                                 const tensorflow::string& file_path) {
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name);
    predictRequest.mutable_model_spec()->set_signature_name(
        model_signature_name);

    google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs =
        *predictRequest.mutable_inputs();

    tensorflow::TensorProto proto;

    auto image_data_tup = read_image(file_path);
    char* image = std::get<0>(image_data_tup);
    long fileSize = std::get<1>(image_data_tup);

    proto.set_dtype(tensorflow::DataType::DT_STRING);
    proto.add_string_val(image, fileSize);

    proto.mutable_tensor_shape()->add_dim()->set_size(1);

    inputs["images"] = proto;

    // auto start = std::chrono::high_resolution_clock::now();

    // CompletionQueue cq;
    // Status status;
    // std::unique_ptr<ClientAsyncResponseReader<PredictResponse>> rpc(
    //   stub_->PrepareAsyncPredict(&context, predictRequest, &cq));
    // rpc->StartCall();
    // rpc->Finish(&response, &status, (void*)1);

    AsyncClientCall* call = new AsyncClientCall;
    call->response_reader =
        stub_->PrepareAsyncPredict(&call->context, predictRequest, &cq_);
    call->response_reader->StartCall();
    call->response_reader->Finish(&call->reply, &call->status, (void*)call);

    // void* got_tag;
    // bool ok = false;
    // GPR_ASSERT(cq.Next(&got_tag, &ok));
    // GPR_ASSERT(got_tag == (void*)1);
    // GPR_ASSERT(ok);

    // Status status = stub_->Predict(&context, predictRequest, &response);

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Just Predict Runtime: " << duration.count() << std::endl;

    delete[] image;

    return "Done";
  }

  void AsyncCompleteRpc() {
      void* got_tag;
      bool ok = false;

      auto start = std::chrono::high_resolution_clock::now();

      // Block until the next result is available in the completion queue "cq".
      while (cq_.Next(&got_tag, &ok)) {
          // The tag in this example is the memory location of the call object
          AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);

          // Verify that the request was completed successfully. Note that "ok"
          // corresponds solely to the request for updates introduced by Finish().
          GPR_ASSERT(ok);

          parse_response(call->status, call->reply);

          // Once we're complete, deallocate the call object.
          delete call;

          auto end = std::chrono::high_resolution_clock::now();
          auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
          std::cout << "Time elapsed from the start " 
                    <<  diff.count() << " us\n";
      }
  }


 private:
  std::unique_ptr<PredictionService::Stub> stub_;
  CompletionQueue cq_;
};

int main(int argc, char** argv) {
  tensorflow::string server_port = "localhost:9000";
  tensorflow::string image_file = "";
  tensorflow::string model_name = "inception";
  tensorflow::string model_signature_name = "predict_images";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port,
                       "the IP and port of the server"),
      tensorflow::Flag("image_file", &image_file, "the path to the image"),
      tensorflow::Flag("model_name", &model_name, "name of model"),
      tensorflow::Flag("model_signature_name", &model_signature_name,
                       "name of model signature")};

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || image_file.empty()) {
    std::cout << usage;
    return -1;
  }

  std::shared_ptr<ServingClient> guide = std::make_shared<ServingClient>(
      grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials()));
  std::cout << "calling predict using file: " << image_file << "  ..."
            << std::endl;
  
  
  std::thread thread_ = std::thread(&ServingClient::AsyncCompleteRpc, guide.get());
  // std::vector<std::thread> threads = {};
  
  
  for(int i = 0; i < 50; i++) {
    guide->callPredict(model_name, model_signature_name, image_file);
  }
  thread_.join();

  // for (int i=0; i < 10; i++){
    
  //   // std::cout << guide.callPredict(model_name, model_signature_name, image_file)
  //   //           << std::endl;
  //   threads.push_back(std::thread(
  //     [guide, model_name, model_signature_name, image_file](){
  //       guide->callPredict(model_name, model_signature_name, image_file);
  //     }
  //     ));
  // }
  // for (auto &t:threads) {
  //   t.join();
  // }

  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << "Over all duration " << duration.count() << " us" << std::endl;


  return 0;
}
