#include <libasyik/service.hpp>
#include <libasyik/http.hpp>
#include <iostream>
#include <typeinfo>
#include <memory>
#include <string>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include "cpp_server/error.hpp"
#include "cpp_server/image_processor.hpp"

uint16_t validate_requests(const auto &req_ptr, rapidjson::Document &doc)
{

  uint16_t err_code = 200;
  if (req_ptr->headers["Content-Type"] != "application/json")
  {
    LOG(ERROR) << "Content type error: payload must be defined as application/json"
               << "\n";
    return 415;
  }
  if (doc.Parse(req_ptr->body.c_str(), req_ptr->body.size()).HasParseError())
  {
    LOG(ERROR) << "JSON parse error: " << doc.GetParseError() << " - " << rapidjson::GetParseError_En(doc.GetParseError()) << "\n";
    return 500;
  }

  if (!doc.HasMember("image"))
  {
    LOG(ERROR) << "Data validation error: Image is not available in data"
               << "\n";
    return 422;
  }
  return 200;
}

int main()
{
  auto as = asyik::make_service();
  auto server = asyik::make_http_server(as, "127.0.0.1", 8080);
  server->set_request_body_limit(10485760); // 10MB

  const int batch_size = 1;
  cpp_server::ModelConfig model_config;
  ClientConfig client_config;
  client_config.model_name = "img_cls_efficientnetv2_static";
  client_config.verbose = 1;

  ImageProcessor image_processor(model_config, client_config, batch_size);

  // accept string argument
  server->on_http_request("/classification/image", "POST", [&image_processor](auto req, auto args)
                          {
                            uint16_t r_errcode = 200;
                            rapidjson::Document payload_data, payload_result;

                            r_errcode = validate_requests(req, payload_data);
                            if (r_errcode != 200)
                            {
                              req->response.result(r_errcode);
                            }
                            else
                            {
                              cpp_server::Error proc_code = image_processor.process(payload_data, payload_result);
                              rapidjson::StringBuffer buffer; buffer.Clear();
                              rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                              writer.SetMaxDecimalPlaces(3);
                              payload_result.Accept(writer);

                              if (!proc_code.IsOk()) {
                                req->response.result(422);
                                req->response.body = proc_code.AsString();
                              }
                              else {
                                req->response.body = buffer.GetString();
                                req->response.headers.set("Content-Type", "application/json");
                                req->response.result(200);
                              }
                            } }); // other standard headers like content-length is set by library

  as->run();

  return 0;
}