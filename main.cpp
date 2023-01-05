#include <libasyik/service.hpp>
#include <libasyik/http.hpp>
#include <iostream>
#include <typeinfo>
#include <string>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

uint16_t validate_requests(const auto &req_ptr, rapidjson::Document &doc) {

  uint16_t err_code = 200;
  if (req_ptr->headers["Content-Type"] != "application/json")
  {
    LOG(ERROR) << "Content type error: payload must be defined as application/json" << "\n";
    return 415;
  }
  if (doc.Parse(req_ptr->body.c_str(), req_ptr->body.size()).HasParseError()) {
    LOG(ERROR) << "JSON parse error: " << doc.GetParseError() << " - " << rapidjson::GetParseError_En(doc.GetParseError()) << "\n";
    return 500;
  }

  if (!doc.HasMember("image")){
    LOG(ERROR) << "Data validation error: Image is not available in data" << "\n";
    return 422;
  }
  return 200;
}

int main()
{
  auto as = asyik::make_service();
  auto server = asyik::make_http_server(as, "127.0.0.1", 8080);

  // accept string argument
  server->on_http_request("/name/<string>", "POST", [](auto req, auto args)
                          {
                            uint16_t r_errcode = 200;
                            rapidjson::Document payload_data;
                            r_errcode = validate_requests(req, payload_data);
                            if (r_errcode != 200)
                            {
                              req->response.result(r_errcode);
                            }
                            else
                            {
                              req->response.body = "{\"value\":\"Hello " + args[1] + "!\"}";
                              LOG(INFO) << "X-Asyik=" << req->headers["x-asyik"] << "\n";
                              LOG(INFO) << "Body=" << req->body << " type: " <<  typeid(req->body).name() <<"\n";
                              // Body={"name": "harit", "data": ["key", "bb", "image"]} type: NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
                              req->response.headers.set("Content-Type", "text/json");
                              req->response.result(200);
                            } }); // other standard headers like content-length is set by library

  as->run();

  return 0;
}