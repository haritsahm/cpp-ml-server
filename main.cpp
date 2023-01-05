#include <libasyik/service.hpp>
#include <libasyik/http.hpp>
#include <iostream>
#include <typeinfo>
#include <string>

uint16_t validate_headers(const auto &headers)
{
  if (headers["Content-Type"] != "application/json")
  {
    return 415;
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
                            r_errcode = validate_headers(req->headers);
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