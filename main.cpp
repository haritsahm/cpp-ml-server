#include <libasyik/service.hpp>
#include <libasyik/http.hpp>

int main()
{
  auto as = asyik::make_service();
  auto server = asyik::make_http_server(as, "127.0.0.1", 8080);

  // serve http request
  server->on_http_request("/hello", "GET", [](auto req, auto args)
                          {
      req->response.body = "Hello world!";
      req->response.result(200); });

  as->run();

  return 0;
}