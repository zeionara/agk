import ArgumentParser
import PerfectHTTP
import PerfectHTTPServer

struct StartServer: ParsableCommand {

    @Option(name: .shortAndLong, help: "Port for the server to listen to")
    private var port: Int = 1719

    func handler(request: HTTPRequest, response: HTTPResponse) {
        response.setHeader(.contentType, value: "text/html")
        response.appendBody(string: "<html><title>Hello, world!</title><body>Hello, world!</body></html>")
        response.completed()
    }

    mutating func run() throws {
        print("Starting an http server...")
        var routes = Routes()
        routes.add(method: .get, uri: "/", handler: handler)
        // routes.add(method: .get, uri: "/**", handler: StaticFileHandler(documentRoot: "./webroot", allowResponseFilters: true).handleRequest)
        try HTTPServer.launch(
            name: "localhost",
            port: port,
            routes: routes,
            responseFilters: [
                (
                    PerfectHTTPServer.HTTPFilter.contentCompression(data: [:]),
                    HTTPFilterPriority.high
                )
            ]
        )
    }
}
