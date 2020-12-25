import ArgumentParser
import PerfectHTTP
import PerfectHTTPServer
import Foundation

public extension Array where Element == (String, String) {
    subscript(index: String) -> String? {
        for item in self {
            if item.0 == index {
                return item.1
            }
        }
        return Optional.none
    } 
}

private func encodeCredentials(login: String?, password: String?) -> String? {
    if let login_ = login, let password_ = password {
        return "\(login_):\(password_)".data(using: .utf8)!.base64EncodedString(options: Data.Base64EncodingOptions(rawValue: 0))
    }
    return Optional.none
}

struct StartServer: ParsableCommand {

    @Option(name: .shortAndLong, help: "Port for the server to listen to")
    private var port: Int = 1719

    @Option(name: .shortAndLong, help: "Login for accessing the service")
    private var login: String = ""

    @Option(help: "Password for accessing the service")
    private var password: String = ""

    func parseRequestParameter(request: HTTPRequest, paramName: String, flag: String) -> [String] {
        if let paramValue = request.param(name: paramName) {
            return [flag, paramValue]
        } else {
            return []
        }
    }

    func handler(request: HTTPRequest, response: HTTPResponse) {
        response.setHeader(.contentType, value: "text/html")
        do {
            // var command = CrossValidate(model: "transe", dataset: "deduplicated-dataset.txt")
            let params = parseRequestParameter(request: request, paramName: "model", flag: "-m") + parseRequestParameter(request: request, paramName: "dataset", flag: "-d")
            print(params)
            var command = try CrossValidate.parse(params)
            let result = try command.run()
            response.appendBody(string: "<html><title>Completed!</title><body>\(result)</body></html>")
        } catch {
            response.appendBody(string: "<html><title>Exception!</title><body>\(error)</body></html>")
        }
        response.completed()
    }

    struct AuthenticationFilter: HTTPRequestFilter {
        let targetCredentials: String?

        func filter(request: HTTPRequest, response: HTTPResponse, callback: (HTTPRequestFilterResult) -> ()) {
            if targetCredentials == Optional.none || request.header(.authorization)?.components(separatedBy: " ")[1] ?? "" == targetCredentials! {
                callback(.continue(request, response))
            } else {
                response.status = .forbidden
                callback(.halt(request, response))
            }
        }
    }

    mutating func run() throws {
        print("Starting an http server...")
        
        var routes = Routes()
        routes.add(method: .get, uri: "/", handler: handler)
        
        try HTTPServer.launch(
            name: "localhost",
            port: port,
            routes: routes,
            requestFilters: [
                (
                    AuthenticationFilter(
                        targetCredentials: encodeCredentials(login: login.isEmpty ? Optional.none : login, password: password.isEmpty ? Optional.none : password)
                    ),
                    HTTPFilterPriority.high
                )
            ],
            responseFilters: [
                (
                    PerfectHTTPServer.HTTPFilter.contentCompression(data: [:]),
                    HTTPFilterPriority.high
                )
            ]
        )
    }
}
