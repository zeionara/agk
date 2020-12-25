import ArgumentParser
import PerfectHTTP
import PerfectHTTPServer
import Foundation
import StORM
import MongoDBStORM
// import PerfectMongoDB

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

// var dbClient = try! MongoClient(
//     uri: "mongodb+srv://cluster0.lsifi.mongodb.net"
// )

public let EXPERIMENTS_COLLECTION_NAME = "test-experiments"

struct StartServer: ParsableCommand {

    @Option(name: .shortAndLong, help: "Port for the server to listen to")
    private var port: Int = 1719

    @Option(name: .shortAndLong, help: "Login for accessing the service")
    private var login: String = ""

    @Option(help: "Password for accessing the service")
    private var password: String = ""

    @Option(help: "Database host")
    private var dbHost: String = ""

    @Option(help: "Database port")
    private var dbPort: Int = 27017

    @Option(help: "Database name")
    private var dbName: String = ""

    @Option(help: "Database username")
    private var dbLogin: String = ""

    @Option(help: "Database password")
    private var dbPassword: String = ""

    // func makeDocument() -> BSON {
    //     let bson = BSON()
        
    //     defer {
    //         bson.close()
    //     }

    //     bson.append(key: "foo", string: "bar")

    //     return bson
    // }

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
            // let collection = dbClient.getDatabase(name: "agk").getCollection(name: EXPERIMENTS_COLLECTION_NAME)
            // print(collection)
            // let fnd = collection?.find(query: BSON())
            // let insertionResult = collection?.save(document: makeDocument())
            // print(insertionResult.debugDescription)

            let obj = Experiment()
            obj.id = obj.newUUID()
            obj.isCompleted = false

            try obj.save()

            let params = parseRequestParameter(request: request, paramName: "model", flag: "-m") + parseRequestParameter(request: request, paramName: "dataset", flag: "-d")
            var command = try CrossValidate.parse(params)
            let metrics = try command.run()

            obj.metrics = metrics

            try obj.save()
            
            response.appendBody(string: "<html><title>Completed!</title><body>(fnd)</body></html>") // result
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
        print("Connecting to the databased on \(dbHost)...")

        MongoDBConnection.host = dbHost
        MongoDBConnection.database = dbName
        MongoDBConnection.port = dbPort

        MongoDBConnection.authmode = .standard
        MongoDBConnection.username = dbLogin
        MongoDBConnection.password = dbPassword
        
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
