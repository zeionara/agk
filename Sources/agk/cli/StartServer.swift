import ArgumentParser
import PerfectHTTP
import PerfectHTTPServer
import Foundation
import StORM
import MongoDBStORM
import Foundation
import Logging

public let EXPERIMENTS_COLLECTION_NAME = config["db"]["collections"]["experiments"].string ?? "experiments"
public let N_MAX_CONCURRENT_EXPERIMENTS = config["n-max-concurrent-experiments"].int ?? 1

struct StartServer: ParsableCommand {

    @Option(name: .shortAndLong, help: "Port for the server to listen to")
    private var port: Int = 1719

    @Option(name: .shortAndLong, help: "Default logging level")
    var loggingLevel: Logger.Level = .debug

    func parseRequestParameter(request: HTTPRequest, paramName: String, flag: String) -> [String] {
        if let paramValue = request.param(name: paramName) {
            return [flag, paramValue]
        } else {
            return []
        }
    }

    mutating func run(_ result: inout [String: Any]) throws {
        let logger = Logger("root", loggingLevel)

        logger.trace("Starting an http server...")
        logger.trace("Connecting to the databased on \(env["AGK_DB_HOST"]!)...")

        MongoDBConnection.host = env["AGK_DB_HOST"]!
        MongoDBConnection.database = env["AGK_DB_NAME"]!
        MongoDBConnection.port = Int(env["AGK_DB_PORT"]!)!

        MongoDBConnection.authmode = .standard
        MongoDBConnection.username = env["AGK_DB_LOGIN"]!
        MongoDBConnection.password = env["AGK_DB_PASSWORD"]!
        
        var routes = Routes()
        routes.add(method: .get, uri: "/", handler: runExperiment)
        routes.add(method: .get, uri: "/load-status", handler: getLoadStatus)

        let password: String = env["AGK_PASSWORD"]!
        let login: String = env["AGK_LOGIN"]!
        
        try HTTPServer.launch(
            name: "localhost",
            port: port,
            routes: routes,
            requestFilters: [
                (
                    AuthenticationFilter(
                        targetCredentials: encryptCredentials(login: login.isEmpty ? Optional.none : login, password: password.isEmpty ? Optional.none : password)
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
