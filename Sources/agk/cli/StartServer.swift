import ArgumentParser
import PerfectHTTP
import PerfectHTTPServer
import Foundation
import StORM
import MongoDBStORM
import Foundation
import Logging

let env = ProcessInfo.processInfo.environment

enum EncodingError: Error {
    case cannotEncode(message: String)
}

extension Encodable {
  func asDictionary() throws -> [String: Any] {
    let data = try JSONEncoder().encode(self)
    guard let dictionary = try JSONSerialization.jsonObject(with: data, options: .allowFragments) as? [String: Any] else {
      throw EncodingError.cannotEncode(message: "Cannot encode the given object") 
    }
    return dictionary
  }
}

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

public let EXPERIMENTS_COLLECTION_NAME = "test-experiments"
public let N_MAX_CONCURRENT_EXPERIMENTS = 2

private var experimentConcurrencySemaphore = DispatchSemaphore(value: N_MAX_CONCURRENT_EXPERIMENTS)
private var nActiveExperimentsLock = NSLock()
private var nActiveExperiments = 0

struct StartServer: ParsableCommand {

    @Option(name: .shortAndLong, help: "Port for the server to listen to")
    private var port: Int = 1719

    @Option(name: .shortAndLong, help: "Default logging level")
    private var loggingLevel: Logger.Level = .debug

    func parseRequestParameter(request: HTTPRequest, paramName: String, flag: String) -> [String] {
        if let paramValue = request.param(name: paramName) {
            return [flag, paramValue]
        } else {
            return []
        }
    }

    func runExperiment(request: HTTPRequest, response: HTTPResponse) {
        let logger = Logger("experiment-runner", loggingLevel)
        response.setHeader(.contentType, value: "application/json")
        do {

            // Initialize a new experiment

            let experiment = Experiment()
            
            experiment.id = experiment.newUUID()
            experiment.isCompleted = false
            experiment.startTimestamp = NSDate().timeIntervalSince1970
            experiment.progress = 0.0

            let params = parseRequestParameter(request: request, paramName: "model", flag: "-m") + parseRequestParameter(request: request, paramName: "dataset", flag: "-d") + parseRequestParameter(request: request, paramName: "task", flag: "-t")
            
            var command = try CrossValidate.parse(params)
            experiment.params = try command.asDictionary()

            response.appendBody(["experiment-id": experiment.id])
            response.completed()
            logger.info("Created experiment \(experiment.id)")

            try experiment.save()
            logger.info("Saved experiment \(experiment.id)")

            DispatchQueue.global(qos: .userInitiated).async {

                // Increment number of active experiments
                
                nActiveExperimentsLock.lock()
                nActiveExperiments += 1
                nActiveExperimentsLock.unlock()

                // Obtain a semaphore

                experimentConcurrencySemaphore.wait()

                // Run the initialized experiment
                
                logger.info("Started experiment \(experiment.id)")
                experiment.startTimestamp = NSDate().timeIntervalSince1970

                var metrics = [String: Any]()
                try! command.run(&metrics)

                if experiment.progress < 1 {
                    experiment.progress = 1
                }
                experiment.completionTimestamp = NSDate().timeIntervalSince1970
                logger.info("Completed experiment \(experiment.id)")
                experiment.isCompleted = true
                experiment.metrics = metrics as! [String: Float]
                experiment.params = try! command.asDictionary()

                try! experiment.save()
                logger.info("Saved experiment \(experiment.id)")

                // Release a semaphore

                experimentConcurrencySemaphore.signal()

                // Decrement number of running experiments
                
                nActiveExperimentsLock.lock()
                nActiveExperiments -= 1
                nActiveExperimentsLock.unlock()
            }
        } catch {
            response.appendBody(["error": error.localizedDescription])
            logger.error("Cannot run an experiment: \(error.localizedDescription)")
        }
        response.completed()
    }

    func getLoadStatus(request: HTTPRequest, response: HTTPResponse) {
        response.setHeader(.contentType, value: "application/json")
        nActiveExperimentsLock.lock()
        response.appendBody(["value": Float(nActiveExperiments) / Float(N_MAX_CONCURRENT_EXPERIMENTS)])
        nActiveExperimentsLock.unlock()
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
