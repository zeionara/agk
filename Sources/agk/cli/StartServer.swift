import ArgumentParser
import PerfectHTTP
import PerfectHTTPServer
import Foundation
import StORM
import MongoDBStORM
import Foundation

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

    func parseRequestParameter(request: HTTPRequest, paramName: String, flag: String) -> [String] {
        if let paramValue = request.param(name: paramName) {
            return [flag, paramValue]
        } else {
            return []
        }
    }

    func runExperiment(request: HTTPRequest, response: HTTPResponse) {
        do {

            // Initialize a new experiment

            let experiment = Experiment()
            
            experiment.id = experiment.newUUID()
            experiment.isCompleted = false
            experiment.startTimestamp = NSDate().timeIntervalSince1970
            experiment.progress = 0.0

            let params = parseRequestParameter(request: request, paramName: "model", flag: "-m") + parseRequestParameter(request: request, paramName: "dataset", flag: "-d")
            
            var command = try CrossValidate.parse(params)
            experiment.params = try command.asDictionary()

            response.setHeader(.contentType, value: "application/json")
            response.appendBody(string: String(data: try! JSONEncoder().encode(["experiment-id": experiment.id]), encoding: .utf8)!)
            response.completed()

            try experiment.save()

            // print(experiment.asData())

            DispatchQueue.global(qos: .userInitiated).async { [self] in

                // Increment number of active experiments
                
                nActiveExperimentsLock.lock()
                // let runningExperimentIndex = nActiveExperiments
                nActiveExperiments += 1
                nActiveExperimentsLock.unlock()

                // Obtain a semaphore

                experimentConcurrencySemaphore.wait()
                
                // for i in 0..<10 {
                //     print("\(runningExperimentIndex): \(i)")
                //     sleep(2)
                // }

                // Run the initialized experiment
                
                var metrics = [String: Any]()
                try! command.run(&metrics)

                if experiment.progress < 1 {
                    experiment.progress = 1
                }
                experiment.completionTimestamp = NSDate().timeIntervalSince1970
                experiment.isCompleted = true
                experiment.metrics = metrics as! [String: Float]
                experiment.params = try! command.asDictionary()

                // print(experiment.asData())

                try! experiment.save()

                // Release a semaphore

                experimentConcurrencySemaphore.signal()

                // Decrement number of running experiments
                
                nActiveExperimentsLock.lock()
                nActiveExperiments -= 1
                nActiveExperimentsLock.unlock()

            }
            // response.setHeader(.contentType, value: "application/json")
            // response.appendBody(string: try experiment.asDataDict(1).jsonEncodedString())
            // response.appendBody(string: String(data: try! JSONEncoder().encode(["experiment-id": experiment.id]), encoding: .utf8)!)
        } catch {
            response.setHeader(.contentType, value: "text/html")
            response.appendBody(string: "<html><title>Exception!</title><body>\(error)</body></html>")
        }
        response.completed()
    }

    func getLoadStatus(request: HTTPRequest, response: HTTPResponse) {
        response.setHeader(.contentType, value: "application/json")
        nActiveExperimentsLock.lock()
        response.appendBody(string: String(data: try! JSONEncoder().encode(["value": Float(nActiveExperiments) / Float(N_MAX_CONCURRENT_EXPERIMENTS)]), encoding: .utf8)!)
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
        print("Starting an http server...")
        print("Connecting to the databased on \(dbHost)...")

        MongoDBConnection.host = dbHost
        MongoDBConnection.database = dbName
        MongoDBConnection.port = dbPort

        MongoDBConnection.authmode = .standard
        MongoDBConnection.username = dbLogin
        MongoDBConnection.password = dbPassword
        
        var routes = Routes()
        routes.add(method: .get, uri: "/", handler: runExperiment)
        routes.add(method: .get, uri: "/load-status", handler: getLoadStatus)
        
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
