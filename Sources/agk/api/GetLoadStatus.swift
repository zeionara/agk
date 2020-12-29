import PerfectHTTP
import Foundation

var nActiveExperimentsLock = NSLock()
var nActiveExperiments = 0

extension StartServer {
    func getLoadStatus(request: HTTPRequest, response: HTTPResponse) {
        response.setHeader(.contentType, value: "application/json")
        nActiveExperimentsLock.lock()
        response.appendBody(["value": Float(nActiveExperiments) / Float(N_MAX_CONCURRENT_EXPERIMENTS)])
        nActiveExperimentsLock.unlock()
        response.completed()
    }
}
