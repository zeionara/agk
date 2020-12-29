import PerfectHTTP
import Logging
import Foundation

private var experimentConcurrencySemaphore = DispatchSemaphore(value: N_MAX_CONCURRENT_EXPERIMENTS)

extension StartServer {
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
}
