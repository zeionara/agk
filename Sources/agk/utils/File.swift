import Foundation

public func getProjectRoot() -> URL {
    URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/agk/utils/File.swift", with: ""))
}

public func getCacheRoot() -> URL {
    let url = getProjectRoot().appendingPathComponent("data").appendingPathComponent("cache")
    createDirectory(url)
    return url
}

public func getTensorsCacheRoot() -> URL {
    let url = getCacheRoot().appendingPathComponent("tensors")
    createDirectory(url)
    return url
}

public func getModelsCacheRoot() -> URL {
    let url = getTensorsCacheRoot().appendingPathComponent("models")
    createDirectory(url)
    return url
}

public func getTensorsDatasetCacheRoot() -> URL {
    let url = getTensorsCacheRoot().appendingPathComponent("datasets")
    createDirectory(url)
    return url
}

public func createDirectory(_ path: URL) {
    do {
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true, attributes: nil)
    }
    catch let error as NSError
    {
        NSLog("Unable to create directory \(error.debugDescription)")
    }
}
