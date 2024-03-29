import Foundation

public func read(path: String) throws -> String {
    let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/agk/Utils.swift", with: ""))
    return try String(
            contentsOf: dir.appendingPathComponent("data").appendingPathComponent(path),
            encoding: .utf8
    )
}

public func readLines(path: String) throws -> [String] {
    return try read(path: path).components(separatedBy: "\n")
}

public func writeLines(path: String, lines: [String]) throws {
    let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/agk/Utils.swift", with: ""))
    try lines.joined(separator: "\n").write(
            to: dir.appendingPathComponent("data").appendingPathComponent(path),
            atomically: true,
            encoding: .utf8
    )
}

extension Array where Element == ReportEntry {
    @inlinable public subscript(key: String) -> ReportEntry? {
        for element in self {
            if element.header == key {
                return element
            }
        }
        return nil
    }
}
