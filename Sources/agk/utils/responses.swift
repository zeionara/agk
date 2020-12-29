import PerfectHTTP
import Foundation

public extension HTTPResponse {
    func appendBody<ValueType>(_ body: [String: ValueType]) where ValueType: Encodable {
        self.appendBody(string: String(data: try! JSONEncoder().encode(body), encoding: .utf8)!)
    }
}
