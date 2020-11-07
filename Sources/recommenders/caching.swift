import Foundation

public func save(object: Any, path: String) {
    do {
        let data = try NSKeyedArchiver.archivedData(withRootObject: object, requiringSecureCoding: false)
        try data.write(to: URL(string: path)!)
    } catch {
        print("Couldn't write file")
    }
}
