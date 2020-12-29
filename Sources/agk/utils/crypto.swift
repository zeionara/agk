import Foundation

func encryptCredentials(login: String?, password: String?) -> String? {
    if let login_ = login, let password_ = password {
        return "\(login_):\(password_)".data(using: .utf8)!.base64EncodedString(options: Data.Base64EncodingOptions(rawValue: 0))
    }
    return Optional.none
}
