// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "agk",
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/tensorflow/swift-models.git", .branch("main")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "0.2.0"),
        .package(url: "https://github.com/apple/swift-log.git", .branch("main")),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "agk",
            dependencies: [
                .product(name: "Datasets", package: "swift-models"),
                .product(name: "Checkpoints", package: "swift-models"),
                .product(name: "RecommendationModels", package: "swift-models"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log")
            ]),
        .testTarget(
            name: "agkTests",
            dependencies: ["agk"]),
    ]
)
