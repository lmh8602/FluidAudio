import Foundation

/// Downloads a Supertonic-3 asset set from a plain HTTPS directory instead of
/// HuggingFace.
///
/// The upstream downloader resolves `.mlmodelc` **directories** through the HF
/// tree API, so it cannot point at an arbitrary static file host — nothing
/// tells it which files live inside each bundle. A mirror therefore publishes
/// `manifest.txt`: one repo-relative file path per line, which is exactly the
/// listing the tree API would have provided.
///
/// This exists because re-converted model sets (a different pinned text window
/// `T`, say) are not on HuggingFace. Hosting one is otherwise a static-file
/// problem; the manifest is the only piece the client was missing.
public enum Supertonic3Mirror {

    private static let logger = AppLogger(category: "Supertonic3Mirror")

    public static let manifestName = "manifest.txt"

    /// Fetch every file the mirror lists into `destination`.
    ///
    /// Downloads to a sibling temporary directory and moves it into place only
    /// after the whole set arrives. A half-downloaded bundle would otherwise
    /// satisfy the caller's file-existence check on the next launch and load a
    /// truncated `.mlmodelc` — a failure that looks like model corruption
    /// rather than an interrupted download.
    public static func download(
        from baseURL: URL,
        to destination: URL,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        progressHandler?(DownloadProgress(fractionCompleted: 0, phase: .listing))
        let manifest = try await fetchManifest(from: baseURL)
        guard !manifest.isEmpty else {
            throw Supertonic3Error.downloadFailed("mirror manifest is empty: \(baseURL)")
        }
        logger.info("Mirror lists \(manifest.count) file(s) at \(baseURL.absoluteString)")

        let staging = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).partial", isDirectory: true)
        try? FileManager.default.removeItem(at: staging)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }

        for (index, relativePath) in manifest.enumerated() {
            let remote = baseURL.appendingPathComponent(relativePath)
            let local = staging.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: local.deletingLastPathComponent(), withIntermediateDirectories: true)

            let (temporary, response) = try await URLSession.shared.download(from: remote)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                throw Supertonic3Error.downloadFailed("mirror \(relativePath): HTTP \(code)")
            }
            try FileManager.default.moveItem(at: temporary, to: local)
            progressHandler?(
                DownloadProgress(
                    fractionCompleted: Double(index + 1) / Double(manifest.count),
                    phase: .downloading(completedFiles: index + 1, totalFiles: manifest.count)))
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: staging, to: destination)
        logger.info("Mirror download complete: \(destination.path)")
    }

    /// Repo-relative paths listed by the mirror.
    ///
    /// Absolute paths and `..` segments are rejected — the manifest decides
    /// where bytes land on disk, so a hostile or broken one must not be able to
    /// name a path outside the destination.
    static func fetchManifest(from baseURL: URL) async throws -> [String] {
        let url = baseURL.appendingPathComponent(manifestName)
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw Supertonic3Error.downloadFailed("mirror manifest: HTTP \(code)")
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw Supertonic3Error.downloadFailed("mirror manifest is not UTF-8")
        }
        return
            try text
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .map { path in
                guard !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
                    throw Supertonic3Error.downloadFailed("mirror manifest escapes root: \(path)")
                }
                return path
            }
    }
}
