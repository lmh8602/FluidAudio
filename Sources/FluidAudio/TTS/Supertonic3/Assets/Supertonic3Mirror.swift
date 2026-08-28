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

    /// Where an installed asset set came from.
    ///
    /// Recorded next to the models so a later launch can tell a set that
    /// arrived through the HuggingFace *fallback* apart from one that came from
    /// a mirror. The fallback set may pin a narrower text window `T` than the
    /// mirror set; without this marker a device that installed during a mirror
    /// outage would keep the narrower set forever, because the file-existence
    /// check that gates downloads is satisfied by either set.
    public enum AssetSource: Equatable, Sendable {
        case mirror(URL)
        case huggingFace

        static let huggingFaceMarker = "huggingface"

        var markerText: String {
            switch self {
            case .mirror(let url): return url.absoluteString
            case .huggingFace: return Self.huggingFaceMarker
            }
        }

        init?(markerText: String) {
            let text = markerText.trimmingCharacters(in: .whitespacesAndNewlines)
            if text == Self.huggingFaceMarker { self = .huggingFace; return }
            guard let url = URL(string: text), url.scheme != nil else { return nil }
            self = .mirror(url)
        }
    }

    /// File inside the asset directory that records `AssetSource`.
    /// Absent for sets that predate the marker or were placed by hand — those
    /// are left alone.
    public static let sourceMarkerName = ".asset-source"

    public static func installedSource(in directory: URL) -> AssetSource? {
        let url = directory.appendingPathComponent(sourceMarkerName)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        return AssetSource(markerText: text)
    }

    static func writeSourceMarker(_ source: AssetSource, in directory: URL) throws {
        try source.markerText.write(
            to: directory.appendingPathComponent(sourceMarkerName),
            atomically: true, encoding: .utf8)
    }

    /// Sibling directory a set is downloaded into before it is moved into place.
    public static func stagingDirectory(for destination: URL) -> URL {
        destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).partial", isDirectory: true)
    }

    /// Present inside the staging directory once every file has landed and the
    /// swap was deliberately deferred (`download(deferSwap:)`).
    public static let stagedCompleteMarkerName = ".complete"

    /// Move a fully staged set into place. Returns `false` when nothing is
    /// staged. **Call before any model is loaded** — that is the whole reason
    /// a swap gets deferred: replacing files under a loaded set would let a
    /// text encoder pinned to one `T` meet estimator buckets pinned to another.
    @discardableResult
    public static func promoteStagedSet(for destination: URL) throws -> Bool {
        let staging = stagingDirectory(for: destination)
        let complete = staging.appendingPathComponent(stagedCompleteMarkerName)
        guard FileManager.default.fileExists(atPath: complete.path) else { return false }
        try FileManager.default.removeItem(at: complete)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: staging, to: destination)
        logger.info("Promoted staged mirror set into \(destination.path)")
        return true
    }

    /// Fetch every file the mirror lists into `destination`.
    ///
    /// Downloads to a sibling temporary directory and moves it into place only
    /// after the whole set arrives. A half-downloaded bundle would otherwise
    /// satisfy the caller's file-existence check on the next launch and load a
    /// truncated `.mlmodelc` — a failure that looks like model corruption
    /// rather than an interrupted download.
    ///
    /// - Parameter deferSwap: leave the finished set in staging with a
    ///   completion marker instead of moving it into place; a later
    ///   `promoteStagedSet(for:)` does the move. Use it when a set may already
    ///   be loaded from `destination`.
    public static func download(
        from baseURL: URL,
        to destination: URL,
        deferSwap: Bool = false,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        progressHandler?(DownloadProgress(fractionCompleted: 0, phase: .listing))
        let manifest = try await fetchManifest(from: baseURL)
        guard !manifest.isEmpty else {
            throw Supertonic3Error.downloadFailed("mirror manifest is empty: \(baseURL)")
        }
        logger.info("Mirror lists \(manifest.count) file(s) at \(baseURL.absoluteString)")

        let staging = stagingDirectory(for: destination)
        try? FileManager.default.removeItem(at: staging)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        var keepStaging = false
        defer { if !keepStaging { try? FileManager.default.removeItem(at: staging) } }

        for (index, relativePath) in manifest.enumerated() {
            let remote = baseURL.appendingPathComponent(relativePath)
            let local = staging.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: local.deletingLastPathComponent(), withIntermediateDirectories: true)

            let (temporary, response) = try await URLSession.shared.download(from: remote)
            // Non-HTTP responses (a `file://` mirror in tests or a bundled set)
            // have no status code and are taken as success; HTTP must be 200.
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                throw Supertonic3Error.downloadFailed("mirror \(relativePath): HTTP \(http.statusCode)")
            }
            try FileManager.default.moveItem(at: temporary, to: local)
            progressHandler?(
                DownloadProgress(
                    fractionCompleted: Double(index + 1) / Double(manifest.count),
                    phase: .downloading(completedFiles: index + 1, totalFiles: manifest.count)))
        }

        // The marker rides along in staging so it lands atomically with the set.
        try writeSourceMarker(.mirror(baseURL), in: staging)

        if deferSwap {
            try "".write(
                to: staging.appendingPathComponent(stagedCompleteMarkerName),
                atomically: true, encoding: .utf8)
            keepStaging = true
            logger.info("Mirror download staged for the next launch: \(staging.path)")
            return
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
    public static func fetchManifest(from baseURL: URL) async throws -> [String] {
        let url = baseURL.appendingPathComponent(manifestName)
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw Supertonic3Error.downloadFailed("mirror manifest: HTTP \(http.statusCode)")
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
