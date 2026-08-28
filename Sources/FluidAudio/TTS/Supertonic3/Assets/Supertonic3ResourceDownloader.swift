import Foundation

/// Downloads the Supertonic-3 CoreML assets from HuggingFace.
///
/// FluidAudio republishes the upstream ONNX checkpoint as four `.mlmodelc`
/// bundles plus the original `tts.json` + `unicode_indexer.json` companion
/// files at `FluidInference/supertonic-3-coreml`. The bundle layout is
/// produced by `Scripts/convert_supertonic3_to_coreml.py`; see that script
/// for conversion details.
public enum Supertonic3ResourceDownloader {

    private static let logger = AppLogger(category: "Supertonic3ResourceDownloader")

    /// Ensure all required Supertonic-3 model + companion files are present
    /// locally. Returns the resolved repo directory.
    /// - Parameter mirrors: Plain HTTPS directories to try **in order** before
    ///   HuggingFace, each laid out as described by `Supertonic3Mirror`. Use
    ///   them to ship an asset set that is not on HF — a build pinned to a
    ///   different text window `T`, for instance — and to give that set more
    ///   than one home. **Every mirror failing falls back to HuggingFace**
    ///   rather than throwing: unreachable hosts must not cost the app its
    ///   voice. The fallback set may pin a different `T`, which is safe only
    ///   because callers read the window off the loaded model
    ///   (`Supertonic3ModelStore.textTokenLength`) instead of assuming one.
    ///
    ///   A set that arrived through that fallback is marked as such
    ///   (`Supertonic3Mirror.installedSource`). When mirrors are given and the
    ///   installed set carries the fallback marker, this call **tries the
    ///   mirrors again** — otherwise a device that installed during an outage
    ///   would keep the fallback set for good. The replacement is **staged**,
    ///   not swapped in: the running process may already have loaded the old
    ///   set, and the two may pin different `T`. Promotion is the host's job
    ///   (`Supertonic3Mirror.promoteStagedSet`, once at launch before any model
    ///   loads); this function never promotes, and once a completed staging
    ///   exists it does not download again. A mirror set, or a set with no
    ///   marker (placed by hand), is never re-downloaded.
    ///
    ///   Calls for the same directory are serialized: a second caller joins the
    ///   in-flight one and gets its result (its own `progressHandler` stays
    ///   silent). Two downloads into the same staging would delete each other's
    ///   files.
    @discardableResult
    public static func ensureModels(
        directory: URL? = nil,
        veVariant: String? = nil,
        mirrors: [URL] = [],
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = try assetDirectory(under: modelsRoot)
        return try await serializer.run(key: repoDir.path) {
            try await ensureModelsUnserialized(
                modelsRoot: modelsRoot, repoDir: repoDir, veVariant: veVariant,
                mirrors: mirrors, progressHandler: progressHandler)
        }
    }

    /// One in-flight `ensureModels` per asset directory; later callers await it.
    private static let serializer = KeyedTaskSerializer<URL>()

    private static func ensureModelsUnserialized(
        modelsRoot: URL,
        repoDir: URL,
        veVariant: String?,
        mirrors: [URL],
        progressHandler: ProgressHandler?
    ) async throws -> URL {
        let required = ModelNames.Supertonic3.requiredFiles(veVariant: veVariant)
        let allPresent = required.allSatisfy { file in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(file).path)
        }
        let wantsUpgrade = allPresent && !mirrors.isEmpty
            && Supertonic3Mirror.installedSource(in: repoDir) == .huggingFace

        if allPresent && !wantsUpgrade {
            logger.info("Supertonic-3 assets found in cache at \(repoDir.path)")
            return repoDir
        }
        if wantsUpgrade && Supertonic3Mirror.hasCompletedStaging(for: repoDir) {
            logger.info("Upgrade already staged; waiting for the host to promote it")
            return repoDir
        }

        for mirror in mirrors {
            do {
                try await Supertonic3Mirror.download(
                    from: mirror, to: repoDir, deferSwap: wantsUpgrade,
                    progressHandler: progressHandler)
                return repoDir
            } catch {
                logger.warning("Mirror \(mirror.absoluteString) failed (\(error)); trying next")
            }
        }

        if wantsUpgrade {
            // Nothing answered; the fallback set still works. Try again next time.
            logger.warning("No mirror reachable; keeping the HuggingFace fallback set")
            return repoDir
        }

        logger.info("Downloading Supertonic-3 CoreML assets from HuggingFace…")
        do {
            try await ModelHub.download(
                .supertonic3, to: modelsRoot, variant: veVariant,
                progressHandler: progressHandler)
        } catch {
            throw Supertonic3Error.downloadFailed("\(error)")
        }
        // Only mark when a mirror was actually offered — a caller that never
        // configured one has nothing to upgrade to, and the marker would only
        // trigger retries against an empty list.
        if !mirrors.isEmpty {
            try? Supertonic3Mirror.writeSourceMarker(.huggingFace, in: repoDir)
        }
        return repoDir
    }

    /// Where `ensureModels` keeps the set under `directory` (the cache root
    /// when nil). Callers that need to inspect the installed set — its
    /// `Supertonic3Mirror.installedSource`, say — should ask here rather than
    /// rebuild the path.
    public static func assetDirectory(under directory: URL? = nil) throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        return modelsRoot.appendingPathComponent(Repo.supertonic3.folderName)
    }

    /// Single-mirror convenience. `mirror` is required here (no default) so a
    /// call without it resolves to the `mirrors:` overload unambiguously.
    @discardableResult
    public static func ensureModels(
        directory: URL? = nil,
        veVariant: String? = nil,
        mirror: URL?,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        try await ensureModels(
            directory: directory, veVariant: veVariant,
            mirrors: mirror.map { [$0] } ?? [],
            progressHandler: progressHandler)
    }

    /// Ensure a built-in voice style JSON is present locally, downloading it
    /// from `FluidInference/supertonic-3-coreml/voice_styles/` if missing, and
    /// return its local file URL. Custom voices can skip this and load any file
    /// directly via `Supertonic3VoiceStyle.load(from:)`.
    @discardableResult
    public static func downloadVoiceStyle(
        _ voice: Supertonic3Voice,
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.supertonic3.folderName)
        let localURL = repoDir.appendingPathComponent(voice.fileName)

        if FileManager.default.fileExists(atPath: localURL.path) {
            logger.info("Supertonic-3 voice \(voice.rawValue) found in cache")
            return localURL
        }

        logger.info("Downloading Supertonic-3 voice \(voice.rawValue) from HuggingFace…")
        do {
            // The HF tree API only lists directories, so pull the single file
            // out of voice_styles/ by skipping every other entry.
            try await ModelHub.download(
                .supertonic3,
                subdirectory: "voice_styles",
                to: repoDir,
                progressHandler: progressHandler,
                shouldSkip: { $0 != voice.fileName }
            )
        } catch {
            throw Supertonic3Error.downloadFailed("voice \(voice.rawValue): \(error)")
        }

        guard FileManager.default.fileExists(atPath: localURL.path) else {
            throw Supertonic3Error.downloadFailed(
                "voice \(voice.rawValue) missing after download")
        }
        return localURL
    }

    /// Download (if needed) and decode a built-in voice style in one call.
    public static func loadVoiceStyle(
        _ voice: Supertonic3Voice,
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> Supertonic3VoiceStyle {
        let url = try await downloadVoiceStyle(
            voice, directory: directory, progressHandler: progressHandler)
        return try Supertonic3VoiceStyle.load(from: url)
    }

    private static func defaultCacheRoot() throws -> URL {
        // Delegate to the shared TTS cache root (Application Support on iOS,
        // ~/.cache/fluidaudio on macOS) so all backends share one location.
        let root = try TtsCacheDirectory.ensure().appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}

/// Coalesces concurrent async operations by key: the first caller runs the
/// operation, later callers for the same key await the same task and receive
/// its result (or error). Once it finishes the key is free again.
actor KeyedTaskSerializer<Value: Sendable> {
    private var inflight: [String: Task<Value, Error>] = [:]

    func run(
        key: String,
        _ operation: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        if let running = inflight[key] {
            return try await running.value
        }
        let task = Task { try await operation() }
        inflight[key] = task
        defer { inflight[key] = nil }
        return try await task.value
    }
}
