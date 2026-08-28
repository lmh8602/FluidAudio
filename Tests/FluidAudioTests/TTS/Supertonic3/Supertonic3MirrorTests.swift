import Foundation
import XCTest

@testable import FluidAudio

/// Mirror download, the installed-source marker, and the fallback upgrade in
/// `Supertonic3ResourceDownloader.ensureModels(mirrors:)`.
///
/// Mirrors are `file://` directories here — `Supertonic3Mirror` treats a
/// non-HTTP response as success, so no server is needed. The HuggingFace path
/// itself is never exercised (it needs the network); every case that would
/// reach it is arranged so that it does not.
final class Supertonic3MirrorTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("Supertonic3MirrorTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        root = nil
    }

    // MARK: - Manifest

    func testManifestRejectsPathsThatEscapeTheRoot() async throws {
        let mirror = try makeMirror(named: "escaping", files: ["ok.txt": "1"])
        let manifest = mirror.appendingPathComponent(Supertonic3Mirror.manifestName)
        try "ok.txt\n../outside.txt\n".write(to: manifest, atomically: true, encoding: .utf8)

        do {
            _ = try await Supertonic3Mirror.fetchManifest(from: mirror)
            XCTFail("a `..` segment was accepted")
        } catch Supertonic3Error.downloadFailed(let message) {
            XCTAssertTrue(message.contains("escapes root"), message)
        }
    }

    // MARK: - Marker

    func testMirrorDownloadRecordsItsSourceNextToTheFiles() async throws {
        let mirror = try makeMirror(named: "a", files: ["tts.json": "{}", "Sub/w.bin": "w"])
        let destination = root.appendingPathComponent("dest/supertonic-3", isDirectory: true)

        try await Supertonic3Mirror.download(from: mirror, to: destination)

        XCTAssertEqual(
            try String(contentsOf: destination.appendingPathComponent("Sub/w.bin"), encoding: .utf8), "w")
        XCTAssertEqual(Supertonic3Mirror.installedSource(in: destination), .mirror(mirror))
    }

    func testMarkerRoundTripsBothSources() {
        let url = URL(string: "https://example.org/set")!
        XCTAssertEqual(
            Supertonic3Mirror.AssetSource(markerText: Supertonic3Mirror.AssetSource.mirror(url).markerText),
            .mirror(url))
        XCTAssertEqual(
            Supertonic3Mirror.AssetSource(markerText: "huggingface\n"), .huggingFace)
        XCTAssertNil(Supertonic3Mirror.AssetSource(markerText: "not a url"))
        XCTAssertNil(Supertonic3Mirror.installedSource(in: root), "no marker must read as unknown")
    }

    // MARK: - ensureModels

    /// The first mirror that answers wins; a dead one ahead of it is skipped.
    func testEnsureModelsTriesMirrorsInOrderAndSkipsDeadOnes() async throws {
        let dead = root.appendingPathComponent("nowhere", isDirectory: true)
        let live = try makeRequiredSetMirror(named: "live")

        let repoDir = try await Supertonic3ResourceDownloader.ensureModels(
            directory: root, mirrors: [dead, live])

        XCTAssertEqual(Supertonic3Mirror.installedSource(in: repoDir), .mirror(live))
    }

    /// A complete set from a mirror is left alone — no marker games, no re-download.
    func testEnsureModelsLeavesAMirrorSetAlone() async throws {
        let live = try makeRequiredSetMirror(named: "live")
        let repoDir = try await Supertonic3ResourceDownloader.ensureModels(
            directory: root, mirrors: [live])
        let stamp = try modificationDate(of: repoDir.appendingPathComponent(Supertonic3Mirror.sourceMarkerName))

        let other = try makeRequiredSetMirror(named: "other")
        _ = try await Supertonic3ResourceDownloader.ensureModels(directory: root, mirrors: [other])

        XCTAssertEqual(Supertonic3Mirror.installedSource(in: repoDir), .mirror(live))
        XCTAssertEqual(
            try modificationDate(of: repoDir.appendingPathComponent(Supertonic3Mirror.sourceMarkerName)), stamp)
    }

    /// A set marked as the HuggingFace fallback is **staged** once a mirror
    /// answers — the running process may have loaded the old set — and
    /// promoted by the next call, before that call checks what is installed.
    func testEnsureModelsStagesAFallbackUpgradeAndPromotesItNextTime() async throws {
        let repoDir = try makeInstalledSet(source: .huggingFace)
        let live = try makeRequiredSetMirror(named: "live")

        _ = try await Supertonic3ResourceDownloader.ensureModels(directory: root, mirrors: [live])

        XCTAssertEqual(
            Supertonic3Mirror.installedSource(in: repoDir), .huggingFace,
            "the old set was replaced while it may still be loaded")
        let staging = Supertonic3Mirror.stagingDirectory(for: repoDir)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: staging.appendingPathComponent(Supertonic3Mirror.stagedCompleteMarkerName).path),
            "no completed staging to promote next time")

        // Next launch: even with every mirror dead, the staged set goes in.
        let dead = root.appendingPathComponent("nowhere", isDirectory: true)
        _ = try await Supertonic3ResourceDownloader.ensureModels(directory: root, mirrors: [dead])

        XCTAssertEqual(Supertonic3Mirror.installedSource(in: repoDir), .mirror(live))
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path), "staging left behind")
    }

    /// An interrupted staging (no completion marker) is never promoted.
    func testEnsureModelsIgnoresAnIncompleteStaging() async throws {
        let repoDir = try makeInstalledSet(source: nil)
        let staging = Supertonic3Mirror.stagingDirectory(for: repoDir)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try "half".write(to: staging.appendingPathComponent("tts.json"), atomically: true, encoding: .utf8)

        _ = try await Supertonic3ResourceDownloader.ensureModels(directory: root, mirrors: [])

        XCTAssertNil(Supertonic3Mirror.installedSource(in: repoDir))
        XCTAssertEqual(
            try String(contentsOf: repoDir.appendingPathComponent("tts.json"), encoding: .utf8), "installed")
    }

    /// Fresh installs still swap in immediately — nothing could have been loaded.
    func testEnsureModelsSwapsAFreshInstallImmediately() async throws {
        let live = try makeRequiredSetMirror(named: "live")

        let repoDir = try await Supertonic3ResourceDownloader.ensureModels(directory: root, mirrors: [live])

        XCTAssertEqual(Supertonic3Mirror.installedSource(in: repoDir), .mirror(live))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: Supertonic3Mirror.stagingDirectory(for: repoDir).path))
    }

    /// …but is kept, without throwing, while no mirror answers.
    func testEnsureModelsKeepsAFallbackSetWhenNoMirrorAnswers() async throws {
        let repoDir = try makeInstalledSet(source: .huggingFace)
        let dead = root.appendingPathComponent("nowhere", isDirectory: true)

        let returned = try await Supertonic3ResourceDownloader.ensureModels(
            directory: root, mirrors: [dead])

        XCTAssertEqual(returned, repoDir)
        XCTAssertEqual(Supertonic3Mirror.installedSource(in: repoDir), .huggingFace)
    }

    /// A hand-placed set (no marker) is never touched, even with mirrors configured.
    func testEnsureModelsLeavesAnUnmarkedSetAlone() async throws {
        let repoDir = try makeInstalledSet(source: nil)
        let live = try makeRequiredSetMirror(named: "live")

        _ = try await Supertonic3ResourceDownloader.ensureModels(directory: root, mirrors: [live])

        XCTAssertNil(Supertonic3Mirror.installedSource(in: repoDir))
    }

    // MARK: - Helpers

    private func makeMirror(named name: String, files: [String: String]) throws -> URL {
        let dir = root.appendingPathComponent("mirrors/\(name)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (path, content) in files {
            let url = dir.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try content.write(to: url, atomically: true, encoding: .utf8)
        }
        try files.keys.sorted().joined(separator: "\n")
            .write(to: dir.appendingPathComponent(Supertonic3Mirror.manifestName),
                   atomically: true, encoding: .utf8)
        return dir
    }

    /// A mirror whose manifest covers every file `ensureModels` checks for.
    /// `.mlmodelc` entries are plain files here — the existence check does not
    /// care, and the download never opens them.
    private func makeRequiredSetMirror(named name: String) throws -> URL {
        var files: [String: String] = [:]
        for file in ModelNames.Supertonic3.requiredFiles(veVariant: nil) {
            files[file] = name
        }
        return try makeMirror(named: name, files: files)
    }

    private func makeInstalledSet(source: Supertonic3Mirror.AssetSource?) throws -> URL {
        let repoDir = root.appendingPathComponent(Repo.supertonic3.folderName, isDirectory: true)
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        for file in ModelNames.Supertonic3.requiredFiles(veVariant: nil) {
            let url = repoDir.appendingPathComponent(file)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try "installed".write(to: url, atomically: true, encoding: .utf8)
        }
        if let source {
            try Supertonic3Mirror.writeSourceMarker(source, in: repoDir)
        }
        return repoDir
    }

    private func modificationDate(of url: URL) throws -> Date {
        try XCTUnwrap(FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)
    }
}
