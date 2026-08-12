import Foundation

/// What a chunk of text costs against the pinned text window (`T`) of the
/// `text_encoder` / `duration_predictor` bundles, and what falls off the end
/// when it does not fit.
///
/// `T` is measured in NFKD-decomposed unicode *scalars*, not visible
/// characters. Latin text spends roughly one scalar per character, but Hangul
/// syllables decompose into 2-3 conjoining jamo, so Korean spends ~2.0-2.2 —
/// a 56-character Korean chunk can cost 125 scalars while an equally long
/// English one costs 60. Callers that cap their chunker by character count are
/// therefore budgeting in the wrong unit; this report converts between the two
/// so the cap can be checked against the window it actually has to fit in.
public struct Supertonic3TextWindowReport: Sendable, Equatable {

    /// The text as handed to the synthesizer, before normalization.
    public let source: String

    /// ISO language code used to pick the `<lang>…</lang>` wrapper.
    public let language: String

    /// Visible character count of `source` — the unit chunkers usually cap on.
    public let characterCount: Int

    /// NFKD scalar count after normalization, including the language tags and
    /// any sentence-final punctuation the preprocessor appends. This is the
    /// number that has to fit in `windowLength`.
    public let scalarCount: Int

    /// The pinned `T` axis this report was measured against.
    public let windowLength: Int

    /// Normalized text that survives the window.
    public let survivingText: String

    /// Normalized text the encoder never sees. Non-empty means words —
    /// possibly including the closing `</lang>` tag — are silently dropped.
    public let droppedText: String

    public var isTruncated: Bool { scalarCount > windowLength }

    /// How many scalars overrun the window (0 when it fits).
    public var overflowScalars: Int { max(0, scalarCount - windowLength) }

    /// Scalars spent per visible character. The conversion factor between a
    /// character-count cap and the real window budget.
    public var scalarsPerCharacter: Double {
        characterCount > 0 ? Double(scalarCount) / Double(characterCount) : 0
    }
}

/// Measures text against the model's pinned text window without loading or
/// running any model.
///
/// Used to size a chunker's cap against the window it feeds, and to prove
/// whether a given cap already overruns it. See
/// `Supertonic3Constants.textTFixed` for the compile-time default and
/// `Supertonic3ModelStore.textTokenLength` for the value read off a loaded
/// bundle (a re-converted model may pin a different `T`).
public enum Supertonic3TextWindow {

    /// Report what `text` costs against `windowLength` and what is dropped.
    ///
    /// - Throws: `Supertonic3Error.unsupportedLanguage` for a language outside
    ///   `Supertonic3Constants.availableLanguages`, `.emptyText` when
    ///   normalization leaves nothing behind.
    public static func audit(
        text: String,
        language: String,
        windowLength: Int = Supertonic3Constants.textTFixed
    ) throws -> Supertonic3TextWindowReport {
        guard Supertonic3Constants.availableLanguages.contains(language) else {
            throw Supertonic3Error.unsupportedLanguage(language)
        }
        let processed = Supertonic3UnicodeProcessor.preprocess(text: text, lang: language)
        guard !processed.isEmpty else { throw Supertonic3Error.emptyText }

        let scalars = Array(processed.unicodeScalars)
        let keep = min(scalars.count, max(0, windowLength))

        return Supertonic3TextWindowReport(
            source: text,
            language: language,
            characterCount: text.count,
            scalarCount: scalars.count,
            windowLength: windowLength,
            survivingText: String(String.UnicodeScalarView(scalars[..<keep])),
            droppedText: String(String.UnicodeScalarView(scalars[keep...]))
        )
    }

    /// The longest prefix of `text` that still fits `windowLength`, cut on a
    /// word boundary when one is available.
    ///
    /// Answers "how many characters of *this* text can I actually send?" —
    /// the per-text version of a chunker cap, for texts whose scalar density
    /// differs from the corpus average.
    public static func longestFittingPrefix(
        of text: String,
        language: String,
        windowLength: Int = Supertonic3Constants.textTFixed
    ) throws -> String {
        if try !audit(text: text, language: language, windowLength: windowLength).isTruncated {
            return text
        }
        var candidate = text
        while !candidate.isEmpty {
            let cut =
                candidate.lastIndex(of: " ")
                ?? candidate.index(before: candidate.endIndex)
            candidate = String(candidate[..<cut])
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return "" }
            if try !audit(text: trimmed, language: language, windowLength: windowLength).isTruncated {
                return trimmed
            }
            candidate = trimmed
        }
        return ""
    }
}
