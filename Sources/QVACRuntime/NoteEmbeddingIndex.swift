import Foundation

/// In-memory semantic index over Note embeddings. Holds one vector per Note and
/// returns the Notes most similar to a query vector by cosine similarity, above a
/// caller-supplied threshold. Pure Swift (no SDK dependency) so the ranking and
/// threshold logic is unit-testable off-device.
final class NoteEmbeddingIndex {
    private struct Record {
        let noteID: NoteID
        let vector: [Float]
        let norm: Float
    }

    private var records: [Record] = []

    func rebuild(from notes: [Note], provider: NoteEmbeddingProvider) throws {
        records = try notes.map { note in
            let vector = try provider.embed("\(note.title)\n\(note.body)")
            return Record(noteID: note.id, vector: vector, norm: Self.norm(vector))
        }
    }

    /// Note IDs whose embedding cosine-similarity to `queryVector` is at least
    /// `threshold`, most-similar first, capped to `topK`.
    func search(queryVector: [Float], topK: Int, threshold: Float) -> [NoteID] {
        let queryNorm = Self.norm(queryVector)
        guard queryNorm > 0 else { return [] }

        let scored: [(noteID: NoteID, score: Float)] = records.compactMap { record in
            guard record.norm > 0 else { return nil }
            let score = Self.dot(record.vector, queryVector) / (record.norm * queryNorm)
            return score >= threshold ? (record.noteID, score) : nil
        }

        return scored
            .sorted { $0.score > $1.score }
            .prefix(topK)
            .map { $0.noteID }
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        let count = min(a.count, b.count)
        var sum: Float = 0
        for i in 0..<count {
            sum += a[i] * b[i]
        }
        return sum
    }

    private static func norm(_ vector: [Float]) -> Float {
        var sum: Float = 0
        for value in vector {
            sum += value * value
        }
        return sum.squareRoot()
    }
}
