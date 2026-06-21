import Foundation

/// Produces an embedding vector for a piece of text. The production implementation
/// bridges the on-device embedding model (the worklet `embed()` call); tests inject
/// a deterministic fake. Synchronous from the runtime's point of view — the async
/// worklet call is bridged off the main thread at the adapter boundary, exactly like
/// `AIRuntimeAdapter` for generation.
public protocol NoteEmbeddingProvider {
    func embed(_ text: String) throws -> [Float]
}

/// Deterministic, model-free embedding for off-device tests. Hashes content tokens
/// into a fixed-width bag-of-words vector, so texts that share content words land
/// close in cosine space while unrelated texts stay near-orthogonal. Uses a stable
/// FNV-1a hash (NOT Swift's per-process-seeded `Hasher`) so vectors are identical
/// across runs. The real semantic quality is verified on-device with EmbeddingGemma.
public struct FakeEmbeddingProvider: NoteEmbeddingProvider {
    private let dimensions: Int

    public init(dimensions: Int = 512) {
        self.dimensions = dimensions
    }

    public func embed(_ text: String) throws -> [Float] {
        var vector = [Float](repeating: 0, count: dimensions)
        let tokens = text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        for token in tokens where token.count > 1 {
            let index = Int(Self.fnv1a(token) % UInt64(dimensions))
            vector[index] += 1
        }
        return vector
    }

    private static func fnv1a(_ token: Substring) -> UInt64 {
        var hash: UInt64 = 1469598103934665603 // FNV-1a 64-bit offset basis
        for byte in token.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 1099511628211 // FNV-1a 64-bit prime
        }
        return hash
    }
}
