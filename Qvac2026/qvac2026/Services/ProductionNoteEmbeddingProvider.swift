import Foundation
import QVACRuntime

/// Production `NoteEmbeddingProvider` that embeds text on-device by round-tripping to
/// the EmbeddingGemma model living in the one persistent answer worklet (via the
/// `qvac.host.embed` request the worklet already handles).
///
/// - Important: `embed(_:)` blocks the calling thread until the worklet replies, and
///   that reply is delivered on the **main queue**. Calling `embed` on the main thread
///   would therefore deadlock. The runtime guarantees this never happens: the
///   synchronous embedding work runs off the main thread, on the background
///   `DispatchQueue.global` that `KnowledgeRuntimeService.answerAsync` dispatches to —
///   exactly the documented hazard and mitigation of the answer path
///   (`ProductionEmbeddedQVACHostBridgeAdapter`).
final class ProductionNoteEmbeddingProvider: NoteEmbeddingProvider {

    /// Stable id of the EmbeddingGemma-300M-Q4_0 model the worklet loads. Persisted by
    /// the SQLite layer alongside each stored vector so that a future model change
    /// invalidates (re-embeds) stale vectors.
    var modelID: String { "embeddinggemma-300m-q4_0" }

    // Model load + first-run download (~278MB for EmbeddingGemma) can take well over a
    // minute on first call; embedding itself is fast. Matches the answer path's generous
    // budget so a cold start never spuriously times out.
    private let timeoutSeconds: TimeInterval

    init(timeoutSeconds: TimeInterval = 600) {
        self.timeoutSeconds = timeoutSeconds
    }

    func embed(_ text: String) throws -> [Float] {
        let requestID = UUID().uuidString
        let requestPayload = ProductionNoteEmbeddingRequestPayload(requestID: requestID, text: text)
        let requestData = try JSONEncoder().encode(requestPayload)

        // Bridge the callback-based Obj-C transport to a synchronous call. The reply
        // lands on the main queue, so this MUST be invoked off the main thread (see the
        // type doc) or it deadlocks. Mirrors the DispatchSemaphore + timeout mechanism in
        // ProductionEmbeddedQVACHostBridgeAdapter.
        let semaphore = DispatchSemaphore(value: 0)
        var responseData: Data?
        var bridgeError: Error?

        ProductionEmbeddedQVACBareHostBridge.sendEmbedRequest(requestData) { data, error in
            responseData = data
            bridgeError = error
            semaphore.signal()
        }

        guard semaphore.wait(timeout: .now() + timeoutSeconds) == .success else {
            throw ProductionNoteEmbeddingProviderError.timeout
        }

        if let bridgeError {
            throw bridgeError
        }
        guard let responseData else {
            throw ProductionNoteEmbeddingProviderError.invalidResponse
        }

        let responsePayload = try JSONDecoder().decode(
            ProductionNoteEmbeddingResponsePayload.self,
            from: responseData
        )

        guard responsePayload.protocolValue == ProductionNoteEmbeddingRequestPayload.protocolName,
              responsePayload.type == "qvac.host.embed.response",
              responsePayload.requestID == requestID else {
            throw ProductionNoteEmbeddingProviderError.invalidResponse
        }

        switch responsePayload.status {
        case "completed":
            guard let embedding = responsePayload.embedding else {
                throw ProductionNoteEmbeddingProviderError.invalidResponse
            }
            return embedding
        case "error":
            throw ProductionNoteEmbeddingProviderError.embedFailed(
                code: responsePayload.errorCode ?? "unknown",
                message: responsePayload.errorMessage ?? "Unknown error"
            )
        default:
            throw ProductionNoteEmbeddingProviderError.invalidResponse
        }
    }
}

enum ProductionNoteEmbeddingProviderError: Error {
    case invalidResponse
    case timeout
    case embedFailed(code: String, message: String)
}

// MARK: - Wire protocol types

private struct ProductionNoteEmbeddingRequestPayload: Codable {
    static let protocolName = "qvac.embeddedHost.answer.v1"

    let protocolValue: String
    let type: String
    let requestID: String
    let text: String

    init(requestID: String, text: String) {
        self.protocolValue = Self.protocolName
        self.type = "qvac.host.embed"
        self.requestID = requestID
        self.text = text
    }

    enum CodingKeys: String, CodingKey {
        case protocolValue = "protocol"
        case type
        case requestID
        case text
    }
}

private struct ProductionNoteEmbeddingResponsePayload: Codable {
    let protocolValue: String
    let type: String
    let requestID: String?
    let status: String
    let embedding: [Float]?
    let errorCode: String?
    let errorMessage: String?

    enum CodingKeys: String, CodingKey {
        case protocolValue = "protocol"
        case type
        case requestID
        case status
        case embedding
        case errorCode
        case errorMessage
    }
}
