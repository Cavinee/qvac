/**
 * In-process QVAC answer + embed worklet. Runs inside a react-native-bare-kit
 * BareWorklet: takes a General or Note-grounded Answer request (or a note/query
 * Embed request) over the BareKit push channel, runs the @qvac/sdk model locally,
 * and replies. The Swift runtime owns retrieval and citations; note-grounded
 * requests arrive with the selected note context already chosen, which
 * buildAnswerHistory feeds to the model as a system message. Embed requests return
 * the EmbeddingGemma vector for a single text so the Swift side can cosine-rank.
 *
 * Must be built with bare-pack — raw node_modules imports don't resolve in a
 * worklet. Never log prompt, response, context, or embed text.
 */

import process from "process";
import fs from "fs";
import path from "path";
import { initializeWorkerCore } from "@qvac/sdk/worker-core";
import { registerPlugin } from "@qvac/sdk/plugins";
import { llmPlugin } from "@qvac/sdk/llamacpp-completion/plugin";
import { embeddingsPlugin } from "@qvac/sdk/llamacpp-embedding/plugin";
import {
  loadModel,
  completion,
  embed,
  LLAMA_3_2_1B_INST_Q4_0,
  EMBEDDINGGEMMA_300M_Q4_0
} from "@qvac/sdk";
import { buildAnswerHistory } from "./build-answer-history.mjs";

const PROTOCOL = "qvac.embeddedHost.answer.v1";
const REQUEST_TYPE = "qvac.host.answer";
const RESPONSE_TYPE = "qvac.host.answer.response";
const EMBED_REQUEST_TYPE = "qvac.host.embed";
const EMBED_RESPONSE_TYPE = "qvac.host.embed.response";

let pendingReply = null;
let pendingRequestID = null;
// Which response type the in-flight request expects, so the process-level
// fatal-path handlers below reply with the matching type (answer vs embed).
// Requests are serialized one-at-a-time by the Obj-C bridge, so a single slot
// is safe.
let pendingResponseType = RESPONSE_TYPE;

function errorResponse(requestID, errorCode, errorMessage, responseType = RESPONSE_TYPE) {
  return {
    protocol: PROTOCOL,
    type: responseType,
    requestID,
    status: "error",
    errorCode,
    errorMessage
  };
}

function completedResponse(requestID, text) {
  return {
    protocol: PROTOCOL,
    type: RESPONSE_TYPE,
    requestID,
    status: "completed",
    text
  };
}

function embedCompletedResponse(requestID, embedding) {
  return {
    protocol: PROTOCOL,
    type: EMBED_RESPONSE_TYPE,
    requestID,
    status: "completed",
    embedding
  };
}

function settleReply(payload) {
  const reply = pendingReply;
  pendingReply = null;
  pendingRequestID = null;
  pendingResponseType = RESPONSE_TYPE;
  if (reply) {
    try {
      reply(null, Buffer.from(JSON.stringify(payload)));
    } catch {}
  }
}

function describe(reason) {
  if (reason && typeof reason.message === "string") {
    return reason.message;
  }
  return String(reason);
}

// Bare kills the whole process on an unhandled rejection or uncaught exception,
// so catch them here and reply with the error instead of crashing the app.
process.on("unhandledRejection", (reason) => {
  settleReply(errorResponse(pendingRequestID, "unhandled-rejection", describe(reason), pendingResponseType));
});
process.on("uncaughtException", (error) => {
  settleReply(errorResponse(pendingRequestID, "uncaught-exception", describe(error), pendingResponseType));
});

// The SDK keeps its models and lock file under `${HOME_DIR}/.qvac`. On iOS the
// container root isn't writable, so point HOME_DIR at Documents (which is) and
// create `.qvac` ourselves before initializeWorkerCore writes its lock file there.
// console/logger output doesn't reach os_log this early, so init failures are
// stashed and replied back to the app on the next request.
let resolvedHomeDir = "(unset)";
try {
  const baseHome =
    (process.argv && process.argv[0]) ||
    process.env.HOME ||
    process.env.USERPROFILE ||
    "/tmp";
  resolvedHomeDir = path.join(baseHome, "Documents");
  if (Array.isArray(process.argv)) {
    process.argv[0] = resolvedHomeDir;
  }
  process.env.HOME = resolvedHomeDir;
  const qvacDir = path.join(resolvedHomeDir, ".qvac");
  fs.mkdirSync(qvacDir, { recursive: true });
  if (!fs.existsSync(qvacDir)) {
    globalThis.__qvacAnswerInitError = "qvac dir missing after mkdir: " + qvacDir;
  }
} catch (error) {
  globalThis.__qvacAnswerInitError =
    "mkdir .qvac failed home=" + resolvedHomeDir + " err=" + describe(error);
}

try {
  initializeWorkerCore();
} catch (error) {
  if (!globalThis.__qvacAnswerInitError) {
    globalThis.__qvacAnswerInitError =
      "initCore home=" + resolvedHomeDir + " err=" + describe(error);
  }
}

// Register both the completion and embedding plugins. The plugin registry lives
// in the shared native addon and outlives a single worklet, so a repeat
// registration on a later instance is fine to ignore.
function registerPluginIgnoringDuplicate(plugin) {
  try {
    registerPlugin(plugin);
  } catch (error) {
    if (!String(describe(error)).toLowerCase().includes("already registered")) {
      if (!globalThis.__qvacAnswerInitError) {
        globalThis.__qvacAnswerInitError = "registerPlugin err=" + describe(error);
      }
    }
  }
}
registerPluginIgnoringDuplicate(llmPlugin);
registerPluginIgnoringDuplicate(embeddingsPlugin);

let modelIdPromise = null;
function ensureModelLoaded() {
  if (modelIdPromise === null) {
    modelIdPromise = loadModel({
      modelSrc: LLAMA_3_2_1B_INST_Q4_0,
      modelType: "llm"
    }).catch((error) => {
      modelIdPromise = null;
      throw error;
    });
  }
  return modelIdPromise;
}

let embeddingModelIdPromise = null;
function ensureEmbeddingModelLoaded() {
  if (embeddingModelIdPromise === null) {
    // loadModel infers modelType ("embedding") from the EmbeddingGemma modelSrc.
    embeddingModelIdPromise = loadModel({
      modelSrc: EMBEDDINGGEMMA_300M_Q4_0
    }).catch((error) => {
      embeddingModelIdPromise = null;
      throw error;
    });
  }
  return embeddingModelIdPromise;
}

async function generate(request) {
  const modelId = await ensureModelLoaded();

  // For a note-grounded request, buildAnswerHistory prepends a system message that
  // constrains the model to the Swift-selected notes; general stays prompt-only.
  // Await only `final` — iterating events/tokenStream in-process can leave a
  // sibling promise rejecting with no handler, which Bare treats as fatal.
  const run = completion({
    modelId,
    history: buildAnswerHistory(request),
    stream: false
  });

  const final = await run.final;
  return final.contentText ?? (final.raw && final.raw.fullText) ?? "";
}

async function embedText(request) {
  const modelId = await ensureEmbeddingModelLoaded();
  // `embed` is a single awaitable promise (no event/stream surfaces), so there is
  // no floating sibling promise to leak.
  const result = await embed({ modelId, text: request.text });
  return result.embedding;
}

BareKit.on("push", (data, reply) => {
  let requestID = null;
  let requestType = null;

  try {
    const request = JSON.parse(data.toString());
    requestID = typeof request.requestID === "string" ? request.requestID : null;
    requestType = typeof request.type === "string" ? request.type : null;
  } catch (error) {
    try {
      reply(null, Buffer.from(JSON.stringify(errorResponse(null, "invalid-request", describe(error)))));
    } catch {}
    return;
  }

  // Remember the reply (and its expected response type) so the process-level
  // handlers above can settle it on a fatal path.
  pendingReply = reply;
  pendingRequestID = requestID;
  pendingResponseType = requestType === EMBED_REQUEST_TYPE ? EMBED_RESPONSE_TYPE : RESPONSE_TYPE;

  (async () => {
    try {
      const request = JSON.parse(data.toString());

      if (globalThis.__qvacAnswerInitError) {
        settleReply(errorResponse(requestID, "worker-init-failed", globalThis.__qvacAnswerInitError, pendingResponseType));
        return;
      }
      if (request.protocol !== PROTOCOL) {
        settleReply(errorResponse(requestID, "invalid-protocol", "Invalid protocol in request.", pendingResponseType));
        return;
      }

      if (request.type === EMBED_REQUEST_TYPE) {
        if (typeof request.text !== "string" || request.text.length === 0) {
          settleReply(errorResponse(requestID, "invalid-text", "Embed request is missing text.", EMBED_RESPONSE_TYPE));
          return;
        }
        const embedding = await embedText(request);
        settleReply(embedCompletedResponse(requestID, embedding));
        return;
      }

      if (request.type !== REQUEST_TYPE) {
        settleReply(errorResponse(requestID, "invalid-protocol", "Invalid type in request."));
        return;
      }
      if (typeof request.prompt !== "string" || request.prompt.length === 0) {
        settleReply(errorResponse(requestID, "invalid-prompt", "Answer request is missing a prompt."));
        return;
      }

      const text = await generate(request);
      settleReply(completedResponse(requestID, text));
    } catch (error) {
      settleReply(errorResponse(requestID, "generation-failed", describe(error), pendingResponseType));
    }
  })().catch((error) => {
    settleReply(errorResponse(requestID, "answer-handler-failed", describe(error), pendingResponseType));
  });
});
