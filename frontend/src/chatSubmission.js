export const CHAT_SUBMISSION_INTERVAL_MS = 3000;

export function createChatSubmissionManager({
  intervalMs = CHAT_SUBMISSION_INTERVAL_MS,
  now = () => Date.now(),
} = {}) {
  const lastAcceptedAtBySession = new Map();
  let sequence = 0;

  function remaining(sessionId, at = now()) {
    const lastAcceptedAt = lastAcceptedAtBySession.get(sessionId);
    if (lastAcceptedAt === undefined) return 0;
    return Math.max(0, intervalMs - (at - lastAcceptedAt));
  }

  function begin(sessionId, submit, {bypassCooldown = false} = {}) {
    const acceptedAt = now();
    const remainingMs = remaining(sessionId, acceptedAt);
    if (!bypassCooldown && remainingMs > 0) {
      return {accepted: false, remainingMs};
    }

    if (!bypassCooldown) lastAcceptedAtBySession.set(sessionId, acceptedAt);
    const requestId = `chat-${acceptedAt}-${++sequence}`;
    let promise;
    try {
      promise = Promise.resolve(typeof submit === "function" ? submit(requestId) : undefined);
    } catch (error) {
      promise = Promise.reject(error);
    }
    return {
      accepted: true,
      requestId,
      acceptedAt,
      availableAt: acceptedAt + intervalMs,
      promise,
    };
  }

  return {begin, remaining};
}
