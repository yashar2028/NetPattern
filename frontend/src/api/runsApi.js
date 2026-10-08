import { API_BASE_URL } from "../constants";
import { apiClient, storedToken } from "./client";

export async function startRun(sandboxId, payload) {
  const response = await apiClient.post(`/v1/sandboxes/${sandboxId}/runs`, payload);
  return response.data;
}

export async function listRuns(params = {}) {
  const response = await apiClient.get("/v1/runs", { params: { limit: 50, ...params } });
  return response.data.data;
}

export async function getRun(runId) {
  const response = await apiClient.get(`/v1/runs/${runId}`);
  return response.data;
}

export async function cancelRun(runId) {
  const response = await apiClient.post(`/v1/runs/${runId}/cancel`);
  return response.data;
}

export async function rerun(runId) {
  const response = await apiClient.post(`/v1/runs/${runId}/rerun`);
  return response.data;
}

/**
 * Server-Sent Events with an Authorization header (EventSource cannot send one).
 * Calls onEvent({id, type, data}) for every event until the stream ends or `signal` aborts.
 */
export async function streamRunEvents(runId, { onEvent, signal }) {
  const response = await fetch(`${API_BASE_URL}/v1/runs/${runId}/events/stream`, {
    headers: { Authorization: `Bearer ${storedToken()}` },
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`event stream failed (${response.status})`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = { id: null, type: "message", data: "" };
      for (const line of chunk.split("\n")) {
        if (line.startsWith("id: ")) event.id = Number(line.slice(4));
        else if (line.startsWith("event: ")) event.type = line.slice(7);
        else if (line.startsWith("data: ")) event.data += line.slice(6);
      }
      if (event.data) onEvent({ ...event, data: JSON.parse(event.data) });
    }
  }
}
