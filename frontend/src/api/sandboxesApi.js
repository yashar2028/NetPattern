import { apiClient } from "./client";

export async function listSandboxes() {
  const response = await apiClient.get("/v1/sandboxes", { params: { limit: 100 } });
  return response.data.data;
}

export async function createSandbox(payload) {
  const response = await apiClient.post("/v1/sandboxes", payload);
  return response.data;
}

export async function getSandbox(sandboxId) {
  const response = await apiClient.get(`/v1/sandboxes/${sandboxId}`);
  return response.data;
}

export async function upgradeSandbox(sandboxId) {
  const response = await apiClient.post(`/v1/sandboxes/${sandboxId}/environment/upgrade`);
  return response.data;
}

export async function environmentTemplates() {
  const response = await apiClient.get("/v1/environment-templates");
  return response.data;
}

export async function listPipelines(sandboxId) {
  const response = await apiClient.get(`/v1/sandboxes/${sandboxId}/pipelines`, {
    params: { limit: 100 },
  });
  return response.data.data;
}

export async function createPipeline(sandboxId, payload) {
  const response = await apiClient.post(`/v1/sandboxes/${sandboxId}/pipelines`, payload);
  return response.data;
}

export async function getPipeline(pipelineId) {
  const response = await apiClient.get(`/v1/pipelines/${pipelineId}`);
  return response.data;
}

export async function listVersions(pipelineId) {
  const response = await apiClient.get(`/v1/pipelines/${pipelineId}/versions`);
  return response.data;
}

export async function getVersion(pipelineId, number) {
  const response = await apiClient.get(`/v1/pipelines/${pipelineId}/versions/${number}`);
  return response.data;
}

export async function saveVersion(pipelineId, spec) {
  const response = await apiClient.post(`/v1/pipelines/${pipelineId}/versions`, { spec });
  return response.data;
}

/** Editor features answered by the sandbox's own engine session. */
export async function sessionCall(sandboxId, method, params = {}) {
  const response = await apiClient.post(`/v1/sandboxes/${sandboxId}/session/${method}`, params);
  return response.data;
}

export async function getUsage() {
  const response = await apiClient.get("/v1/usage");
  return response.data;
}
