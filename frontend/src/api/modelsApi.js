import { apiClient } from "./client";

export async function listModels() {
  const response = await apiClient.get("/v1/models");
  return response.data;
}

export async function getModel(modelId) {
  const response = await apiClient.get(`/v1/models/${modelId}`);
  return response.data;
}

/** A completed training run becomes a new model ({name}) or a new version ({model_id}). */
export async function registerModel(payload) {
  const response = await apiClient.post("/v1/models", payload);
  return response.data;
}

export async function listDeployments() {
  const response = await apiClient.get("/v1/deployments");
  return response.data;
}

export async function getDeployment(deploymentId) {
  const response = await apiClient.get(`/v1/deployments/${deploymentId}`);
  return response.data;
}

export async function createDeployment(payload) {
  const response = await apiClient.post("/v1/deployments", payload);
  return response.data;
}

export async function setDeploymentState(deploymentId, action) {
  const response = await apiClient.post(`/v1/deployments/${deploymentId}/${action}`);
  return response.data;
}

export async function deleteDeployment(deploymentId) {
  await apiClient.delete(`/v1/deployments/${deploymentId}`);
}

export async function deploymentSchema(deploymentId) {
  const response = await apiClient.get(`/v1/deployments/${deploymentId}/schema`);
  return response.data;
}

export async function deploymentRequests(deploymentId) {
  const response = await apiClient.get(`/v1/deployments/${deploymentId}/requests`);
  return response.data;
}

export async function predict(deploymentId, files, options = {}) {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  form.append("options", JSON.stringify(options));
  const response = await apiClient.post(`/v1/deployments/${deploymentId}/predict`, form);
  return response.data;
}

/** Starts an export job in the model's environment; returns the run. */
export async function exportVersion(modelId, number, formats) {
  const response = await apiClient.post(`/v1/models/${modelId}/versions/${number}/export`, { formats });
  return response.data;
}
