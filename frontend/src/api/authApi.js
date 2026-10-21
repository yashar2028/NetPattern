import { apiClient } from "./client";

export async function registerUser(payload) {
  const response = await apiClient.post("/v1/auth/register", payload);
  return response.data;
}

export async function loginUser(payload) {
  const response = await apiClient.post("/v1/auth/login", payload);
  return response.data;
}

export async function getCurrentUser() {
  const response = await apiClient.get("/v1/auth/me");
  return response.data;
}

export async function listApiKeys() {
  const response = await apiClient.get("/v1/api-keys");
  return response.data;
}

export async function createApiKey(payload) {
  const response = await apiClient.post("/v1/api-keys", payload);
  return response.data;
}

export async function revokeApiKey(keyId) {
  await apiClient.delete(`/v1/api-keys/${keyId}`);
}
