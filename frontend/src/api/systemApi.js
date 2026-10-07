import { apiClient } from "./client";

export async function getHealth() {
  const response = await apiClient.get("/health", { timeout: 5000 });
  return response.data;
}

export async function getInfo() {
  const response = await apiClient.get("/v1/info");
  return response.data;
}
