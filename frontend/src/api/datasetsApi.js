import { apiClient } from "./client";

export async function listDatasets() {
  const response = await apiClient.get("/v1/datasets", { params: { limit: 100 } });
  return response.data.data;
}

export async function createDataset(payload) {
  const response = await apiClient.post("/v1/datasets", payload);
  return response.data;
}

export async function getDataset(datasetId) {
  const response = await apiClient.get(`/v1/datasets/${datasetId}`);
  return response.data;
}

export async function uploadVersion(datasetId, file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const response = await apiClient.post(`/v1/datasets/${datasetId}/versions`, form, {
    onUploadProgress: (event) => {
      if (onProgress && event.total) onProgress(event.loaded / event.total);
    },
  });
  return response.data;
}

export async function profileVersion(datasetId, number, payload) {
  const response = await apiClient.post(
    `/v1/datasets/${datasetId}/versions/${number}/profile`,
    payload
  );
  return response.data;
}
