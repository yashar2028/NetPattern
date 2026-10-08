import axios from "axios";

import { API_BASE_URL, AUTH_STORAGE_KEY } from "../constants";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export function storedToken() {
  return window.localStorage.getItem(AUTH_STORAGE_KEY) || "";
}

apiClient.interceptors.request.use((config) => {
  const token = storedToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/** A readable message from an API error (problem+json) or a network failure. */
export function errorMessage(error) {
  const data = error?.response?.data;
  if (data?.title) {
    return data.detail ? `${data.title}: ${data.detail}` : data.title;
  }
  return error?.message || "Something went wrong";
}

/** Per-node issues from a problem+json response, if any. */
export function errorIssues(error) {
  return error?.response?.data?.errors || [];
}
