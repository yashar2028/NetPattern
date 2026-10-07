import axios from "axios";

import { API_BASE_URL } from "../constants";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export function authConfig(token) {
  if (!token) {
    return {};
  }

  return {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  };
}
