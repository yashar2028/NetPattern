import { apiClient } from "./client";

export async function listTeams() {
  const response = await apiClient.get("/v1/teams");
  return response.data;
}

export async function createTeam(name) {
  const response = await apiClient.post("/v1/teams", { name });
  return response.data;
}

export async function addMember(teamId, email, role = "member") {
  const response = await apiClient.post(`/v1/teams/${teamId}/members`, { email, role });
  return response.data;
}

export async function removeMember(teamId, userId) {
  await apiClient.delete(`/v1/teams/${teamId}/members/${userId}`);
}

export async function teamUsage(teamId) {
  const response = await apiClient.get(`/v1/teams/${teamId}/usage`);
  return response.data;
}
