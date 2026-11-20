import { TEMPLATES } from "./defaults.js";

export async function createProject({ form, pipelineName, pipelineTemplate, existingSandbox }, api, onSandboxCreated) {
  const selectedTemplate = TEMPLATES.find((item) => item.id === pipelineTemplate);
  if (!selectedTemplate) throw new Error("Please choose a valid pipeline starting point.");

  const sandbox = existingSandbox || await api.createSandbox({ ...form, team_id: form.team_id || null });
  if (!existingSandbox) onSandboxCreated(sandbox);
  const pipeline = await api.createPipeline(sandbox.id, { name: pipelineName, spec: selectedTemplate.spec() });
  return { sandbox, pipeline };
}
