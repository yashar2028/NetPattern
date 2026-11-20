import assert from "node:assert/strict";
import test from "node:test";

import { createProject } from "../src/features/pipeline/createProject.js";

const form = { name: "Images", template: "pytorch-cpu", team_id: "" };

test("creates a sandbox and the chosen first pipeline", async () => {
  const requests = [];
  const sandbox = { id: "sandbox-1" };
  const result = await createProject(
    { form, pipelineName: "Experiment", pipelineTemplate: "resnet18", existingSandbox: null },
    {
      createSandbox: async (payload) => { requests.push(payload); return sandbox; },
      createPipeline: async (id, payload) => { requests.push({ id, ...payload }); return { id: "pipeline-1" }; },
    },
    (created) => assert.equal(created, sandbox)
  );
  assert.deepEqual(requests[0], { ...form, team_id: null });
  assert.equal(requests[1].id, sandbox.id);
  assert.equal(requests[1].name, "Experiment");
  assert.equal(requests[1].spec.task.type, "classification.single_label");
  assert.equal(result.pipeline.id, "pipeline-1");
});

test("retries pipeline creation without creating a duplicate sandbox", async () => {
  let created;
  const api = {
    createSandbox: async () => ({ id: "sandbox-1" }),
    createPipeline: async () => { throw new Error("pipeline failed"); },
  };
  const input = { form, pipelineName: "Experiment", pipelineTemplate: "unet", existingSandbox: null };
  await assert.rejects(createProject(input, api, (sandbox) => { created = sandbox; }), /pipeline failed/);
  const result = await createProject(
    { ...input, existingSandbox: created },
    {
      createSandbox: async () => { throw new Error("must not create another sandbox"); },
      createPipeline: async (id, { spec }) => ({ id, task: spec.task.type }),
    },
    () => { throw new Error("must not announce another sandbox"); }
  );
  assert.equal(result.pipeline.id, "sandbox-1");
  assert.equal(result.pipeline.task, "segmentation.semantic");
});

test("rejects an invalid template before creating a sandbox", async () => {
  await assert.rejects(
    createProject(
      { form, pipelineName: "Experiment", pipelineTemplate: "missing" },
      { createSandbox: async () => { throw new Error("must not create a sandbox"); } },
      () => {}
    ),
    /valid pipeline starting point/
  );
});
