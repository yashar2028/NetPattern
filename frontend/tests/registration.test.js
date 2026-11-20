import assert from "node:assert/strict";
import test from "node:test";

import { registrationPayload } from "../src/features/registration.js";

const form = { email: "researcher@example.test", password: "password1", confirm: "password1", full_name: "  Alex  ", purpose: "  Research  ", accepted: true };
const agreement = { version: "test-version" };

test("registration includes the reviewed agreement and purpose", () => {
  assert.deepEqual(registrationPayload(form, agreement), {
    email: form.email,
    password: form.password,
    full_name: "Alex",
    purpose: "Research",
    agreement_version: "test-version",
    accept_agreement: true,
  });
});

test("registration rejects missing consent, missing agreement, blank purpose and mismatched passwords", () => {
  assert.throws(() => registrationPayload({ ...form, accepted: false }, agreement), /accept/);
  assert.throws(() => registrationPayload(form, null), /agreement/);
  assert.throws(() => registrationPayload({ ...form, purpose: " " }, agreement), /purpose/);
  assert.throws(() => registrationPayload({ ...form, confirm: "different" }, agreement), /match/);
});
