export function registrationPayload(form, agreement) {
  if (!agreement?.version) throw new Error("The agreement could not be loaded. Please try again.");
  if (!form.purpose.trim()) throw new Error("Please describe your purpose of use.");
  if (form.password !== form.confirm) throw new Error("Passwords do not match.");
  if (!form.accepted) throw new Error("Please accept the agreement to continue.");
  return {
    email: form.email,
    password: form.password,
    full_name: form.full_name.trim() || null,
    purpose: form.purpose.trim(),
    agreement_version: agreement.version,
    accept_agreement: true,
  };
}
