/**
 * A small form renderer for the engine's JSON Schema (PLAN P6): the forms follow the
 * sandbox's own engine version, so new options need no frontend code.
 *
 * Supports what the engine's pydantic models produce: objects, $ref/$defs, nullable
 * fields (anyOf with null), discriminated unions (oneOf + discriminator), enums,
 * numbers, booleans, strings, tuples and lists. Anything else is edited as JSON.
 * Only values the user sets are written; everything else keeps the engine default.
 */
import { useState } from "react";

function resolve(schema, root) {
  let current = schema || {};
  while (current.$ref) {
    const name = current.$ref.split("/").pop();
    const { $ref: _ref, ...rest } = current;
    current = { ...(root.$defs?.[name] || {}), ...rest };
  }
  return current;
}

function prettify(name) {
  return name.replaceAll("_", " ");
}

function setKey(object, key, value) {
  const next = { ...(object || {}) };
  if (value === undefined) delete next[key];
  else next[key] = value;
  return next;
}

export default function SchemaForm({ schema, root = schema, value, onChange, hidden = [] }) {
  const resolved = resolve(schema, root);
  if (resolved.oneOf && resolved.discriminator) {
    return <UnionFields schema={resolved} root={root} value={value || {}} onChange={onChange} hidden={hidden} />;
  }
  return <ObjectFields schema={resolved} root={root} value={value || {}} onChange={onChange} hidden={hidden} />;
}

function ObjectFields({ schema, root, value, onChange, hidden = [] }) {
  const properties = Object.entries(schema.properties || {}).filter(([key]) => !hidden.includes(key));
  const required = new Set(schema.required || []);
  return (
    <div className="schema-fields">
      {properties.map(([key, property]) => (
        <Field
          key={key}
          name={key}
          schema={property}
          root={root}
          required={required.has(key)}
          value={value[key]}
          onChange={(next) => onChange(setKey(value, key, next))}
        />
      ))}
    </div>
  );
}

function UnionFields({ schema, root, value, onChange, hidden = [] }) {
  const key = schema.discriminator.propertyName;
  const options = Object.entries(schema.discriminator.mapping || {});
  const current = value[key] ?? options[0]?.[0];
  const variant = resolve({ $ref: schema.discriminator.mapping?.[current] }, root);
  return (
    <div className="schema-fields">
      <label className="field">
        <span>{prettify(key)}</span>
        <select value={current} onChange={(event) => onChange({ [key]: event.target.value })}>
          {options.map(([option]) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
      <ObjectFields
        schema={variant}
        root={root}
        value={{ ...value, [key]: current }}
        onChange={(next) => onChange({ ...next, [key]: current })}
        hidden={[key, ...hidden]}
      />
    </div>
  );
}

function Field({ name, schema, root, required, value, onChange }) {
  const resolved = resolve(schema, root);
  const label = (
    <span>
      {prettify(name)}
      {required && <abbr title="required"> *</abbr>}
    </span>
  );

  if (resolved.oneOf && resolved.discriminator) {
    return (
      <fieldset className="fieldset">
        <legend>{prettify(name)}</legend>
        <UnionFields schema={resolved} root={root} value={value || {}} onChange={onChange} />
      </fieldset>
    );
  }

  if (resolved.anyOf) {
    const options = resolved.anyOf.map((option) => resolve(option, root));
    const nonNull = options.filter((option) => option.type !== "null");
    if (nonNull.length === 1) {
      const inner = { ...nonNull[0], default: resolved.default };
      if (inner.type === "object" && inner.properties) {
        return <OptionalObject name={name} schema={inner} root={root} value={value} onChange={onChange} />;
      }
      return <Field name={name} schema={inner} root={root} required={required} value={value} onChange={onChange} />;
    }
    return <LooseField label={label} defaultValue={resolved.default} value={value} onChange={onChange} />;
  }

  if (resolved.enum || resolved.const !== undefined) {
    const choices = resolved.enum || [resolved.const];
    return (
      <label className="field">
        {label}
        <select
          value={value ?? ""}
          onChange={(event) => {
            const raw = event.target.value;
            if (raw === "") return onChange(undefined);
            const match = choices.find((choice) => String(choice) === raw);
            onChange(match);
          }}
        >
          <option value="">{resolved.default !== undefined ? `default (${resolved.default})` : "—"}</option>
          {choices.map((choice) => (
            <option key={String(choice)} value={String(choice)}>
              {String(choice)}
            </option>
          ))}
        </select>
      </label>
    );
  }

  switch (resolved.type) {
    case "boolean":
      return (
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={value ?? resolved.default ?? false}
            onChange={(event) => onChange(event.target.checked)}
          />
          {label}
        </label>
      );
    case "integer":
    case "number":
      return (
        <label className="field">
          {label}
          <input
            type="number"
            step={resolved.type === "integer" ? 1 : "any"}
            min={resolved.minimum ?? resolved.exclusiveMinimum}
            max={resolved.maximum ?? resolved.exclusiveMaximum}
            placeholder={resolved.default != null ? String(resolved.default) : ""}
            value={value ?? ""}
            onChange={(event) => {
              const raw = event.target.value;
              onChange(raw === "" ? undefined : Number(raw));
            }}
          />
        </label>
      );
    case "string":
      return (
        <label className="field">
          {label}
          <input
            placeholder={resolved.default != null ? String(resolved.default) : ""}
            value={value ?? ""}
            onChange={(event) => onChange(event.target.value === "" ? undefined : event.target.value)}
          />
        </label>
      );
    case "array":
      if (resolved.prefixItems || ["string", "number", "integer"].includes(resolve(resolved.items, root).type)) {
        const numeric = resolved.prefixItems
          ? resolved.prefixItems.every((item) => ["number", "integer"].includes(item.type))
          : resolve(resolved.items, root).type !== "string";
        return (
          <ListField
            label={label}
            numeric={numeric}
            defaultValue={resolved.default}
            value={value}
            onChange={onChange}
          />
        );
      }
      return <JsonField label={label} value={value} defaultValue={resolved.default} onChange={onChange} />;
    case "object":
      if (resolved.properties) {
        return (
          <fieldset className="fieldset">
            <legend>{prettify(name)}</legend>
            <ObjectFields schema={resolved} root={root} value={value || {}} onChange={onChange} />
          </fieldset>
        );
      }
      return <JsonField label={label} value={value} defaultValue={resolved.default} onChange={onChange} />;
    default:
      return <JsonField label={label} value={value} defaultValue={resolved.default} onChange={onChange} />;
  }
}

function OptionalObject({ name, schema, root, value, onChange }) {
  const enabled = value !== undefined && value !== null;
  return (
    <fieldset className="fieldset">
      <legend>
        <label className="checkbox">
          <input type="checkbox" checked={enabled} onChange={(event) => onChange(event.target.checked ? {} : undefined)} />
          {prettify(name)}
        </label>
      </legend>
      {enabled && <ObjectFields schema={schema} root={root} value={value} onChange={onChange} />}
    </fieldset>
  );
}

/** Comma-separated values: "a, b" -> ["a", "b"] or [1, 2]. */
function ListField({ label, numeric, defaultValue, value, onChange }) {
  const [text, setText] = useState(Array.isArray(value) ? value.join(", ") : "");
  return (
    <label className="field">
      {label}
      <input
        value={text}
        placeholder={Array.isArray(defaultValue) ? defaultValue.join(", ") : "comma-separated"}
        onChange={(event) => {
          setText(event.target.value);
          const parts = event.target.value.split(",").map((part) => part.trim()).filter(Boolean);
          if (parts.length === 0) return onChange(undefined);
          onChange(numeric ? parts.map(Number) : parts);
        }}
      />
    </label>
  );
}

/** Values that may be a number, a pair or a word (e.g. kernel_size, padding "auto"). */
function LooseField({ label, defaultValue, value, onChange }) {
  const show = Array.isArray(value) ? value.join(", ") : value ?? "";
  return (
    <label className="field">
      {label}
      <input
        value={show}
        placeholder={defaultValue != null ? String(defaultValue) : ""}
        onChange={(event) => {
          const raw = event.target.value.trim();
          if (raw === "") return onChange(undefined);
          const parts = raw.split(",").map((part) => part.trim());
          if (parts.length > 1 && parts.every((part) => part !== "" && !Number.isNaN(Number(part)))) {
            return onChange(parts.map(Number));
          }
          onChange(!Number.isNaN(Number(raw)) ? Number(raw) : raw);
        }}
      />
    </label>
  );
}

export function JsonField({ label, value, defaultValue, onChange }) {
  const [text, setText] = useState(value === undefined ? "" : JSON.stringify(value, null, 2));
  const [invalid, setInvalid] = useState(false);
  return (
    <label className="field">
      {label}
      <textarea
        rows={3}
        className={invalid ? "invalid" : undefined}
        placeholder={defaultValue !== undefined ? JSON.stringify(defaultValue) : "JSON"}
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          if (event.target.value.trim() === "") {
            setInvalid(false);
            return onChange(undefined);
          }
          try {
            onChange(JSON.parse(event.target.value));
            setInvalid(false);
          } catch {
            setInvalid(true);
          }
        }}
      />
      {invalid && <small className="error">not valid JSON yet</small>}
    </label>
  );
}
