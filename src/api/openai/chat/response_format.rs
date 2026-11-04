//! `response_format` on `/v1/chat/completions` (A5): `json_object` and
//! `json_schema`, honoured by prompt and check, not by constrained decoding.
//!
//! The model is told to answer with JSON only (and given the schema); the
//! reply is parsed, checked against the schema, and regenerated up to
//! [`ATTEMPTS`] times if it does not comply, then refused with
//! [`FormatUnmet`] (a 502). A compliant reply is returned as compact JSON
//! text with any prose or code fence around it removed.

use super::{ChatCompletionRequest, ChatCompletionResponse};
use crate::api::AppState;
use serde_json::Value;

/// How many times a reply that is not the requested JSON is regenerated.
pub(super) const ATTEMPTS: usize = 3;

/// The JSON a request asked for.
#[derive(Debug, Clone)]
pub(super) enum Format {
    /// Any JSON object.
    Object,
    /// JSON that validates against `schema`.
    Schema { name: String, schema: Value },
}

/// A reply that was not the requested JSON after [`ATTEMPTS`] tries.
#[derive(Debug)]
pub(crate) struct FormatUnmet(pub(super) String);

impl std::fmt::Display for FormatUnmet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FormatUnmet {}

/// Read and validate a request's `response_format`. `Ok(None)` means plain
/// text (absent, or `{"type": "text"}`); `Err` is the message for a 400.
pub(super) fn parse(raw: Option<&Value>) -> Result<Option<Format>, String> {
    let Some(raw) = raw else { return Ok(None) };
    match raw.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(Format::Object)),
        Some("json_schema") => parse_schema(raw).map(Some),
        _ => Err(format!(
            "response_format {raw} is not supported; use {{\"type\": \"text\"}}, \
             {{\"type\": \"json_object\"}} or {{\"type\": \"json_schema\", \"json_schema\": \
             {{\"name\": ..., \"schema\": {{...}}}}}}"
        )),
    }
}

fn parse_schema(raw: &Value) -> Result<Format, String> {
    let spec = raw
        .get("json_schema")
        .ok_or("response_format json_schema needs a `json_schema` object")?;
    let schema = spec
        .get("schema")
        .filter(|s| s.is_object() || s.is_boolean())
        .ok_or("response_format json_schema needs a `schema` object")?;
    jsonschema::validator_for(schema)
        .map_err(|e| format!("response_format json_schema.schema is not a valid schema: {e}"))?;
    let name = spec
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("response")
        .to_string();
    Ok(Format::Schema {
        name,
        schema: schema.clone(),
    })
}

/// The system-prompt text that asks for `format`.
pub(super) fn instruction(format: &Format) -> String {
    let mut s = String::from(
        "Respond with a single JSON value and nothing else: no prose before or \
         after it and no Markdown code fence.",
    );
    match format {
        Format::Object => s.push_str(" The value must be a JSON object."),
        Format::Schema { name, schema } => {
            s.push_str(&format!(
                " The value must conform to this JSON Schema (\"{name}\"):\n{schema}"
            ));
        }
    }
    s
}

/// The compact JSON text of `reply` if it is what `format` asks for.
pub(super) fn check(format: &Format, reply: &str) -> Option<String> {
    let value = crate::contracts::validation::parse_json(reply)?;
    let ok = match format {
        Format::Object => value.is_object(),
        Format::Schema { schema, .. } => jsonschema::is_valid(schema, &value),
    };
    ok.then(|| value.to_string())
}

/// The format a request asks for, if any and if valid (see [`validate_request`]).
pub(super) fn active(request: &ChatCompletionRequest) -> Option<Format> {
    parse(request.response_format.as_ref()).ok().flatten()
}

/// The 400 message for a request whose `response_format` cannot be served.
pub(super) fn validate_request(request: &ChatCompletionRequest) -> Result<(), String> {
    if parse(request.response_format.as_ref())?.is_none() {
        return Ok(());
    }
    if super::request_wants_tools(request) {
        return Err("`response_format` cannot be combined with `tools`".to_string());
    }
    if request
        .lightbulb
        .as_ref()
        .is_some_and(|lb| lb.output_contract.is_some())
    {
        return Err(
            "`response_format` cannot be combined with `lightbulb.output_contract`".to_string(),
        );
    }
    Ok(())
}

/// `create_chat_completion` for a request with a `response_format`: ask for
/// the JSON, check the reply, regenerate up to [`ATTEMPTS`] times, else
/// [`FormatUnmet`]. At `temperature` 0 a regeneration repeats the same reply,
/// so the retries only help sampled requests.
pub(super) async fn run(
    state: AppState,
    mut request: ChatCompletionRequest,
    priority: crate::engine::Priority,
    format: Format,
) -> anyhow::Result<ChatCompletionResponse> {
    request.messages = super::inject_system_text(&request.messages, instruction(&format));
    for _ in 0..ATTEMPTS {
        let mut resp =
            super::create_chat_completion(state.clone(), request.clone(), priority).await?;
        if let Some(json) = resp
            .choices
            .first()
            .and_then(|c| check(&format, &c.message.content))
        {
            resp.choices[0].message.content = json;
            return Ok(resp);
        }
    }
    Err(FormatUnmet(format!(
        "the model did not produce JSON matching response_format in {ATTEMPTS} attempts"
    ))
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema_format() -> Format {
        parse(Some(&json!({"type": "json_schema", "json_schema": {
            "name": "p",
            "schema": {"type": "object", "required": ["n"],
                       "properties": {"n": {"type": "integer"}}}
        }})))
        .unwrap()
        .unwrap()
    }

    #[test]
    fn text_and_absent_mean_plain_text() {
        assert!(parse(None).unwrap().is_none());
        assert!(parse(Some(&json!({"type": "text"}))).unwrap().is_none());
    }

    #[test]
    fn unknown_types_and_bad_schemas_are_refused() {
        assert!(parse(Some(&json!({"type": "xml"}))).is_err());
        assert!(parse(Some(&json!({"type": "json_schema"}))).is_err());
        let bad = json!({"type": "json_schema", "json_schema": {"schema": {"type": 7}}});
        assert!(parse(Some(&bad)).is_err());
    }

    #[test]
    fn json_object_accepts_only_objects() {
        let f = Format::Object;
        assert_eq!(check(&f, "{\"a\": 1}").as_deref(), Some("{\"a\":1}"));
        assert_eq!(
            check(&f, "```json\n{\"a\": 1}\n```").as_deref(),
            Some("{\"a\":1}")
        );
        assert!(check(&f, "[1, 2]").is_none());
        assert!(check(&f, "not json").is_none());
    }

    #[test]
    fn json_schema_checks_the_schema_not_just_the_syntax() {
        let f = schema_format();
        assert_eq!(check(&f, "{\"n\": 3}").as_deref(), Some("{\"n\":3}"));
        assert!(check(&f, "{\"n\": \"three\"}").is_none());
        assert!(check(&f, "{}").is_none());
    }

    #[test]
    fn the_instruction_carries_the_schema() {
        assert!(instruction(&schema_format()).contains("\"required\""));
        assert!(instruction(&Format::Object).contains("JSON object"));
    }
}
