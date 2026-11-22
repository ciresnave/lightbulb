// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-family tool-call parsers (ROADMAP A4).
//!
//! When a checkpoint's own chat template renders `tools` (`ChatTemplate::render_with_tools`), the model answers
//! in the format that template taught it, not in Lightbulb's own `<tool_call>{json}</tool_call>` tag. There is no
//! universal wire format, so each family's convention gets a small pure parser here:
//!
//! - Llama-3.1: `<|python_tag|>{"name": .., "parameters": {..}}` (several calls separated by `;`).
//! - Mistral: `[TOOL_CALLS] [{"name": .., "arguments": {..}}, ..]`.
//! - Qwen3.5 / Qwen3-Coder: `<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`.
//! - A bare JSON object (or array of objects) that is the WHOLE reply (Llama-3.1's JSON-only mode).
//!
//! Hermes / Qwen2.5 style `<tool_call>{json}</tool_call>` is parsed by `chat::extract_tool_calls`, which this
//! module does not duplicate.
//!
//! **A parser only claims a call whose name is one of the offered tools.** The bare-JSON form in particular
//! would otherwise turn any JSON a model writes in prose into a tool call.

use super::chat::Tool;
use serde_json::{Map, Value};

/// A call a parser found: the function name and its arguments as a JSON value (the caller serialises it).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParsedCall {
    pub name: String,
    pub arguments: Value,
}

const PYTHON_TAG: &str = "<|python_tag|>";
const MISTRAL_TAG: &str = "[TOOL_CALLS]";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// `tools` in the shape a chat template's `tools` variable expects (the OpenAI / Hugging Face convention).
pub(crate) fn tools_for_template(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|Tool::Function { function }| {
            let mut f = Map::new();
            f.insert("name".into(), Value::String(function.name.clone()));
            if let Some(d) = &function.description {
                f.insert("description".into(), Value::String(d.clone()));
            }
            if let Some(p) = &function.parameters {
                f.insert("parameters".into(), p.clone());
            }
            serde_json::json!({"type": "function", "function": Value::Object(f)})
        })
        .collect()
}

/// The first family format found in `text`: the prose before the call (trimmed) and the calls, or `None`.
pub(crate) fn parse_family_calls(text: &str, tools: &[Tool]) -> Option<(String, Vec<ParsedCall>)> {
    parse_python_tag(text, tools)
        .or_else(|| parse_mistral(text, tools))
        .or_else(|| parse_qwen_xml(text, tools))
        .or_else(|| parse_bare_json(text, tools))
}

fn is_offered(tools: &[Tool], name: &str) -> bool {
    tools
        .iter()
        .any(|Tool::Function { function }| function.name == name)
}

/// Arguments are an object normally, but Mistral and some fine-tunes write them as a JSON-encoded string.
fn normalise_arguments(v: Option<&Value>) -> Value {
    match v {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(other) => other.clone(),
        None => Value::Object(Map::new()),
    }
}

/// `{"name": .., "arguments"|"parameters": ..}` naming an offered tool.
fn call_from_value(v: &Value, tools: &[Tool]) -> Option<ParsedCall> {
    let name = v.get("name")?.as_str()?;
    if !is_offered(tools, name) {
        return None;
    }
    let args = v.get("arguments").or_else(|| v.get("parameters"));
    Some(ParsedCall {
        name: name.to_string(),
        arguments: normalise_arguments(args),
    })
}

/// Consecutive JSON values from `s`, separated by whitespace or `;`. Stops at the first thing that is not JSON.
fn json_values(s: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut rest = s.trim_start();
    while !rest.is_empty() {
        let mut stream = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let Some(Ok(v)) = stream.next() else { break };
        let used = stream.byte_offset();
        out.push(v);
        rest = rest[used..]
            .trim_start()
            .trim_start_matches(';')
            .trim_start();
    }
    out
}

/// Every value a tool call could be: an array contributes its elements, anything else itself.
fn flatten_calls(values: Vec<Value>) -> Vec<Value> {
    values
        .into_iter()
        .flat_map(|v| match v {
            Value::Array(items) => items,
            other => vec![other],
        })
        .collect()
}

fn calls_from_values(values: Vec<Value>, tools: &[Tool]) -> Vec<ParsedCall> {
    flatten_calls(values)
        .iter()
        .filter_map(|v| call_from_value(v, tools))
        .collect()
}

fn finish(leading: &str, calls: Vec<ParsedCall>) -> Option<(String, Vec<ParsedCall>)> {
    (!calls.is_empty()).then(|| (leading.trim().to_string(), calls))
}

fn parse_python_tag(text: &str, tools: &[Tool]) -> Option<(String, Vec<ParsedCall>)> {
    let at = text.find(PYTHON_TAG)?;
    let calls = calls_from_values(json_values(&text[at + PYTHON_TAG.len()..]), tools);
    finish(&text[..at], calls)
}

fn parse_mistral(text: &str, tools: &[Tool]) -> Option<(String, Vec<ParsedCall>)> {
    let at = text.find(MISTRAL_TAG)?;
    let calls = calls_from_values(json_values(&text[at + MISTRAL_TAG.len()..]), tools);
    finish(&text[..at], calls)
}

/// The whole reply is the call(s): nothing before or after the JSON.
fn parse_bare_json(text: &str, tools: &[Tool]) -> Option<(String, Vec<ParsedCall>)> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    let items = flatten_calls(vec![value]);
    let calls: Vec<ParsedCall> = items
        .iter()
        .filter_map(|v| call_from_value(v, tools))
        .collect();
    // Every element must be a call: a JSON array of other things is data, not a tool call.
    (calls.len() == items.len()).then_some(())?;
    finish("", calls)
}

/// The JSON-schema `type` of `tool.parameters.properties[key]`, if the schema says.
fn schema_type<'a>(tools: &'a [Tool], tool: &str, key: &str) -> Option<&'a str> {
    tools
        .iter()
        .map(|Tool::Function { function }| function)
        .find(|f| f.name == tool)?
        .parameters
        .as_ref()?
        .get("properties")?
        .get(key)?
        .get("type")?
        .as_str()
}

/// An XML parameter value: kept as a string unless the schema says it is some other JSON type.
fn coerce_parameter(raw: &str, ty: Option<&str>) -> Value {
    let text = raw.strip_prefix('\n').unwrap_or(raw);
    let text = text.strip_suffix('\n').unwrap_or(text);
    match ty {
        Some("integer" | "number" | "boolean" | "object" | "array" | "null") => {
            serde_json::from_str(text.trim()).unwrap_or_else(|_| Value::String(text.to_string()))
        }
        _ => Value::String(text.to_string()),
    }
}

/// `<function=NAME>` then any number of `<parameter=KEY>value</parameter>`.
fn parse_function_block(body: &str, tools: &[Tool]) -> Option<ParsedCall> {
    let after = body.trim().strip_prefix("<function=")?;
    let (name, mut rest) = after.split_once('>')?;
    if !is_offered(tools, name) {
        return None;
    }
    let mut args = Map::new();
    while let Some(p) = rest.find("<parameter=") {
        let after_key = &rest[p + "<parameter=".len()..];
        let (key, value_and_rest) = after_key.split_once('>')?;
        let (value, next) = value_and_rest.split_once("</parameter>")?;
        let ty = schema_type(tools, name, key);
        args.insert(key.to_string(), coerce_parameter(value, ty));
        rest = next;
    }
    Some(ParsedCall {
        name: name.to_string(),
        arguments: Value::Object(args),
    })
}

fn parse_qwen_xml(text: &str, tools: &[Tool]) -> Option<(String, Vec<ParsedCall>)> {
    let first = text.find(CALL_OPEN)?;
    let mut calls = Vec::new();
    let mut rest = &text[first..];
    while let Some(start) = rest.find(CALL_OPEN) {
        let after = &rest[start + CALL_OPEN.len()..];
        // An unterminated final block is cut off mid-call; the template says to end with the closing tag.
        let Some(end) = after.find(CALL_CLOSE) else {
            break;
        };
        calls.extend(parse_function_block(&after[..end], tools));
        rest = &after[end + CALL_CLOSE.len()..];
    }
    finish(&text[..first], calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Tool> {
        serde_json::from_value(serde_json::json!([
            {"type": "function", "function": {"name": "get_weather", "description": "Weather",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "days": {"type": "integer"}}}}},
            {"type": "function", "function": {"name": "get_time"}}
        ]))
        .unwrap()
    }

    fn call(name: &str, args: Value) -> ParsedCall {
        ParsedCall {
            name: name.into(),
            arguments: args,
        }
    }

    #[test]
    fn python_tag_parses_one_and_several_calls() {
        let t = tools();
        let one = "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eom_id|>";
        let (lead, calls) = parse_family_calls(one, &t).unwrap();
        assert_eq!(lead, "");
        assert_eq!(
            calls,
            [call("get_weather", serde_json::json!({"city": "Paris"}))]
        );
        let two = "Checking. <|python_tag|>{\"name\": \"get_time\", \"parameters\": {}}; {\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}";
        let (lead, calls) = parse_family_calls(two, &t).unwrap();
        assert_eq!(lead, "Checking.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "get_weather");
    }

    #[test]
    fn mistral_parses_an_array_and_string_arguments() {
        let t = tools();
        let text = "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}, \"id\": \"abc123def\"}, {\"name\": \"get_time\", \"arguments\": \"{}\"}]";
        let (_, calls) = parse_family_calls(text, &t).unwrap();
        assert_eq!(
            calls,
            [
                call("get_weather", serde_json::json!({"city": "Oslo"})),
                call("get_time", serde_json::json!({})),
            ]
        );
    }

    #[test]
    fn qwen_xml_uses_the_schema_to_type_parameters() {
        let t = tools();
        let text = "Let me look.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let (lead, calls) = parse_family_calls(text, &t).unwrap();
        assert_eq!(lead, "Let me look.");
        // `city` is a string and stays one; `days` is an integer by the schema.
        assert_eq!(
            calls,
            [call(
                "get_weather",
                serde_json::json!({"city": "Paris", "days": 3})
            )]
        );
        // A numeric-looking value for a string parameter must stay a string.
        let numeric = "<tool_call>\n<function=get_weather>\n<parameter=city>\n12345\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_family_calls(numeric, &t).unwrap();
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "12345"}));
    }

    #[test]
    fn bare_json_is_a_call_only_when_it_is_the_whole_reply_and_names_an_offered_tool() {
        let t = tools();
        let ok = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}";
        assert_eq!(parse_family_calls(ok, &t).unwrap().1.len(), 1);
        // Prose around it: not a call.
        assert!(parse_family_calls(&format!("Here you go: {ok}"), &t).is_none());
        // Not an offered tool: not a call.
        assert!(parse_family_calls("{\"name\": \"rm_rf\", \"parameters\": {}}", &t).is_none());
        // JSON data that merely has a `name` key.
        assert!(parse_family_calls("{\"name\": \"Paris\", \"population\": 2}", &t).is_none());
        // An array mixing a call with something else is data.
        let mixed = "[{\"name\": \"get_time\"}, 5]";
        assert!(parse_family_calls(mixed, &t).is_none());
    }

    #[test]
    fn ordinary_text_and_unknown_tools_yield_nothing() {
        let t = tools();
        assert!(parse_family_calls("The capital of France is Paris.", &t).is_none());
        assert!(parse_family_calls("<|python_tag|>brave_search.call(query=\"x\")", &t).is_none());
        let unknown = "<|python_tag|>{\"name\": \"nope\", \"parameters\": {}}";
        assert!(parse_family_calls(unknown, &t).is_none());
        // Unterminated XML block is not a call.
        let cut = "<tool_call>\n<function=get_time>\n";
        assert!(parse_family_calls(cut, &t).is_none());
    }

    #[test]
    fn tools_for_template_has_the_hf_shape() {
        let v = tools_for_template(&tools());
        assert_eq!(v[0]["type"], "function");
        assert_eq!(v[0]["function"]["name"], "get_weather");
        assert_eq!(v[0]["function"]["description"], "Weather");
        assert!(v[0]["function"]["parameters"]["properties"]["city"].is_object());
        assert!(v[1]["function"].get("description").is_none());
    }
}
