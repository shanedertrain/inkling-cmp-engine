// Ported from engine/server/src/api/openai.{h,cpp} (`parse_gen`) @ 4171493, which ports vLLM's
// entrypoints/openai/ request-to-SamplingParams code (`get_max_tokens`, protocol validators) (Apache-2.0,
// Copyright contributors to the vLLM project; config B overlay Hiutaky/vllm @ 6e9d1cce).
//! Sampling, stop and stream options common to both endpoints (`parse_gen` of the C++ port).

use ink_backend::SamplingParams;
use ink_text::StopParams;
use ink_text::pyjson::truthy;
use serde_json::Value;

use crate::config::ServerConfig;
use crate::error::ApiError;

#[derive(Debug)]
pub(crate) struct GenParams {
    pub sampling: SamplingParams,
    pub stop: StopParams,
    pub stream: bool,
    pub include_usage: bool,
    pub continuous_usage: bool,
    pub return_token_ids: bool,
    pub cache_salt: String,
    pub session: String,
}

/// The value of `key` unless it is absent or null.
pub(crate) fn present<'v>(body: &'v Value, key: &str) -> Option<&'v Value> {
    body.get(key).filter(|v| !v.is_null())
}

/// Python truthiness of a field; absent is false.
pub(crate) fn truthy_field(body: &Value, key: &str) -> bool {
    body.get(key).is_some_and(truthy)
}

fn number(body: &Value, key: &str, default: f64) -> Result<f64, ApiError> {
    match present(body, key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .ok_or_else(|| ApiError::bad_param(format!("Input should be a valid number: {key}"), key)),
    }
}

/// An integer field as the C++ port reads it (`get<int>()`): any JSON number, a float truncated.
pub(crate) fn int(value: &Value, key: &str) -> Result<i64, ApiError> {
    match value {
        Value::Number(n) => Ok(n.as_i64().unwrap_or_else(|| n.as_f64().map_or(0, |f| f as i64))),
        _ => Err(ApiError::bad_param(
            format!("Input should be a valid integer: {key}"),
            key,
        )),
    }
}

fn string(value: &Value, key: &str) -> Result<String, ApiError> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| ApiError::bad_param(format!("Input should be a valid string: {key}"), key))
}

fn string_list(value: &Value, key: &str) -> Result<Vec<String>, ApiError> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(items) => items.iter().map(|s| string(s, key)).collect(),
        _ => Err(ApiError::bad_param(
            format!("Input should be a valid string or list of strings: {key}"),
            key,
        )),
    }
}

/// Reads the generation options of a chat (`chat`) or completions request whose prompt has `prompt_len` tokens.
///
/// # Errors
/// HTTP 400 for a malformed field, a prompt over the context length, or `n` other than 1.
pub(crate) fn parse_gen(
    body: &Value,
    prompt_len: usize,
    cfg: &ServerConfig,
    chat: bool,
) -> Result<GenParams, ApiError> {
    let sampling = SamplingParams {
        temperature: number(body, "temperature", 1.0)?,
        top_p: number(body, "top_p", 1.0)?,
        min_p: number(body, "min_p", 0.0)?,
        // A negative top_k means "off" like 0 (float-to-integer casts saturate).
        top_k: number(body, "top_k", 0.0)? as u32,
        // an integer seed keeps all its digits (the f64 path is only for a seed written as a float)
        seed: match present(body, "seed") {
            Some(v) => Some(
                v.as_i64()
                    .or_else(|| v.as_f64().map(|f| f as i64))
                    .ok_or_else(|| ApiError::bad_param("Input should be a valid integer: seed", "seed"))?,
            ),
            None => None,
        },
        num_logprobs: None,
    };

    // get_max_tokens
    let req_max = match (
        chat,
        present(body, "max_completion_tokens"),
        present(body, "max_tokens"),
    ) {
        (true, Some(v), _) => Some(int(v, "max_completion_tokens")?),
        (_, _, Some(v)) => Some(int(v, "max_tokens")?),
        (false, _, None) => Some(16), // CompletionRequest.max_tokens default
        (true, None, None) => None,
    };
    let prompt_len = i64::try_from(prompt_len).unwrap_or(i64::MAX);
    let max_len = i64::from(cfg.max_model_len);
    if max_len < prompt_len {
        return Err(ApiError::bad_request(format!(
            "Input length ({prompt_len}) exceeds model's maximum context length ({max_len})."
        )));
    }
    let model_max = max_len - prompt_len;
    let max_tokens = req_max.map_or(model_max, |m| m.min(model_max));
    if let Some(m) = req_max
        && m.saturating_add(prompt_len) > max_len
    {
        return Err(ApiError::bad_param(
            format!(
                "This model's maximum context length is {max_len} tokens. However, you requested {} tokens \
                 ({prompt_len} in the messages, {m} in the completion). Please reduce the length of the messages \
                 or completion.",
                m.saturating_add(prompt_len)
            ),
            "max_tokens",
        ));
    }
    // vLLM's SamplingParams refuses it on both endpoints; the C++ port checked chat only, and a completions request
    // with `max_tokens` 0 or below then reached the back end with nothing to generate
    if max_tokens < 1 {
        return Err(ApiError::bad_request(format!(
            "max_tokens must be at least 1, got {max_tokens}"
        )));
    }
    // SamplingParams._verify_args
    let min_tokens = present(body, "min_tokens").map_or(Ok(0), |v| int(v, "min_tokens"))?;
    if min_tokens < 0 {
        return Err(ApiError::bad_request(format!(
            "min_tokens must be greater than or equal to 0, got {min_tokens}."
        )));
    }
    if min_tokens > max_tokens {
        return Err(ApiError::bad_request(format!(
            "min_tokens must be less than or equal to max_tokens={max_tokens}, got {min_tokens}."
        )));
    }

    let stop_strings = present(body, "stop").map_or(Ok(Vec::new()), |v| string_list(v, "stop"))?;
    let stop_token_ids = match present(body, "stop_token_ids") {
        None => Vec::new(),
        Some(Value::Array(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            for item in items {
                // an id outside u32 can never be generated
                if let Ok(id) = u32::try_from(int(item, "stop_token_ids")?) {
                    ids.push(id);
                }
            }
            ids
        }
        Some(_) => {
            return Err(ApiError::bad_param(
                "Input should be a valid list: stop_token_ids",
                "stop_token_ids",
            ));
        }
    };
    let stop = StopParams {
        stop: stop_strings,
        stop_token_ids,
        eos_token_id: Some(cfg.eos_token_id),
        ignore_eos: truthy_field(body, "ignore_eos"),
        include_stop_str_in_output: truthy_field(body, "include_stop_str_in_output"),
        min_tokens: usize::try_from(min_tokens).unwrap_or(usize::MAX),
        max_tokens: usize::try_from(max_tokens.max(0)).unwrap_or(usize::MAX),
        // chat: the reasoning parser forces skip_special_tokens off
        skip_special_tokens: !chat && present(body, "skip_special_tokens").is_none_or(truthy),
    };

    let stream = truthy_field(body, "stream");
    let (mut include_usage, mut continuous_usage) = (false, false);
    if let Some(options) = present(body, "stream_options") {
        if !stream {
            return Err(ApiError::bad_param(
                "Stream options can only be defined when `stream=True`.",
                "stream_options",
            ));
        }
        include_usage = truthy_field(options, "include_usage");
        continuous_usage = include_usage && truthy_field(options, "continuous_usage_stats");
    }
    if let Some(n) = present(body, "n")
        && int(n, "n")? != 1
    {
        return Err(ApiError::bad_param("n > 1 is not supported by this server", "n"));
    }
    Ok(GenParams {
        sampling,
        stop,
        stream,
        include_usage,
        continuous_usage,
        return_token_ids: truthy_field(body, "return_token_ids"),
        cache_salt: present(body, "cache_salt").map_or(Ok(String::new()), |v| string(v, "cache_salt"))?,
        session: present(body, "user")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    })
}
