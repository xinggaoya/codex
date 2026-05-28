use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

const OPENAI_MODEL_HEADER: &str = "openai-model";
const REQUEST_ID_HEADER: &str = "x-request-id";

pub fn spawn_chat_completions_response_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    turn_state: Option<Arc<OnceLock<String>>>,
) -> ResponseStream {
    let server_model = stream_response
        .headers
        .get(OPENAI_MODEL_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    if let Some(turn_state) = turn_state.as_ref()
        && let Some(header_value) = stream_response
            .headers
            .get("x-codex-turn-state")
            .and_then(|v| v.to_str().ok())
    {
        let _ = turn_state.set(header_value.to_string());
    }
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        if let Some(model) = server_model {
            let _ = tx_event.send(Ok(ResponseEvent::ServerModel(model))).await;
        }
        process_chat_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    model: Option<String>,
    choices: Option<Vec<ChatChoice>>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    index: Option<u32>,
    delta: Option<ChatDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatDelta {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<ChatDeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChatDeltaToolCall {
    index: u32,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: Option<ChatDeltaFunction>,
}

#[derive(Debug, Deserialize)]
struct ChatDeltaFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    total_tokens: Option<i64>,
}

/// Tracks incremental state for tool calls across streamed chunks.
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
}

pub async fn process_chat_sse(
    bytes: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    _telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = bytes.eventsource();
    let mut tool_call_states: HashMap<u32, ToolCallState> = HashMap::new();
    let mut text_started = false;
    let mut full_text = String::new();

    loop {
        match timeout(idle_timeout, stream.next()).await {
            Ok(Some(Ok(event))) => {
                let data = event.data.trim();
                if data == "[DONE]" {
                    break;
                }

                let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        debug!("failed to parse chat completion chunk: {e}");
                        trace!("raw chunk: {data}");
                        continue;
                    }
                };

                // Process usage if present in the final chunk.
                let token_usage = chunk.usage.map(|u| TokenUsage {
                    input_tokens: u.prompt_tokens.unwrap_or(0),
                    cached_input_tokens: 0,
                    output_tokens: u.completion_tokens.unwrap_or(0),
                    reasoning_output_tokens: 0,
                    total_tokens: u.total_tokens.unwrap_or(0),
                });

                if let Some(choices) = chunk.choices {
                    for choice in choices {
                        if let Some(delta) = choice.delta {
                            // Text content delta.
                            if let Some(content) = delta.content
                                && !content.is_empty()
                            {
                                // Emit OutputItemAdded on first text chunk.
                                if !text_started {
                                    text_started = true;
                                    let item = ResponseItem::Message {
                                        id: None,
                                        role: "assistant".to_string(),
                                        content: vec![],
                                        phase: None,
                                    };
                                    let _ = tx_event
                                        .send(Ok(ResponseEvent::OutputItemAdded(item)))
                                        .await;
                                }
                                full_text.push_str(&content);
                                let _ = tx_event
                                    .send(Ok(ResponseEvent::OutputTextDelta(content)))
                                    .await;
                            }

                            // Tool calls (incremental).
                            if let Some(tool_calls) = delta.tool_calls {
                                for tc in tool_calls {
                                    let entry =
                                        tool_call_states.entry(tc.index).or_insert_with(|| {
                                            ToolCallState {
                                                id: tc
                                                    .id
                                                    .clone()
                                                    .unwrap_or_else(|| "pending".to_string()),
                                                name: String::new(),
                                                arguments: String::new(),
                                            }
                                        });
                                    if let Some(id) = tc.id {
                                        entry.id = id;
                                    }
                                    if let Some(func) = tc.function {
                                        if let Some(name) = func.name {
                                            entry.name = name;
                                        }
                                        if let Some(arguments) = func.arguments {
                                            entry.arguments.push_str(&arguments);
                                        }
                                    }
                                }
                            }
                        }

                        // Check for finish reason.
                        if let Some(finish_reason) = choice.finish_reason.as_deref() {
                            // Emit completed text message if we started one.
                            if text_started {
                                let item = ResponseItem::Message {
                                    id: None,
                                    role: "assistant".to_string(),
                                    content: vec![
                                        codex_protocol::models::ContentItem::OutputText {
                                            text: full_text.clone(),
                                        },
                                    ],
                                    phase: None,
                                };
                                let _ = tx_event
                                    .send(Ok(ResponseEvent::OutputItemDone(item)))
                                    .await;
                            }

                            // Emit completed tool calls.
                            for (_index, state) in tool_call_states.drain() {
                                let item = ResponseItem::FunctionCall {
                                    id: None,
                                    name: state.name,
                                    namespace: None,
                                    arguments: state.arguments,
                                    call_id: state.id,
                                };
                                let _ = tx_event
                                    .send(Ok(ResponseEvent::OutputItemDone(item)))
                                    .await;
                            }

                            let end_turn = finish_reason == "stop" || finish_reason == "length";
                            let _ = tx_event
                                .send(Ok(ResponseEvent::Completed {
                                    response_id: String::new(),
                                    token_usage: token_usage.clone(),
                                    end_turn: Some(end_turn),
                                }))
                                .await;
                        }
                    }
                }
            }
            Ok(Some(Err(e))) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!("SSE parse error: {e}"))))
                    .await;
                break;
            }
            Ok(None) => {
                // Stream ended without [DONE].
                // Flush any remaining tool calls.
                for (_index, state) in tool_call_states.drain() {
                    let item = ResponseItem::FunctionCall {
                        id: None,
                        name: state.name,
                        namespace: None,
                        arguments: state.arguments,
                        call_id: state.id,
                    };
                    let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                }
                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id: String::new(),
                        token_usage: None,
                        end_turn: Some(true),
                    }))
                    .await;
                break;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!(
                        "stream idle timeout after {}s",
                        idle_timeout.as_secs()
                    ))))
                    .await;
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn make_chunk_json(
        content: Option<&str>,
        tool_calls: Option<Vec<serde_json::Value>>,
        finish_reason: Option<&str>,
        usage: Option<serde_json::Value>,
    ) -> String {
        let mut delta = serde_json::json!({});
        if let Some(c) = content {
            delta["content"] = serde_json::json!(c);
        }
        if let Some(tc) = tool_calls {
            delta["tool_calls"] = serde_json::json!(tc);
        }
        let mut choice = serde_json::json!({
            "index": 0,
            "delta": delta,
        });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = serde_json::json!(fr);
        }
        let mut chunk = serde_json::json!({
            "id": "chatcmpl-test",
            "choices": [choice],
        });
        if let Some(u) = usage {
            chunk["usage"] = u;
        }
        serde_json::to_string(&chunk).unwrap()
    }

    #[tokio::test]
    async fn test_simple_text_stream() {
        let events = vec![
            make_chunk_json(Some("Hello"), None, None, None),
            make_chunk_json(Some(" world"), None, None, None),
            make_chunk_json(None, None, Some("stop"), None),
        ];

        // Format as proper SSE events.
        let mut sse_data: String = events
            .into_iter()
            .map(|e| format!("data: {e}\n\n"))
            .collect();
        sse_data.push_str("data: [DONE]\n\n");

        let byte_stream = Box::pin(stream::once(async move {
            Ok::<_, codex_client::TransportError>(bytes::Bytes::from(sse_data))
        })) as ByteStream;

        let (tx, mut rx) = mpsc::channel(16);
        process_chat_sse(byte_stream, tx, Duration::from_secs(30), None).await;

        let mut results = Vec::new();
        while let Some(event) = rx.recv().await {
            results.push(event.unwrap());
        }

        assert!(matches!(&results[0], ResponseEvent::OutputTextDelta(s) if s == "Hello"));
        assert!(matches!(&results[1], ResponseEvent::OutputTextDelta(s) if s == " world"));
        assert!(matches!(
            &results[2],
            ResponseEvent::Completed {
                end_turn: Some(true),
                ..
            }
        ));
    }
}
