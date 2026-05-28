use crate::common::ChatCompletionsRequest;
use crate::common::ChatFunctionCall;
use crate::common::ChatMessage;
use crate::common::ChatToolCall;
use crate::common::ChatToolDefinition;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use serde_json::Value;

/// Converts a list of `ResponseItem`s (Responses API format) into a list of
/// `ChatMessage`s (Chat Completions API format).
///
/// The conversion handles:
/// - `Message` → `ChatMessage` with role and text content
/// - `FunctionCall` / `CustomToolCall` → assistant `ChatMessage` with `tool_calls`
/// - `FunctionCallOutput` / `CustomToolCallOutput` → `tool` role `ChatMessage`
/// - `LocalShellCall` → assistant `ChatMessage` with tool call
/// - `Reasoning` → skipped (Chat API does not support reasoning)
/// - Other types → skipped
pub fn response_items_to_chat_messages(items: &[ResponseItem]) -> Vec<ChatMessage> {
    let mut messages: Vec<ChatMessage> = Vec::new();

    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let text = content_items_to_text(content);
                // Normalize role: "developer" (OpenAI Responses API) → "system" (Chat Completions)
                // Third-party providers typically don't support "developer" role.
                let normalized_role = match role.as_str() {
                    "developer" => "system",
                    other => other,
                };
                // Merge consecutive system messages into one (some providers like MiniMax
                // don't support multiple system messages).
                if normalized_role == "system"
                    && let Some(last) = messages.last_mut()
                    && last.role == "system"
                {
                    last.content = Some(format!(
                        "{}\n\n{}",
                        last.content.as_deref().unwrap_or(""),
                        text
                    ));
                } else {
                    messages.push(ChatMessage {
                        role: normalized_role.to_string(),
                        content: Some(text),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                // Merge into the last assistant message or create a new one.
                let tool_call = ChatToolCall {
                    id: call_id.clone(),
                    call_type: "function".to_string(),
                    function: ChatFunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                };
                if let Some(last) = messages.last_mut()
                    && last.role == "assistant"
                {
                    last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                } else {
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: None,
                        tool_calls: Some(vec![tool_call]),
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
            ResponseItem::CustomToolCall {
                name,
                input,
                call_id,
                ..
            } => {
                let tool_call = ChatToolCall {
                    id: call_id.clone(),
                    call_type: "function".to_string(),
                    function: ChatFunctionCall {
                        name: name.clone(),
                        arguments: input.clone(),
                    },
                };
                if let Some(last) = messages.last_mut()
                    && last.role == "assistant"
                {
                    last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                } else {
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: None,
                        tool_calls: Some(vec![tool_call]),
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
            ResponseItem::LocalShellCall {
                call_id, action, ..
            } => {
                let call_id = call_id
                    .clone()
                    .unwrap_or_else(|| format!("local_shell_{}", messages.len()));
                let arguments = serde_json::to_string(action).unwrap_or_default();
                let tool_call = ChatToolCall {
                    id: call_id,
                    call_type: "function".to_string(),
                    function: ChatFunctionCall {
                        name: "shell".to_string(),
                        arguments,
                    },
                };
                if let Some(last) = messages.last_mut()
                    && last.role == "assistant"
                {
                    last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                } else {
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: None,
                        tool_calls: Some(vec![tool_call]),
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                let content = output.body.to_text().unwrap_or_default();
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                    name: None,
                });
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let content = output.body.to_text().unwrap_or_default();
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                    name: None,
                });
            }
            // Reasoning, WebSearchCall, ToolSearchCall, ToolSearchOutput,
            // ImageGenerationCall, Compaction, CompactionTrigger,
            // ContextCompaction, Other are not representable in the Chat
            // Completions format — skip them.
            ResponseItem::Reasoning { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {}
        }
    }

    messages
}

/// Converts a `system` instruction string and a list of `ResponseItem`s into a
/// `ChatCompletionsRequest`.
pub fn build_chat_completions_request(
    model: &str,
    instructions: &str,
    input: &[ResponseItem],
    tools: Vec<ChatToolDefinition>,
    stream: bool,
) -> ChatCompletionsRequest {
    let mut messages = Vec::new();

    // Convert input items first.
    let input_messages = response_items_to_chat_messages(input);

    // Merge instructions with the first system message from input, or add as a new one.
    // Some providers (like MiniMax) don't support multiple system messages.
    let mut instructions_added = false;
    if !instructions.is_empty() {
        if let Some(first_system) = input_messages.iter().find(|m| m.role == "system") {
            // Merge instructions with the first system message.
            let merged_content = format!(
                "{}\n\n{}",
                instructions,
                first_system.content.as_deref().unwrap_or("")
            );
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(merged_content),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
            instructions_added = true;
        } else {
            // No system message in input, add instructions as system message.
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(instructions.to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
            instructions_added = true;
        }
    }

    // Add remaining messages from input, skipping the first system message if we merged it.
    let mut skipped_system = false;
    for msg in input_messages {
        if instructions_added && !skipped_system && msg.role == "system" {
            skipped_system = true;
            continue;
        }
        messages.push(msg);
    }

    let tools = if tools.is_empty() { None } else { Some(tools) };

    // Only set tool_choice when tools are present.
    let tool_choice = if tools.is_some() {
        Some("auto".to_string())
    } else {
        None
    };

    ChatCompletionsRequest {
        model: model.to_string(),
        messages,
        tools,
        tool_choice,
        stream,
        temperature: None,
    }
}

fn content_items_to_text(items: &[ContentItem]) -> String {
    items
        .iter()
        .map(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => text.as_str(),
            ContentItem::InputImage { .. } => "[image]",
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Converts Responses API tool JSON values (as produced by
/// `create_tools_json_for_responses_api`) into `ChatToolDefinition`s.
///
/// Only `"function"` type tools are converted. Other types (namespace,
/// web_search, image_generation, custom, tool_search) are skipped since they
/// are not representable in the Chat Completions format.
pub fn responses_tool_json_to_chat_tools(tool_jsons: &[Value]) -> Vec<ChatToolDefinition> {
    let mut chat_tools = Vec::new();

    for json in tool_jsons {
        let tool_type = json.get("type").and_then(Value::as_str).unwrap_or("");
        match tool_type {
            "function" => {
                let name = json
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let description = json
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let parameters = json
                    .get("parameters")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                if !name.is_empty() {
                    chat_tools.push(ChatToolDefinition {
                        tool_type: "function".to_string(),
                        function: crate::common::ChatFunctionDefinition {
                            name,
                            description: if description.is_empty() {
                                None
                            } else {
                                Some(description)
                            },
                            parameters,
                        },
                    });
                }
            }
            "namespace" => {
                // Flatten namespace tools.
                if let Some(tools) = json.get("tools").and_then(Value::as_array) {
                    for ns_tool in tools {
                        let ns_type = ns_tool.get("type").and_then(Value::as_str).unwrap_or("");
                        if ns_type == "function" {
                            let name = ns_tool
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            let description = ns_tool
                                .get("description")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            let parameters = ns_tool
                                .get("parameters")
                                .cloned()
                                .unwrap_or(serde_json::json!({}));
                            if !name.is_empty() {
                                chat_tools.push(ChatToolDefinition {
                                    tool_type: "function".to_string(),
                                    function: crate::common::ChatFunctionDefinition {
                                        name,
                                        description: if description.is_empty() {
                                            None
                                        } else {
                                            Some(description)
                                        },
                                        parameters,
                                    },
                                });
                            }
                        }
                    }
                }
            }
            "custom" => {
                let name = json
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let description = json
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !name.is_empty() {
                    chat_tools.push(ChatToolDefinition {
                        tool_type: "function".to_string(),
                        function: crate::common::ChatFunctionDefinition {
                            name,
                            description: if description.is_empty() {
                                None
                            } else {
                                Some(description)
                            },
                            parameters: serde_json::json!({
                                "type": "object",
                                "properties": {}
                            }),
                        },
                    });
                }
            }
            // web_search, image_generation, tool_search — skip.
            _ => {}
        }
    }

    chat_tools
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::FunctionCallOutputPayload;

    #[test]
    fn test_message_conversion() {
        let items = vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Hello".to_string(),
            }],
            phase: None,
        }];
        let messages = response_items_to_chat_messages(&items);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_function_call_conversion() {
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "Run a command".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"command":"ls"}"#.to_string(),
                call_id: "call_123".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_123".to_string(),
                output: FunctionCallOutputPayload::from_text("file1.txt\nfile2.txt".to_string()),
            },
        ];
        let messages = response_items_to_chat_messages(&items);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].tool_calls.is_some());
        assert_eq!(messages[1].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(messages[1].tool_calls.as_ref().unwrap()[0].id, "call_123");
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_reasoning_skipped() {
        let items = vec![
            ResponseItem::Reasoning {
                id: "r1".to_string(),
                summary: vec![],
                content: None,
                encrypted_content: None,
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "Done".to_string(),
                }],
                phase: None,
            },
        ];
        let messages = response_items_to_chat_messages(&items);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "assistant");
    }
}
