/*
 * codehawk
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * codehawk is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * codehawk is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with codehawk.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

use dirs;
use log::{debug, info, trace, warn};
use reqwest::blocking::Client as ReqwestClient;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

/// Error type for when an operation is interrupted by the user (Ctrl-C)
#[derive(Debug)]
pub struct InterruptedError {
    pub message: String,
}

impl std::fmt::Display for InterruptedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for InterruptedError {}

impl InterruptedError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

/// Check for Ctrl-C signal and return InterruptedError if found
fn check_ctrl_c_signal(
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<(), Box<dyn Error>> {
    if let Some(rx) = ctrl_c_rx {
        if let Ok(receiver) = rx.lock() {
            if receiver.try_recv().is_ok() {
                debug!("Received Ctrl-C signal, exiting operation");
                return Err(
                    Box::new(InterruptedError::new("Operation interrupted by user"))
                        as Box<dyn Error>,
                );
            }
        }
    }
    Ok(())
}

pub struct Opts {
    pub max_tokens: Option<u32>,
    pub model: String,
    pub endpoint: String,
    pub tool_choice: Option<String>,
    pub api_key: Option<String>,
    pub max_retries: Option<usize>,
    pub retry_base_delay_secs: Option<u64>,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

pub type ToolCallback = fn(&String, &crate::ToolContext) -> Result<String, Box<dyn Error>>;
pub type ToolsCollection = HashMap<String, ToolItem>;
pub struct ToolItem {
    pub callback: ToolCallback,
    pub schema: String,
}

const MAX_TOKENS: u32 = 16384;
const OPEN_ROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPEN_ROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// Reads the API key from the specified file or the default OpenRouter key file.
fn read_api_key(api_key_file: Option<&String>) -> Result<String, Box<dyn Error>> {
    let key_path = if let Some(custom_path) = api_key_file {
        std::path::PathBuf::from(custom_path)
    } else {
        let home_dir = dirs::home_dir().ok_or("Could not find home directory")?;
        home_dir.join(".openrouter").join("key")
    };

    let api_key = std::fs::read_to_string(key_path)?;

    let api_key = api_key.trim().to_string();
    if api_key.is_empty() {
        return Err("API key file is empty".into());
    }

    Ok(api_key)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ParameterProperty {
    #[serde(rename = "type")]
    pub param_type: String,
    pub description: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub param_type: String,
    pub properties: HashMap<String, ParameterProperty>,
    pub required: Vec<String>,

    #[serde(rename = "additionalProperties")]
    pub additional_properties: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Debug)]
pub struct OpenAIRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIErrorMetadata {
    pub raw: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIError {
    pub code: Option<u64>,
    pub message: String,
    pub metadata: Option<OpenAIErrorMetadata>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIResponse {
    pub error: Option<OpenAIError>,
    pub choices: Option<Vec<Choice>>,
    pub usage: Option<Usage>,

    #[serde(skip_deserializing)]
    pub history: Vec<Message>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
    pub native_finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct StreamingChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct StreamingToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    pub function: StreamingFunctionCall,
}

#[derive(Deserialize, Debug)]
pub struct StreamingFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct Delta {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<StreamingToolCall>>,
}

#[derive(Deserialize, Debug)]
pub struct StreamingResponse {
    pub choices: Option<Vec<StreamingChoice>>,
    pub error: Option<OpenAIError>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub enum StatusUpdate {
    Thinking,
    ToolAccumulating {
        name: String,
        arguments: String,
    },
    ToolExecuting {
        name: String,
        arguments: String,
    },
    ToolComplete {
        name: String,
        arguments: String,
        duration_ms: u64,
    },
    StreamProcessing {
        bytes_read: usize,
        chunks_processed: u32,
    },
    Complete {
        usage: Option<Usage>,
    },
}

#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub status: StatusUpdate,
    pub elapsed_ms: u64,
}

pub enum ResponseMode {
    Complete,
    Streaming {
        stream_handler: Box<dyn Fn(&str) -> Result<(), Box<dyn Error>>>,
        progress_handler: Box<dyn Fn(&ProgressInfo) -> Result<(), Box<dyn Error>>>,
    },
}

/// Represents a single model entry.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub hugging_face_id: Option<String>,
    pub name: String,
    pub created: u64,
    pub description: String,
    pub context_length: u32,
    pub architecture: Architecture,
    pub pricing: Pricing,
    pub top_provider: TopProvider,
    pub supported_parameters: Vec<String>,
}

/// Represents the architecture details of a model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Architecture {
    pub modality: String,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub tokenizer: String,
    pub instruct_type: Option<String>,
}

/// Represents the pricing details for a model.
/// All monetary values are stored as strings as they appear in the JSON.
/// These could be parsed into a decimal type if needed.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Pricing {
    pub prompt: String,
    pub completion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cache_read: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cache_write: Option<String>,
}

/// Represents the top provider details for a model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopProvider {
    pub context_length: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub is_moderated: bool,
}

#[derive(Deserialize, Debug)]
struct ModelsApiResponse {
    data: Vec<ModelInfo>,
}

/// Fetches the list of available models from OpenRouter.
pub fn list_models() -> Result<Vec<ModelInfo>, Box<dyn Error>> {
    debug!("Fetching list of models from {}", OPEN_ROUTER_MODELS_URL);

    let client = ReqwestClient::builder().build()?;

    let response = client.get(OPEN_ROUTER_MODELS_URL).send()?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .unwrap_or_else(|e| format!("Failed to read error body: {}", e));
        warn!(
            "Failed to fetch models. Status: {}. Body: {}",
            status, error_text
        );
        return Err(format!("Failed to fetch models: {} - {}", status, error_text).into());
    }

    let models_api_response: ModelsApiResponse = response.json()?;

    Ok(models_api_response.data)
}

/// Perform a tool call and return the message to send back.
fn tool_call(
    tools_collection: &ToolsCollection,
    req: &ToolCall,
    ctx: &crate::ToolContext,
) -> Result<Message, Box<dyn Error>> {
    let tool_name = &req.function.name;

    // Validate tool call has complete data
    if tool_name.is_empty() {
        return Err("Tool call missing name".into());
    }
    if req.function.arguments.is_empty() {
        return Err(format!("Tool call '{}' missing arguments", tool_name).into());
    }
    if req.id.is_empty() {
        return Err(format!("Tool call '{}' missing ID", tool_name).into());
    }

    info!("Requesting tool {}", tool_name);
    let tool = tools_collection.get(tool_name);
    let content: String = match tool {
        None => {
            let error_msg = format!("error: invalid tool requested: '{}'", tool_name);
            warn!("{}", error_msg);
            error_msg
        }
        Some(t) => {
            info!("Executing tool {}", tool_name);
            debug!(
                "Passing arguments {:?} to tool {}",
                req.function.arguments, tool_name
            );
            match (t.callback)(&req.function.arguments, ctx) {
                Ok(result) => result,
                Err(e) => {
                    let error_msg = format!("error: tool '{}' failed: {}", tool_name, e);
                    warn!("{}", error_msg);
                    error_msg
                }
            }
        }
    };

    trace!("Tool {} gave output {:?}", tool_name, content);

    let msg = Message {
        role: "tool".to_string(),
        content: Some(content),
        tool_call_id: Some(req.id.clone()),
        name: Some(tool_name.clone()),
        tool_calls: None,
    };
    Ok(msg)
}

/// Create a Message from the specified role and content.
pub fn make_message(role: &str, content: String) -> Message {
    Message {
        role: role.to_string(),
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

/// Sends a POST request to the OpenAI API with the given messages and options.
pub fn post_request(
    messages: Vec<Message>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
    ctx: &crate::ToolContext,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    post_request_with_mode(
        messages,
        tools_collection,
        opts,
        ResponseMode::Complete,
        ctx,
        None,
    )
}

pub fn post_request_with_mode(
    messages: Vec<Message>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
    mode: ResponseMode,
    ctx: &crate::ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    post_request_with_mode_and_recursion(messages, tools_collection, opts, mode, ctx, ctrl_c_rx)
}

/// Internal function with iterative tool call handling.
fn post_request_with_mode_and_recursion(
    messages: Vec<Message>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
    mode: ResponseMode,
    ctx: &crate::ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let start_time = Instant::now();
    let mut messages = messages;

    loop {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if opts.endpoint == OPEN_ROUTER_URL
            || opts.endpoint == OPEN_ROUTER_MODELS_URL
            || opts.api_key.is_some()
        {
            let api_key = read_api_key(opts.api_key.as_ref())?;
            let bearer_auth = format!("Bearer {}", &api_key);
            headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_auth)?);
        }

        let mut tools: Vec<Tool> = vec![];
        for t in tools_collection.values() {
            let tool_schema = serde_json::from_str::<Tool>(&t.schema)?;
            tools.push(tool_schema);
        }

        let tool_choice = if tools.len() > 0 {
            // If tools are available, use user's choice or default to "auto"
            opts.tool_choice.clone()
        } else {
            None
        };

        // Use streaming based on the mode
        let use_streaming = match mode {
            ResponseMode::Streaming { .. } => true,
            ResponseMode::Complete => false,
        };

        let request_body = OpenAIRequest {
            model: opts.model.clone(),
            max_tokens: opts.max_tokens.unwrap_or_else(|| MAX_TOKENS),
            messages: messages.clone(),
            tools: if tools.len() > 0 { Some(tools) } else { None },
            tool_choice: tool_choice,
            stream: if use_streaming { Some(true) } else { None },
            parameters: opts.parameters.clone(),
        };

        let request_json = serde_json::to_string(&request_body)?;
        trace!("Send request: {}", request_json);
        debug!("Request has {} custom parameters", opts.parameters.len());

        let client = ReqwestClient::builder()
            .timeout(Duration::from_secs(1000))
            .build()?;

        let max_retries = opts.max_retries.unwrap_or(5);
        let base_delay_secs = opts.retry_base_delay_secs.unwrap_or(1);
        let mut response = None;

        for attempt in 1..=max_retries {
            let response_result = client
                .post(&opts.endpoint)
                .headers(headers.clone())
                .json(&request_body)
                .send();

            check_ctrl_c_signal(&ctrl_c_rx)?;

            match response_result {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        response = Some(resp);
                        break;
                    }

                    // Retry on server errors (5xx) including 503 Service Unavailable
                    if status.is_server_error() && attempt < max_retries {
                        let delay_duration =
                            Duration::from_secs(base_delay_secs * 2_u64.pow(attempt as u32 - 1));
                        warn!(
                            "Server error {} (attempt {}/{}). Retrying after {} seconds for endpoint: {}",
                            status,
                            attempt,
                            max_retries,
                            delay_duration.as_secs(),
                            opts.endpoint
                        );
                        // Check for interruption during sleep in smaller intervals
                        let sleep_interval = Duration::from_millis(100);
                        let mut remaining = delay_duration;
                        while remaining > Duration::ZERO {
                            check_ctrl_c_signal(&ctrl_c_rx)?;
                            let sleep_time = std::cmp::min(sleep_interval, remaining);
                            thread::sleep(sleep_time);
                            remaining = remaining.saturating_sub(sleep_time);
                        }
                        continue;
                    }

                    // Non-retryable error or max retries reached
                    let error_text = resp
                        .text()
                        .unwrap_or_else(|_| "Unable to read response".to_string());
                    return Err(format!("got error code: {}: {}", status, error_text).into());
                }
                Err(e) => {
                    if attempt < max_retries {
                        let delay_duration =
                            Duration::from_secs(base_delay_secs * 2_u64.pow(attempt as u32 - 1));
                        warn!(
                            "Request failed (attempt {}/{}): {}. Retrying after {} seconds for endpoint: {}",
                            attempt,
                            max_retries,
                            e,
                            delay_duration.as_secs(),
                            opts.endpoint
                        );
                        // Check for interruption during sleep in smaller intervals
                        let sleep_interval = Duration::from_millis(100);
                        let mut remaining = delay_duration;
                        while remaining > Duration::ZERO {
                            check_ctrl_c_signal(&ctrl_c_rx)?;
                            let sleep_time = std::cmp::min(sleep_interval, remaining);
                            thread::sleep(sleep_time);
                            remaining = remaining.saturating_sub(sleep_time);
                        }
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        let response = response.ok_or("Max retries reached without successful response")?;

        let mut openai_response: OpenAIResponse = if use_streaming {
            handle_streaming_response(response, &mode, ctrl_c_rx.clone())?
        } else {
            let response_text = response.text()?;
            trace!("Got response {:?}", response_text);
            serde_json::from_str(&response_text)?
        };

        if let Some(mut err) = openai_response.error {
            if err.metadata.is_some() {
                if let Some(raw) = err.metadata.unwrap().raw {
                    let raw_response: OpenAIResponse =
                        serde_json::from_str::<OpenAIResponse>(&raw)?;
                    if let Some(inner_err) = raw_response.error {
                        err = inner_err;
                    }
                }
            }
            return Err(format!(
                "got API error code: {}: {}",
                err.code.unwrap_or_else(|| 400),
                err.message
            )
            .into());
        }

        let mut finish: bool = true;

        if let Some(choices) = &openai_response.choices {
            trace!("Got {} choices", choices.len());
            if let Some(choice) = choices.first() {
                trace!("Got choice {:?}", choice);
                let finish_reason = choice
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| "".to_string());
                finish = finish_reason != ""
                    && (finish_reason != "tool_calls" || tools_collection.is_empty());
                if finish_reason == "error" {
                    let native_finish_reason = choice
                        .native_finish_reason
                        .clone()
                        .unwrap_or_else(|| finish_reason);
                    return Err(format!("got API error: {}", native_finish_reason).into());
                }

                // Add the complete assistant message (with both content and tool_calls if present)
                let assistant_msg = choice.message.clone();
                if assistant_msg.content.is_some() || assistant_msg.tool_calls.is_some() {
                    debug!(
                        "Adding assistant message to conversation history - content: {:?}, tool_calls: {}",
                        assistant_msg.content.as_ref().map(|c| c.len()),
                        assistant_msg
                            .tool_calls
                            .as_ref()
                            .map(|tc| tc.len())
                            .unwrap_or(0)
                    );
                    messages.push(assistant_msg);
                }

                if choice.message.tool_calls.is_some() {
                    let tool_calls = choice
                        .message
                        .tool_calls
                        .as_ref()
                        .ok_or("Invalid response")?;

                    for tool_call_request in tool_calls {
                        // Show progress for tool execution
                        if let ResponseMode::Streaming {
                            progress_handler, ..
                        } = &mode
                        {
                            let progress_info = ProgressInfo {
                                status: StatusUpdate::ToolExecuting {
                                    name: tool_call_request.function.name.clone(),
                                    arguments: tool_call_request.function.arguments.clone(),
                                },
                                elapsed_ms: start_time.elapsed().as_millis() as u64,
                            };
                            progress_handler(&progress_info)?;
                        }

                        debug!(
                            "Executing tool call '{}' with complete arguments (length: {})",
                            tool_call_request.function.name,
                            tool_call_request.function.arguments.len()
                        );
                        let tool_start_time = start_time.elapsed();
                        let msg = tool_call(&tools_collection, tool_call_request, ctx)?;
                        let tool_duration = start_time.elapsed() - tool_start_time;

                        // Show progress for tool execution completion
                        if let ResponseMode::Streaming {
                            progress_handler, ..
                        } = &mode
                        {
                            let progress_info = ProgressInfo {
                                status: StatusUpdate::ToolComplete {
                                    name: tool_call_request.function.name.clone(),
                                    arguments: tool_call_request.function.arguments.clone(),
                                    duration_ms: tool_duration.as_millis() as u64,
                                },
                                elapsed_ms: start_time.elapsed().as_millis() as u64,
                            };
                            progress_handler(&progress_info)?;
                        }

                        debug!(
                            "Adding tool response message for tool: {}",
                            tool_call_request.function.name
                        );
                        messages.push(msg);
                    }
                }
            }
        }

        if !finish {
            debug!(
                "Continuing conversation: finish={}, current_messages={}",
                finish,
                messages.len()
            );

            // Continue the loop to make another API request with the updated messages
            continue;
        }

        // If we reach here, the conversation is finished
        // Show final completion status with usage information
        if let ResponseMode::Streaming {
            progress_handler, ..
        } = &mode
        {
            let progress_info = ProgressInfo {
                status: StatusUpdate::Complete {
                    usage: openai_response.usage.clone(),
                },
                elapsed_ms: start_time.elapsed().as_millis() as u64,
            };
            progress_handler(&progress_info)?;
        }

        debug!("Final response: messages in history = {}", messages.len());
        openai_response.history = messages;
        return Ok(openai_response);
    }
}

/// Handle streaming response from the API
fn handle_streaming_response(
    response: reqwest::blocking::Response,
    mode: &ResponseMode,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let (stream_handler, progress_handler) = match mode {
        ResponseMode::Streaming {
            stream_handler,
            progress_handler,
        } => (stream_handler, progress_handler),
        ResponseMode::Complete => return Err("Invalid mode for streaming response".into()),
    };

    let reader = BufReader::new(response);
    let mut accumulated_content = String::new();
    let mut accumulated_tool_calls: HashMap<usize, ToolCall> = HashMap::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
    let mut in_tool_mode = false;
    let mut bytes_read = 0usize;
    let mut chunks_processed = 0u32;
    let mut tool_accumulation_start: Option<std::time::Instant> = None;
    let mut thinking_reported = false;
    let start_time = std::time::Instant::now();

    // Create a channel for line reading
    let (line_tx, line_rx) = mpsc::channel::<Result<String, std::io::Error>>();

    let _reader_thread = std::thread::spawn(move || {
        for line in reader.lines() {
            match line {
                Ok(line_content) => {
                    if line_tx.send(Ok(line_content)).is_err() {
                        break; // Receiver dropped, exit thread
                    }
                }
                Err(e) => {
                    let _ = line_tx.send(Err(e));
                    break;
                }
            }
        }
    });

    loop {
        // Check for Ctrl-C signal first
        check_ctrl_c_signal(&ctrl_c_rx)?;

        match line_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Ok(line)) => {
                bytes_read += line.len();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..]; // Remove "data: " prefix
                chunks_processed += 1;

                if data == "" {
                    continue;
                }

                if data == "[DONE]" {
                    break;
                }

                let streaming_response: StreamingResponse = match serde_json::from_str(data) {
                    Ok(response) => response,
                    Err(e) => {
                        // Skip invalid JSON chunks - this is common in streaming responses
                        debug!("Skipping invalid JSON chunk: {}, data: '{}'", e, data);
                        // Report parsing issues for debugging
                        if data.len() > 10 {
                            warn!(
                                "Large chunk failed to parse, potential data loss: {} chars",
                                data.len()
                            );
                        }
                        continue;
                    }
                };

                if let Some(error) = streaming_response.error {
                    return Err(format!("Streaming API error: {}", error.message).into());
                }

                if let Some(choices) = streaming_response.choices {
                    if let Some(choice) = choices.first() {
                        // Report thinking status on first meaningful chunk
                        if !thinking_reported {
                            let progress_info = ProgressInfo {
                                status: StatusUpdate::Thinking,
                                elapsed_ms: start_time.elapsed().as_millis() as u64,
                            };
                            let _ = progress_handler(&progress_info);
                            thinking_reported = true;
                        }

                        // Check if we have tool calls - if so, enter tool mode
                        if choice.delta.tool_calls.is_some() {
                            in_tool_mode = true;
                        }

                        if let Some(content) = &choice.delta.content {
                            if content == "" {
                                continue;
                            }

                            accumulated_content.push_str(content);

                            // Only call stream handler if we're not in tool mode
                            if !in_tool_mode {
                                stream_handler(content)?;

                                if chunks_processed % 10 == 0 {
                                    let progress_info = ProgressInfo {
                                        status: StatusUpdate::StreamProcessing {
                                            bytes_read,
                                            chunks_processed,
                                        },
                                        elapsed_ms: 0,
                                    };
                                    let _ = progress_handler(&progress_info);
                                }
                            }
                        }

                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            // Set accumulation start time if this is the first tool call chunk
                            if tool_accumulation_start.is_none() {
                                tool_accumulation_start = Some(std::time::Instant::now());
                            }

                            for streaming_tool_call in tool_calls {
                                let index =
                                    streaming_tool_call.index.unwrap_or(usize::MAX as u64) as usize;

                                let updated_tool_call = match accumulated_tool_calls.get_mut(&index)
                                {
                                    Some(existing_call) => {
                                        // Accumulate the arguments
                                        if let Some(args) = &streaming_tool_call.function.arguments
                                        {
                                            existing_call.function.arguments.push_str(args);
                                        }
                                        // Update other fields if they have values
                                        if let Some(name) = &streaming_tool_call.function.name {
                                            if !name.is_empty() {
                                                existing_call.function.name = name.clone();
                                            }
                                        }
                                        if let Some(id) = &streaming_tool_call.id {
                                            if !id.is_empty() {
                                                existing_call.id = id.clone();
                                            }
                                        }
                                        if let Some(tool_type) = &streaming_tool_call.tool_type {
                                            if !tool_type.is_empty() {
                                                existing_call.tool_type = tool_type.clone();
                                            }
                                        }
                                        existing_call.clone()
                                    }
                                    None => {
                                        // First chunk for this tool call - convert to regular ToolCall
                                        debug!(
                                            "Creating new tool call at index {}, streaming_tool_call.id: {:?}",
                                            index, streaming_tool_call.id
                                        );
                                        // Use empty ID initially, will be filled when available or at the end
                                        let tool_call = ToolCall {
                                            index: streaming_tool_call.index,
                                            id: streaming_tool_call.id.clone().unwrap_or_default(),
                                            tool_type: streaming_tool_call
                                                .tool_type
                                                .clone()
                                                .unwrap_or("function".to_string()),
                                            function: FunctionCall {
                                                name: streaming_tool_call
                                                    .function
                                                    .name
                                                    .clone()
                                                    .unwrap_or_default(),
                                                arguments: streaming_tool_call
                                                    .function
                                                    .arguments
                                                    .clone()
                                                    .unwrap_or_default(),
                                            },
                                        };
                                        debug!("Created tool call: {:?}", tool_call);
                                        accumulated_tool_calls.insert(index, tool_call.clone());
                                        tool_call
                                    }
                                };

                                // Show accumulating status if we have a progress handler
                                if let ResponseMode::Streaming {
                                    progress_handler, ..
                                } = mode
                                {
                                    let elapsed_ms = tool_accumulation_start
                                        .map(|start| start.elapsed().as_millis() as u64)
                                        .unwrap_or(0);

                                    let progress_info = ProgressInfo {
                                        status: StatusUpdate::ToolAccumulating {
                                            name: updated_tool_call.function.name.clone(),
                                            arguments: updated_tool_call.function.arguments.clone(),
                                        },
                                        elapsed_ms,
                                    };
                                    let _ = progress_handler(&progress_info);
                                }
                            }
                        }

                        if choice.finish_reason.is_some() {
                            finish_reason = choice.finish_reason.clone();
                        }
                    }
                }

                // Capture usage information if available in this chunk
                if let Some(chunk_usage) = streaming_response.usage {
                    usage = Some(chunk_usage);
                }
            }
            Ok(Err(e)) => {
                // Line reading error
                return Err(Box::new(e));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Timeout - continue checking for Ctrl-C signals
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Reader thread finished - break out of loop
                break;
            }
        }
    }

    // Ensure we call the handler one final time to flush output
    stream_handler("")?;

    // Build the complete response
    let tool_calls_vec: Option<Vec<ToolCall>> = if accumulated_tool_calls.is_empty() {
        None
    } else {
        let calls: Vec<ToolCall> = accumulated_tool_calls.into_values().collect();

        debug!("Accumulated calls {:?}", calls);

        // Use index as ID when actual ID is not available
        let calls_with_ids: Vec<ToolCall> = calls
            .into_iter()
            .map(|mut call| {
                if call.id.is_empty() {
                    call.id = call.index.unwrap_or(0).to_string();
                    debug!("Using index '{}' as ID for tool call", call.id);
                }
                call
            })
            .collect();

        // Validate tool calls are complete before including them
        let valid_calls: Vec<ToolCall> = calls_with_ids
            .into_iter()
            .filter(|call| {
                if call.function.name.is_empty() {
                    warn!("Dropping tool call with empty name");
                    false
                } else if call.function.arguments.is_empty() {
                    warn!(
                        "Dropping tool call '{}' with empty arguments",
                        call.function.name
                    );
                    false
                } else if call.id.is_empty() {
                    warn!("Dropping tool call '{}' with empty ID", call.function.name);
                    false
                } else {
                    true
                }
            })
            .collect();

        if valid_calls.is_empty() {
            warn!("All tool calls were invalid and dropped, clearing finish_reason");
            // If all tool calls were invalid, clear finish_reason to continue conversation
            // instead of causing "Invalid response" error or premature exit
            finish_reason = None;
            None
        } else {
            // Check for duplicate IDs and make them unique
            let mut seen_ids = std::collections::HashSet::new();
            let mut unique_calls: Vec<ToolCall> = valid_calls
                .into_iter()
                .map(|mut call| {
                    if seen_ids.contains(&call.id) {
                        // Make ID unique by appending the index
                        let new_id = format!("{}_{}", call.id, call.index.unwrap_or(0));
                        warn!(
                            "Duplicate tool call ID '{}' found, changing to '{}'",
                            call.id, new_id
                        );
                        call.id = new_id;
                    }
                    seen_ids.insert(call.id.clone());
                    call
                })
                .collect();

            unique_calls.sort_by_key(|call| call.index.unwrap_or(usize::MAX as u64));
            Some(unique_calls)
        }
    };

    let message = Message {
        role: "assistant".to_string(),
        content: if accumulated_content.is_empty() {
            None
        } else {
            Some(accumulated_content)
        },
        tool_calls: tool_calls_vec,
        tool_call_id: None,
        name: None,
    };

    let choice = Choice {
        message,
        finish_reason,
        native_finish_reason: None,
    };

    // Use captured usage information from streaming response if available

    Ok(OpenAIResponse {
        error: None,
        choices: Some(vec![choice]),
        usage,
        history: vec![],
    })
}
