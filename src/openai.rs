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
use log::{debug, info, trace};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::time::Duration;

pub struct Opts {
    pub max_tokens: Option<u32>,
    pub model: String,
    pub endpoint: String,
}

pub type ToolCallback = fn(&String) -> Result<String, Box<dyn Error>>;
pub type ToolsCollection = HashMap<String, ToolItem>;
pub struct ToolItem {
    pub callback: ToolCallback,
    pub schema: String,
}

const MAX_TOKENS: u32 = 16384;
const OPEN_ROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Reads the OpenRouter API key from the file `~/.openrouter/key`.
fn read_api_key() -> Result<String, Box<dyn Error>> {
    let home_dir = dirs::home_dir().ok_or("Could not find home directory")?;
    let key_path = home_dir.join(".openrouter").join("key");

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
    pub index: u64,
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
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
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

#[derive(Deserialize, Debug)]
pub struct OpenAIResponse {
    pub error: Option<OpenAIError>,
    pub choices: Option<Vec<Choice>>,

    #[serde(skip_deserializing)]
    pub history: Vec<Message>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
    pub native_finish_reason: Option<String>,
}

/// Perform a tool call and return the message to send back.
fn tool_call(
    tools_collection: &ToolsCollection,
    req: &ToolCall,
) -> Result<Message, Box<dyn Error>> {
    let tool_name = &req.function.name;
    info!("Requesting tool {}", tool_name);
    let tool = tools_collection.get(tool_name);
    let content: String = match tool {
        None => return Err("invalid tool requested".into()),
        Some(t) => {
            info!("Executing tool {}", tool_name);
            debug!(
                "Passing arguments {:?} to tool {}",
                req.function.arguments, tool_name
            );
            (t.callback)(&req.function.arguments)?
        }
    };

    trace!("Tool {} gave output {:?}", tool_name, content);

    let msg = Message {
        role: "tool".to_string(),
        content: content,
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
        content: content,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

/// Sends a POST request to the OpenAI API with the given prompt and options.
pub fn post_request(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    other_messages: Option<Vec<Message>>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let mut messages: Vec<Message> = vec![];

    if let Some(system_prompts) = &system_prompts {
        for system_prompt in system_prompts {
            messages.push(Message {
                role: "system".to_string(),
                content: system_prompt.clone(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    if let Some(other_messages) = &other_messages {
        for msg in other_messages {
            messages.push(msg.clone());
        }
    }

    messages.push(Message {
        role: "user".to_string(),
        content: prompt.to_string(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    post_request_messages(messages, tools_collection, opts)
}

/// Sends a POST request to the OpenAI API with the given messages and options.
pub fn post_request_messages(
    mut messages: Vec<Message>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if opts.endpoint == OPEN_ROUTER_URL {
        let api_key = read_api_key()?;
        let bearer_auth = format!("Bearer {}", &api_key);
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_auth)?);
    }

    let mut tools: Vec<Tool> = vec![];
    for t in tools_collection.values() {
        let tool_schema = serde_json::from_str::<Tool>(&t.schema)?;
        tools.push(tool_schema);
    }

    let request_body = OpenAIRequest {
        model: opts.model.clone(),
        max_tokens: opts.max_tokens.unwrap_or_else(|| MAX_TOKENS),
        messages: messages,
        tools: if tools.len() > 0 { Some(tools) } else { None },
    };

    let response = Client::new()
        .post(&opts.endpoint)
        .timeout(Duration::from_secs(1000))
        .headers(headers)
        .json(&request_body)
        .send()?;

    messages = request_body.messages;

    if !response.status().is_success() {
        return Err(format!(
            "got error code: {}: {}",
            response.status(),
            response.text()?
        )
        .into());
    }

    let response_text = response.text()?;

    trace!("Got response {:?}", response_text);

    let mut openai_response: OpenAIResponse = serde_json::from_str(&response_text)?;

    if let Some(mut err) = openai_response.error {
        if err.metadata.is_some() {
            if let Some(raw) = err.metadata.unwrap().raw {
                let raw_response: OpenAIResponse = serde_json::from_str::<OpenAIResponse>(&raw)?;
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

    let mut tool_call_messages: Vec<Message> = vec![];
    if let Some(choices) = &openai_response.choices {
        trace!("Got {} choices", choices.len());
        for choice in choices {
            trace!("Got choice {:?}", choice);
            let finish_reason = choice
                .finish_reason
                .clone()
                .unwrap_or_else(|| "".to_string());
            if finish_reason == "error" {
                let native_finish_reason = choice
                    .native_finish_reason
                    .clone()
                    .unwrap_or_else(|| finish_reason);
                return Err(format!("got API error: {}", native_finish_reason).into());
            } else if finish_reason == "tool_calls" {
                let tool_calls = choice
                    .message
                    .tool_calls
                    .as_ref()
                    .ok_or("Invalid response")?;
                let tool_request_msg = choice.message.clone();
                trace!("Add tool request message: {:?}", tool_request_msg);
                tool_call_messages.push(tool_request_msg);
                for tool_call_request in tool_calls {
                    let msg = tool_call(&tools_collection, tool_call_request)?;
                    trace!("Add tool response message: {:?}", msg);
                    tool_call_messages.push(msg);
                }
            }
        }
    }
    // If there were tool requests, repeat the request with the new results
    if tool_call_messages.len() > 0 {
        info!(
            "Repeat request with {} new messages from tools",
            tool_call_messages.len()
        );
        messages.extend(tool_call_messages);

        return post_request_messages(messages, &tools_collection, &opts);
    }
    openai_response.history = messages;
    Ok(openai_response)
}
