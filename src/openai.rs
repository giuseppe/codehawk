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

use crate::github;

use dirs;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::process::Command;
use std::time::Duration;

use github::{get_github_issue, get_github_issue_comments};

pub struct Opts {
    pub max_tokens: Option<u32>,
    pub model: Option<String>,
}

const OPEN_ROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const MODEL: &str = "google/gemini-2.5-pro-preview-03-25";
const MAX_TOKENS: u32 = 16384;

/// Reads the OpenRouter API key from the file `~/.openrouter/key`.
fn read_api_key() -> Result<String, Box<dyn Error>> {
    let home_dir = dirs::home_dir().ok_or("Could not find home directory")?;
    let key_path = home_dir.join(".openrouter").join("key");

    let mut file = File::open(&key_path)
        .map_err(|e| format!("Failed to open key file at {:?}: {}", key_path, e))?;

    let mut api_key = String::new();
    file.read_to_string(&mut api_key)?;

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
    pub tools: Vec<Tool>,
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
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: String,
}

/// List of supported tools, must be in sync with `tool_call`.  It must be something like:
// {
//     "type": "function",
//     "function": {
//         "name": "get_weather",
//         "description": "Get current temperature for a given location.",
//         "parameters": {
//             "type": "object",
//             "properties": {
//                 "location": {
//                     "type": "string",
//                     "description": "City and country e.g. Bogotá, Colombia"
//                 }
//             },
//             "required": [
//                 "location"
//             ],
//             "additionalProperties": false
//         }
//     }
// }
const TOOLS_DATA: &str = r#"
    [
        {
            "type": "function",
            "function": {
                "name": "github_issue",
                "description": "Get the github issue description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "issue": {
                            "type": "number",
                            "description": "number of the github issue"
                        }
                    },
                    "required": [
                    ],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "github_issue_comments",
                "description": "Get the comments associated with the github issue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "issue": {
                            "type": "number",
                            "description": "number of the github issue"
                        }
                    },
                    "required": [
                    ],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_all_files",
                "description": "Get the list of all the files in the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [
                    ],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Get the content of a file stored in the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
    ]
    "#;

/// entrypoint for the list_all_files tool
fn tool_list_all_files() -> Result<String, Box<dyn Error>> {
    let mut cmd = Command::new("git");
    cmd.arg("ls-files");
    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr)?;
        let err: Box<dyn Error> = stderr.into();
        return Err(err);
    }
    let r = String::from_utf8(output.stdout)?;
    Ok(r)
}

/// entrypoint for the github_issue tool
fn tool_github_issue(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        issue: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;
    let issue = get_github_issue(&params.repo, params.issue)?;
    Ok(serde_json::to_string(&issue)?)
}

/// entrypoint for the github_issue tool
fn tool_github_issue_comments(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        issue: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;
    let comments = get_github_issue_comments(&params.repo, params.issue)?;
    Ok(serde_json::to_string(&comments)?)
}

/// entrypoint for the read_file tool
fn tool_read_file(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = Command::new("git");
    cmd.arg("show").arg(format!("HEAD:{}", params.path));

    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr)?;
        let err: Box<dyn Error> = stderr.into();
        return Err(err);
    }
    let r = String::from_utf8(output.stdout)?;
    Ok(r)
}

/// Perform a tool call and return the message to send back.
fn tool_call(req: &ToolCall) -> Result<Message, Box<dyn Error>> {
    let content: String = match req.function.name.as_str() {
        "list_all_files" => tool_list_all_files()?,
        "read_file" => tool_read_file(&req.function.arguments)?,
        "github_issue" => tool_github_issue(&req.function.arguments)?,
        "github_issue_comments" => tool_github_issue_comments(&req.function.arguments)?,
        _ => return Err("invalid tool used".into()),
    };

    let msg = Message {
        role: "tool".to_string(),
        content: content,
        tool_call_id: Some(req.id.clone()),
        name: Some(req.function.name.clone()),
        tool_calls: None,
    };
    Ok(msg)
}

/// Sends a POST request to the OpenAI API with the given prompt and options.
pub fn post_request(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    other_messages: Option<Vec<Message>>,
    opts: &Opts,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let api_key = read_api_key()?;

    let bearer_auth = format!("Bearer {}", &api_key);

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_auth)?);

    let model: String = match &opts.model {
        Some(s) => s.to_string(),
        None => MODEL.to_string(),
    };

    let tools: Vec<Tool> = serde_json::from_str::<Vec<Tool>>(&TOOLS_DATA)?;

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
    messages.push(Message {
        role: "user".to_string(),
        content: prompt.to_string(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    if let Some(other_messages) = &other_messages {
        for msg in other_messages {
            messages.push(msg.clone());
        }
    }

    let request_body = OpenAIRequest {
        model: model,
        max_tokens: opts.max_tokens.unwrap_or_else(|| MAX_TOKENS),
        messages: messages,
        tools: tools,
    };

    let response = Client::new()
        .post(OPEN_ROUTER_URL)
        .timeout(Duration::from_secs(1000))
        .headers(headers)
        .json(&request_body)
        .send()?;

    if !response.status().is_success() {
        return Err(format!(
            "got error code: {}: {}",
            response.status(),
            response.text()?
        )
        .into());
    }

    let openai_response: OpenAIResponse = response.json()?;

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
        for choice in choices {
            if choice.finish_reason == "tool_calls" {
                let tool_calls = choice
                    .message
                    .tool_calls
                    .as_ref()
                    .ok_or("Invalid response")?;
                tool_call_messages.push(choice.message.clone());
                for tool_call_request in tool_calls {
                    let msg = tool_call(tool_call_request)?;
                    tool_call_messages.push(msg);
                }
            }
        }
    }
    // If there were tool requests, repeat the request with the new results
    if tool_call_messages.len() > 0 {
        if let Some(mut other_messages) = other_messages {
            tool_call_messages.append(&mut other_messages);
        }

        return post_request(&prompt, system_prompts, Some(tool_call_messages), &opts);
    }
    Ok(openai_response)
}
