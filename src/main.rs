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

mod github;

use clap::{Parser, Subcommand};
use dirs;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::process::Command;
use std::time::Duration;
use string_builder::Builder;

use github::{
    Issues, PullRequests, get_github_issue, get_github_issue_comments, get_github_issues,
    get_github_pull_request, get_github_pull_request_patch, get_github_pull_requests,
};

const OPEN_ROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const MODEL: &str = "google/gemini-2.5-pro-preview-03-25";
const MAX_TOKENS: u32 = 16384;
const DEFAULT_DAYS: u64 = 7;

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
struct ParameterProperty {
    #[serde(rename = "type")]
    param_type: String,
    description: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ToolParameters {
    #[serde(rename = "type")]
    param_type: String,
    properties: HashMap<String, ParameterProperty>,
    required: Vec<String>,

    #[serde(rename = "additionalProperties")]
    additional_properties: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct Function {
    name: String,
    description: String,
    parameters: ToolParameters,
}

#[derive(Serialize, Deserialize, Debug)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: Function,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ToolCall {
    index: u64,
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: FunctionCall,
}

#[derive(Serialize, Debug)]
struct OpenRouterRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Message {
    role: String,
    content: String,
    tool_call_id: Option<String>,
    name: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterErrorMetadata {
    raw: String,
}

#[derive(Deserialize, Debug)]
struct OpenRouterError {
    code: Option<u64>,
    message: String,
    metadata: Option<OpenRouterErrorMetadata>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterResponse {
    error: Option<OpenRouterError>,
    choices: Option<Vec<Choice>>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: Message,
    finish_reason: String,
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

/// Sends a POST request to the OpenRouter API with the given prompt and options.
fn open_router_post_request(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    other_messages: Option<Vec<Message>>,
    opts: &Opts,
) -> Result<OpenRouterResponse, Box<dyn Error>> {
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

    let request_body = OpenRouterRequest {
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

    let open_router_response: OpenRouterResponse = response.json()?;

    if let Some(mut err) = open_router_response.error {
        if let Some(metadata) = err.metadata {
            let raw_response: OpenRouterResponse =
                serde_json::from_str::<OpenRouterResponse>(&metadata.raw)?;
            if let Some(inner_err) = raw_response.error {
                err = inner_err;
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
    if let Some(choices) = &open_router_response.choices {
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

        return open_router_post_request(&prompt, system_prompts, Some(tool_call_messages), &opts);
    }
    Ok(open_router_response)
}

/// Sends a prompt to the OpenRouter API and prints the AI's response to standard output.
fn post_request_and_print_output(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    let response: OpenRouterResponse =
        open_router_post_request(&prompt, system_prompts, None, opts)?;
    let mut builder = Builder::default();
    if let Some(choices) = response.choices {
        for choice in choices {
            builder.append(choice.message.content);
        }
    }
    let msg = builder.string()?;
    println!("{}", &msg);
    Ok(())
}

/// Fetches a GitHub pull request and its patch, then sends them to the AI for review.
fn review_pull_request(repo: &String, pr_id: u64, opts: &Opts) -> Result<(), Box<dyn Error>> {
    let pr = get_github_pull_request(repo, pr_id)?;
    let patch = get_github_pull_request_patch(repo, pr_id)?;

    let system_prompts: Vec<String> = vec![patch, serde_json::to_string(&pr)?];
    let prompt = "Review the following pull request and report any issue with it, pay attention to the code.  Report only what is wrong, don't highlight what is done correctly.".to_string();

    post_request_and_print_output(&prompt, Some(system_prompts), opts)
}

/// Fetches a GitHub issue and its comments, then sends them to the AI for triaging.
fn triage_issue(repo: &String, issue_id: u64, opts: &Opts) -> Result<(), Box<dyn Error>> {
    let issue = get_github_issue(repo, issue_id)?;
    let comments = get_github_issue_comments(repo, issue_id)?;
    let prompt = "Provide a triage for the specified issue, show a minimal reproducer for the issue reducing the dependencies needed to run
it.".to_string();

    let system_prompts: Vec<String> = vec![
        serde_json::to_string(&issue)?,
        serde_json::to_string(&comments)?,
    ];

    post_request_and_print_output(&prompt, Some(system_prompts), opts)
}

/// Fetches recent issues and pull requests from specified repositories and sends them to the AI with a given command prompt.
fn prompt_issues_and_pull_requests(
    prompt: &str,
    repos: &Vec<String>,
    days: Option<u64>,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    let days = days.unwrap_or_else(|| DEFAULT_DAYS);
    let mut issues: Issues = Issues::new();
    let mut pull_requests: PullRequests = PullRequests::new();
    for repo in repos {
        let mut repo_issues = get_github_issues(repo, days)?;
        issues.append(&mut repo_issues);

        let mut repo_pull_requests = get_github_pull_requests(repo, days)?;
        pull_requests.append(&mut repo_pull_requests);
    }
    let system_prompts: Vec<String> = vec![
        serde_json::to_string(&issues)?,
        serde_json::to_string(&pull_requests)?,
    ];

    let prompt = prompt.to_string();
    post_request_and_print_output(&prompt, Some(system_prompts), opts)
}

/// Analyzes recent issues and pull requests for the specified repositories.
fn analyze_repos(
    repos: &Vec<String>,
    days: Option<u64>,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    prompt_issues_and_pull_requests(
        "Provide a summary of all the issues and pull requests listed, highlighting the most important ones\n",
        repos,
        days,
        opts,
    )
}

/// Prioritizes recent issues and pull requests for the specified repositories.
fn prioritize_repos(
    repos: &Vec<String>,
    days: Option<u64>,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    prompt_issues_and_pull_requests(
        "Given the issues and pull requests listed, order them by importance and highlight the ones I must address first and why\n",
        repos,
        days,
        opts,
    )
}

/// Sends the concatenated content of specified files as a prompt to the AI.
fn prompt_command(prompt: &String, files: &Vec<String>, opts: &Opts) -> Result<(), Box<dyn Error>> {
    let mut system_prompts: Vec<String> = vec![];

    for file in files {
        let contents = fs::read_to_string(file)?;
        system_prompts.push(contents);
    }

    post_request_and_print_output(prompt, Some(system_prompts), opts)
}

#[derive(Parser, Debug)]
#[clap(version = env!("CARGO_PKG_VERSION"))]
struct Opts {
    #[clap(short, long)]
    max_tokens: Option<u32>,
    #[clap(long)]
    model: Option<String>,

    #[clap(subcommand)]
    command: CliCommand,

    #[clap()]
    args: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Prioritize the issues and pull requests happened in the last DAYS
    Prioritize {
        /// Maximum age in days for the issue or pull request
        #[clap(long)]
        days: Option<u64>,
        /// Repository
        repo: Vec<String>,
    },
    /// Analyze the issues and pull requests happened in the last DAYS
    Analyze {
        /// Maximum age in days for the issue or pull request
        #[clap(long)]
        days: Option<u64>,
        /// Repository
        repo: Vec<String>,
    },
    /// Triage a specific issue
    Triage {
        /// Repository
        repo: String,
        /// Issue number
        issue: u64,
    },

    /// Review a pull request
    Review {
        /// Repository
        repo: String,
        /// PR number
        pr: u64,
    },

    /// Pass a request to the AI model and print its response
    Prompt {
        /// Prompt command to pass to the AI model
        prompt: String,
        /// List of files that are loaded and used as system context
        files: Vec<String>,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let opts = Opts::parse();
    match opts.command {
        CliCommand::Analyze { days, ref repo } => analyze_repos(&repo, days, &opts)?,
        CliCommand::Prioritize { days, ref repo } => prioritize_repos(&repo, days, &opts)?,
        CliCommand::Triage { ref repo, issue } => triage_issue(&repo, issue, &opts)?,
        CliCommand::Review { ref repo, pr } => review_pull_request(&repo, pr, &opts)?,
        CliCommand::Prompt {
            ref prompt,
            ref files,
        } => prompt_command(&prompt, &files, &opts)?,
    }
    Ok(())
}
