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
mod openai;

use clap::{Parser, Subcommand};
use env_logger::Env;
use log::{debug, trace, warn};
use pathrs::{Root, flags::OpenFlags};
use prettytable::{Cell, Row, Table, format};
use rustyline::DefaultEditor;
use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::fs::Permissions;
use std::io::Read;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

use github::{
    Issues, PullRequests, get_github_issue, get_github_issue_comments, get_github_issues,
    get_github_pull_request, get_github_pull_request_patch, get_github_pull_requests,
};
use openai::{
    Message, OpenAIResponse, ToolCallback, ToolItem, ToolsCollection, list_models, make_message,
    post_request,
};

const OPEN_ROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL: &str = "google/gemini-2.5-pro";
const DEFAULT_DAYS: u64 = 7;

fn append_tool(tools: &mut ToolsCollection, name: String, callback: ToolCallback, schema: String) {
    let item = ToolItem {
        callback: callback,
        schema: schema,
    };
    debug!("Adding tool: {}", name);
    tools.insert(name, item);
}

/// entrypoint for the delete_path tool
fn tool_delete_path(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
    }
    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Remove path: {}", params.path);
    let root = Root::open(".")?;

    let path = PathBuf::from(&params.path);

    root.remove_all(path)?;
    Ok("deleted".to_owned())
}

/// entrypoint for the read_file tool
fn tool_read_file(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
    }
    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Reading file: {}", params.path);

    let root = Root::open(".")?;
    let path = PathBuf::from(&params.path);
    let mut file = root.open_subpath(path, OpenFlags::O_RDONLY)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}

/// entrypoint for the write_file tool
fn tool_write_file(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
        content: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Writing to file: {}", params.path);
    let root = Root::open(".")?;

    if let Some(parent) = PathBuf::from(&params.path).parent() {
        if !parent.as_os_str().is_empty() {
            root.mkdir_all(parent, &Permissions::from_mode(0o755))?;
        }
    }

    let mut file = root
        .open_subpath(&params.path, OpenFlags::O_WRONLY | OpenFlags::O_TRUNC)
        .or_else(|_| {
            debug!("Creating new file: {}", params.path);
            root.create_file(
                &params.path,
                OpenFlags::O_WRONLY,
                &Permissions::from_mode(0o600),
            )
        })?;

    file.write_all(&params.content.as_bytes())?;
    Ok("".into())
}

/// entrypoint for the list_git_files tool
fn tool_list_git_files(_params_str: &String) -> Result<String, Box<dyn Error>> {
    let mut cmd = Command::new("git");
    cmd.arg("ls-files");

    trace!("Executing git command: {:?}", cmd);
    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr)?;
        let err: Box<dyn Error> = stderr.into();
        return Err(err);
    }

    let r = String::from_utf8(output.stdout)?;
    debug!("Successfully listed {} files", r.lines().count());
    Ok(r)
}

/// entrypoint for the run_command tool
fn tool_run_command(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        command: String,
        args: Option<Vec<String>>,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = Command::new(&params.command);
    if let Some(args) = params.args {
        cmd.args(args);
    }

    trace!("Executing command: {:?}", cmd);
    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr)?;
        let err: Box<dyn Error> = stderr.into();
        return Err(err);
    }

    let r = String::from_utf8(output.stdout)?;
    debug!("Successfully run command {}", params.command);
    Ok(r)
}

/// entrypoint for the grep_in_current_directory tool
fn tool_grep_in_current_directory(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        pattern: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = Command::new("grep");
    cmd.arg("-r");
    cmd.arg("-n");
    cmd.arg(&params.pattern);

    debug!(
        "Grepping for pattern '{}' in current directory",
        params.pattern
    );

    trace!("Executing grep command: {:?}", cmd);
    let output = cmd.output()?;
    // Grep returns 1 if no lines were selected, 0 if lines were selected, >1 for errors.
    // We consider no lines selected as a valid, empty result, not an error for the tool.
    if !output.status.success() && output.status.code() != Some(1) {
        let stderr = String::from_utf8(output.stderr)?;
        let err_msg = format!(
            "grep command failed with status {:?}. Stderr: {}",
            output.status, stderr
        );
        let err: Box<dyn Error> = err_msg.into();
        return Err(err);
    }

    let r = String::from_utf8(output.stdout)?;
    debug!(
        "Grep command successfully executed for pattern '{}'",
        params.pattern
    );
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

    debug!("Fetching GitHub issue: {}/{}", params.repo, params.issue);
    let issue = get_github_issue(&params.repo, params.issue)?;

    let s = serde_json::to_string(&issue)?;
    Ok(s)
}

/// entrypoint for the github_issue_comments tool
fn tool_github_issue_comments(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        issue: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub issue comments: {}/{}",
        params.repo, params.issue
    );
    let comments = get_github_issue_comments(&params.repo, params.issue)?;
    let s = serde_json::to_string(&comments)?;
    Ok(s)
}

/// entrypoint for the github_pull_request tool
fn tool_github_pull_request(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        pull_request: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub PR: {}/{}",
        params.repo, params.pull_request
    );
    let pr = get_github_pull_request(&params.repo, params.pull_request)?;
    let s = serde_json::to_string(&pr)?;
    Ok(s)
}

/// entrypoint for the github_pull_request_patch tool
fn tool_github_pull_request_patch(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        pull_request: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub PR patch: {}/{}",
        params.repo, params.pull_request
    );
    let pr = get_github_pull_request_patch(&params.repo, params.pull_request)?;
    Ok(pr)
}

fn initialize_tools(unsafe_tools: bool) -> ToolsCollection {
    let mut tools: ToolsCollection = ToolsCollection::new();

    append_tool(
        &mut tools,
        "github_pull_request".to_string(),
        tool_github_pull_request,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_request",
                "description": "Get information about a github pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "pull_request": {
                            "type": "number",
                            "description": "number of the pull request"
                        }
                    },
                    "required": [
                        "repo",
                        "pull_request"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_pull_request_patch".to_string(),
        tool_github_pull_request_patch,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_request_patch",
                "description": "Get the raw patch for a github pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "pull_request": {
                            "type": "number",
                            "description": "number of the pull request"
                        }
                    },
                    "required": [
                        "repo",
                        "pull_request"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "read_file".to_string(),
        tool_read_file,
        r#"
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
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "delete_path".to_string(),
        tool_delete_path,
        r#"
        {
            "type": "function",
            "function": {
                "name": "delete_path",
                "description": "Delete a file or a directory under the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the current directory"
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "write_file".to_string(),
        tool_write_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or replace the content of a file stored in the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        },
                        "content": {
                            "type": "string",
                            "description": "the content of the new file"
                        }
                    },
                    "required": [
                        "path",
                        "content"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "list_git_files".to_string(),
        tool_list_git_files,
        r#"
        {
            "type": "function",
            "function": {
                "name": "list_git_files",
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
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "grep_in_current_directory".to_string(),
        tool_grep_in_current_directory,
        r#"
        {
            "type": "function",
            "function": {
                "name": "grep_in_current_directory",
                "description": "Grep for a pattern in the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The pattern to search for."
                        }
                    },
                    "required": [
                        "pattern"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issue".to_string(),
        tool_github_issue,
        r#"
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
                        "repo",
                        "issue"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issue_comments".to_string(),
        tool_github_issue_comments,
        r#"
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
                        "repo",
                        "issue"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    if !unsafe_tools {
        return tools;
    }

    append_tool(
        &mut tools,
        "run_command".to_string(),
        tool_run_command,
        r#"
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a command and gets it output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "command to execute, e.g. /usr/bin/ls"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Arguments to pass to the command, e.g. [\"-l\", \"-a\"]"
                        }
                    },
                    "required": [
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    debug!("Tools initialization completed with {} tools", tools.len());
    tools
}

fn add_conservative_prompt(messages: &mut Vec<Message>) {
    let conservative_prompt = "You are a helpful assistant.  You have access to various tools for file operations and code analysis.  Only use these tools when the user explicitly asks for file operations, code analysis, or repository interactions.  For simple questions, conversations, or general requests, respond directly without using tools.".to_string();
    messages.push(make_message("system", conservative_prompt));
}

/// Sends a prompt to the OpenAI API and prints the AI's response to standard output.
fn post_request_and_print_output(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    debug!("Prompt: {}", prompt);

    let model = opts.model.clone().unwrap_or(DEFAULT_MODEL.to_string());
    debug!("Using model: {}", model);

    let openai_opts = openai::Opts {
        max_tokens: opts.max_tokens,
        model: model,
        endpoint: opts
            .endpoint
            .clone()
            .unwrap_or_else(|| OPEN_ROUTER_URL.to_string()),
        tool_choice: opts.tool_choice.clone(),
    };

    let tools = match opts.no_tools {
        true => {
            debug!("Tools are disabled");
            ToolsCollection::new()
        }
        false => {
            debug!("Initializing tools for AI request");
            initialize_tools(opts.unsafe_tools)
        }
    };

    let mut messages: Vec<Message> = vec![];

    // Add conservative tool usage system prompt if tools are available but no explicit choice
    if !tools.is_empty() && opts.tool_choice.is_none() {
        add_conservative_prompt(&mut messages);
    }

    if let Some(ref sys_prompts) = system_prompts {
        debug!("Using {} system prompts", sys_prompts.len());
        for sp in sys_prompts {
            messages.push(make_message("system", sp.clone()));
        }
    }
    messages.push(make_message("user", prompt.clone()));

    let response: OpenAIResponse = post_request(messages, &tools, &openai_opts)?;

    if let Some(choices) = response.choices {
        debug!("Received {} choices in response", choices.len());
        if let Some(choice) = choices.first() {
            if let Some(content) = &choice.message.content {
                println!("{}", content);
            }
        }
    } else {
        warn!("No choices received in the AI response");
    }
    Ok(())
}

/// Fetches a GitHub pull request and its patch, then sends them to the AI for review.
fn review_pull_request(repo: &String, pr_id: u64, opts: &Opts) -> Result<(), Box<dyn Error>> {
    debug!("Reviewing pull request {}/{}", repo, pr_id);

    let pr = get_github_pull_request(repo, pr_id)?;
    let patch = get_github_pull_request_patch(repo, pr_id)?;

    let pr_json = serde_json::to_string(&pr)?;

    let system_prompts: Vec<String> = vec![patch, pr_json];
    let prompt = "Review the following pull request and report any issue with it, pay attention to the code.  Report only what is wrong, don't highlight what is done correctly.".to_string();

    post_request_and_print_output(&prompt, Some(system_prompts), opts)
}

/// Fetches a GitHub issue and its comments, then sends them to the AI for triaging.
fn triage_issue(repo: &String, issue_id: u64, opts: &Opts) -> Result<(), Box<dyn Error>> {
    debug!("Triaging issue {}/{}", repo, issue_id);

    let issue = get_github_issue(repo, issue_id)?;
    let comments = get_github_issue_comments(repo, issue_id)?;

    let prompt = "Provide a triage for the specified issue, show a minimal reproducer for the issue reducing the dependencies needed to run it.".to_string();

    let issue_json = serde_json::to_string(&issue)?;
    let comments_json = serde_json::to_string(&comments)?;

    let system_prompts: Vec<String> = vec![issue_json, comments_json];

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
    debug!(
        "Fetching issues and PRs from {} repositories for the past {} days",
        repos.len(),
        days
    );

    let mut issues: Issues = Issues::new();
    let mut pull_requests: PullRequests = PullRequests::new();

    for repo in repos {
        debug!("Processing repository: {}", repo);

        let mut repo_issues = get_github_issues(repo, days)?;
        issues.append(&mut repo_issues);

        let mut repo_pull_requests = get_github_pull_requests(repo, days)?;
        pull_requests.append(&mut repo_pull_requests);
    }

    debug!(
        "Total: {} issues and {} pull requests found",
        issues.len(),
        pull_requests.len()
    );

    let issues_json = serde_json::to_string(&issues)?;
    let prs_json = serde_json::to_string(&pull_requests)?;
    let system_prompts: Vec<String> = vec![issues_json, prs_json];
    let prompt_string = prompt.to_string();

    post_request_and_print_output(&prompt_string, Some(system_prompts), opts)
}

/// Analyzes recent issues and pull requests for the specified repositories.
fn analyze_repos(
    repos: &Vec<String>,
    days: Option<u64>,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    debug!("Analyzing repos: {:?} for past {:?} days", repos, days);
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
    debug!("Prioritizing repos: {:?} for past {:?} days", repos, days);
    prompt_issues_and_pull_requests(
        "Given the issues and pull requests listed, order them by importance and highlight the ones I must address first and why\n",
        repos,
        days,
        opts,
    )
}

/// Sends the concatenated content of specified files as a prompt to the AI.
fn prompt_command(prompt: &String, files: &Vec<String>, opts: &Opts) -> Result<(), Box<dyn Error>> {
    debug!("Executing prompt command with {} files", files.len());
    let mut system_prompts: Vec<String> = vec![];

    for file in files {
        debug!("Reading file for prompt context: {}", file);
        let contents = fs::read_to_string(file)?;
        system_prompts.push(contents);
    }
    post_request_and_print_output(prompt, Some(system_prompts), opts)
}

/// Interactive session
fn chat_command(opts: &Opts) -> Result<(), Box<dyn Error>> {
    debug!("Executing chat command");

    let mut rl = DefaultEditor::new()?;
    let mut messages: Vec<Message> = vec![];
    let tools = match opts.no_tools {
        true => {
            debug!("Tools are disabled");
            ToolsCollection::new()
        }
        false => {
            debug!("Initializing tools for AI request");
            initialize_tools(opts.unsafe_tools)
        }
    };

    // Add conservative tool usage system prompt if tools are available but no explicit choice
    if !tools.is_empty() && opts.tool_choice.is_none() {
        add_conservative_prompt(&mut messages);
    }

    let model = opts.model.clone().unwrap_or(DEFAULT_MODEL.to_string());
    debug!("Using model: {}", model);

    let openai_opts = openai::Opts {
        max_tokens: opts.max_tokens,
        model: model,
        endpoint: opts
            .endpoint
            .clone()
            .unwrap_or_else(|| OPEN_ROUTER_URL.to_string()),
        tool_choice: opts.tool_choice.clone(),
    };

    loop {
        let line = rl.readline("> ")?.trim().to_string();
        if line.is_empty() {
            continue;
        }
        rl.add_history_entry(line.as_str())?;

        if line == "\\quit" {
            return Ok(());
        }
        if line == "\\clear" {
            messages = vec![];
            println!("Chat history cleared.");
            continue;
        }
        if line == "\\show" {
            if messages.is_empty() {
                println!("Chat history is empty.");
            } else {
                println!("Current chat history:");
                for (i, msg) in messages.iter().enumerate() {
                    println!(
                        "{}: [{}] {}",
                        i + 1,
                        msg.role,
                        msg.content.as_ref().unwrap_or(&"<no content>".to_string())
                    );
                    if let Some(tool_calls) = &msg.tool_calls {
                        if !tool_calls.is_empty() {
                            println!("  Tool Calls:");
                            for (j, tool_call) in tool_calls.iter().enumerate() {
                                println!(
                                    "    {}.{}: {} ({})",
                                    i + 1,
                                    j + 1,
                                    tool_call.function.name,
                                    tool_call.id
                                );
                                println!("      Args: {}", tool_call.function.arguments);
                            }
                        }
                    }
                }
            }
            continue;
        }
        if line.starts_with("\\limit ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                if let Ok(n) = parts[1].parse::<usize>() {
                    if n == 0 {
                        println!("Limit cannot be zero. Clearing history instead.");
                        messages = vec![];
                    } else if messages.len() > n {
                        // Keep the last n messages. System messages might be lost if not handled.
                        // Assuming simple user/assistant history for now.
                        messages = messages.split_off(messages.len() - n);
                        println!("Chat history limited to the last {} messages.", n);
                    } else {
                        println!("Chat history is already within the limit of {}.", n);
                    }
                } else {
                    println!("Invalid number for limit: {}", parts[1]);
                }
            } else {
                println!("Usage: \\limit <number_of_messages>");
            }
            continue;
        }
        if line.starts_with("\\backtrace ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                if let Ok(n) = parts[1].parse::<usize>() {
                    if n == 0 {
                        println!("Backtrace steps must be a positive number.");
                    } else if n > messages.len() {
                        println!(
                            "Cannot go back {} steps, history has only {} messages. Clearing history.",
                            n,
                            messages.len()
                        );
                        messages = vec![];
                    } else {
                        messages.truncate(messages.len() - n);
                        println!("Went back {} steps in chat history.", n);
                    }
                } else {
                    println!("Invalid number for backtrace: {}", parts[1]);
                }
            } else {
                println!("Usage: \\backtrace <number_of_messages>");
            }
            continue;
        }
        if line.starts_with("\\system ") {
            let system_message = line.strip_prefix("\\system ").unwrap_or("").to_string();
            if !system_message.is_empty() {
                messages.push(make_message("system", system_message));
                println!("System message added to conversation.");
            } else {
                println!("Usage: \\system <message>");
            }
            continue;
        }

        messages.push(make_message("user", line));

        let response = post_request(messages, &tools, &openai_opts)?;
        if let Some(choices) = response.choices {
            debug!("Received {} choices in response", choices.len());
            if let Some(choice) = choices.first() {
                if let Some(content) = &choice.message.content {
                    println!("{}", content);
                }
            }
        }
        messages = response.history;
    }
}

// ModelInfo and ModelsApiResponse structs are removed from here.

/// Handles the listing of models by calling the openai module.
fn list_models_command(_opts: &Opts) -> Result<(), Box<dyn Error>> {
    match list_models() {
        Ok(models) => {
            if models.is_empty() {
                println!("No models found.");
            } else {
                println!("Available models from OpenRouter:");
                let mut table = Table::new();
                table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
                table.set_titles(Row::new(vec![
                    Cell::new("ID"),
                    Cell::new("Name"),
                    Cell::new("Context Length"),
                    Cell::new("Prompt_USD/1M"),
                    Cell::new("Compl_USD/1M"),
                    Cell::new("Supported Parameters"),
                ]));

                for model in models {
                    let prompt_price_str = model.pricing.prompt;
                    let completion_price_str = model.pricing.completion;

                    let prompt_price = match prompt_price_str.parse::<f64>() {
                        Ok(p) => format!("{:.6}", p),
                        Err(_) => prompt_price_str,
                    };
                    let completion_price = match completion_price_str.parse::<f64>() {
                        Ok(c) => format!("{:.6}", c),
                        Err(_) => completion_price_str,
                    };
                    let supported_parameters = model.supported_parameters.join(",");

                    table.add_row(Row::new(vec![
                        Cell::new(&model.id),
                        Cell::new(&model.name),
                        Cell::new(&model.context_length.to_string()),
                        Cell::new(&prompt_price),
                        Cell::new(&completion_price),
                        Cell::new(&supported_parameters),
                    ]));
                }
                table.printstd();
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[derive(Parser, Debug)]
#[clap(version = env!("CARGO_PKG_VERSION"))]
struct Opts {
    #[clap(short, long)]
    /// Override the maximum number of tokens to generate
    max_tokens: Option<u32>,
    #[clap(long)]
    /// Specify the AI model to use
    model: Option<String>,
    #[clap(long)]
    /// Override the endpoint URL to use
    endpoint: Option<String>,
    #[clap(long)]
    /// Inhibit usage of any tool
    no_tools: bool,
    /// Enable unsafe tools
    #[clap(long)]
    unsafe_tools: bool,
    #[clap(long)]
    /// Control when tools are used: "auto" (default), "none", "required"
    tool_choice: Option<String>,

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

    /// Interactive session
    Chat {},

    /// List available models from OpenRouter
    Models {},
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize environment logger with custom configuration
    let env = Env::new()
        .filter_or("RUST_LOG", "warning")
        .write_style_or("LOG_STYLE", "always");

    env_logger::init_from_env(env);

    // Parse command line arguments
    let opts = Opts::parse();
    debug!("Command line options parsed");

    // Execute the chosen command
    let result = match &opts.command {
        CliCommand::Analyze { days, repo } => analyze_repos(&repo, *days, &opts),
        CliCommand::Prioritize { days, repo } => prioritize_repos(&repo, *days, &opts),
        CliCommand::Triage { repo, issue } => triage_issue(&repo, *issue, &opts),
        CliCommand::Review { repo, pr } => review_pull_request(&repo, *pr, &opts),
        CliCommand::Prompt { prompt, files } => prompt_command(&prompt, &files, &opts),
        CliCommand::Chat {} => chat_command(&opts),
        CliCommand::Models {} => list_models_command(&opts),
    };

    result
}
