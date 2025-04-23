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
use libc;
use openat2;
use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::os::fd::FromRawFd;
use std::path::Path;
use std::process::Command;
use string_builder::Builder;

use github::{
    Issues, PullRequests, get_github_issue, get_github_issue_comments, get_github_issues,
    get_github_pull_request, get_github_pull_request_patch, get_github_pull_requests,
};
use openai::{OpenAIResponse, ToolCallback, ToolItem, ToolsCollection, post_request};

const DEFAULT_DAYS: u64 = 7;

fn append_tool(tools: &mut ToolsCollection, name: String, callback: ToolCallback, schema: String) {
    let item = ToolItem {
        callback: callback,
        schema: schema,
    };
    tools.insert(name, item);
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

/// entrypoint for the read_file tool
fn tool_write_file(params_str: &String) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
        content: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut how = openat2::OpenHow::new(libc::O_TRUNC | libc::O_WRONLY | libc::O_CREAT, 0o666);
    how.resolve |= openat2::ResolveFlags::IN_ROOT;

    let raw_fd = openat2::openat2(Some(libc::AT_FDCWD), Path::new(&params.path), &how)?;
    let mut file = unsafe { File::from_raw_fd(raw_fd) };

    file.write_all(&params.content.as_bytes())?;
    Ok("".into())
}

/// entrypoint for the list_all_files tool
fn tool_list_all_files(_params_str: &String) -> Result<String, Box<dyn Error>> {
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

fn initialize_tools() -> ToolsCollection {
    let mut tools: ToolsCollection = ToolsCollection::new();

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
        "list_all_files".to_string(),
        tool_list_all_files,
        r#"
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
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    tools
}

/// Sends a prompt to the OpenAI API and prints the AI's response to standard output.
fn post_request_and_print_output(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    let openai_opts = openai::Opts {
        max_tokens: opts.max_tokens,
        model: opts.model.clone(),
    };
    let tools = initialize_tools();
    let response: OpenAIResponse =
        post_request(&prompt, system_prompts, None, &tools, &openai_opts)?;
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
