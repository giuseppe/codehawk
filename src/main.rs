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
use reqwest::blocking::{Client, Response};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Read;
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

#[derive(Serialize)]
struct OpenRouterRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenRouterResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

/// Sends a POST request to the OpenRouter API with the given prompt and options.
fn open_router_post_request(
    system_prompts: Option<Vec<String>>,
    prompt: &String,
    opts: &Opts,
) -> Result<Response, Box<dyn Error>> {
    let api_key = read_api_key()?;

    let bearer_auth = format!("Bearer {}", &api_key);

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_auth)?);

    let model: String = match &opts.model {
        Some(s) => s.to_string(),
        None => MODEL.to_string(),
    };

    let mut messages: Vec<Message> = vec![];

    if let Some(system_prompts) = system_prompts {
        for system_prompt in system_prompts {
            messages.push(Message {
                role: "system".to_string(),
                content: system_prompt,
            });
        }
    }
    messages.push(Message {
        role: "user".to_string(),
        content: prompt.to_string(),
    });

    let request_body = OpenRouterRequest {
        model: model,
        max_tokens: opts.max_tokens.unwrap_or_else(|| MAX_TOKENS),
        messages: messages,
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
    Ok(response)
}

/// Sends a prompt to the OpenRouter API and prints the AI's response to standard output.
fn post_request_and_print_output(
    system_prompts: Option<Vec<String>>,
    prompt: &String,
    opts: &Opts,
) -> Result<(), Box<dyn Error>> {
    let response: OpenRouterResponse =
        open_router_post_request(system_prompts, &prompt, opts)?.json()?;
    let mut builder = Builder::default();
    for choice in response.choices {
        builder.append(choice.message.content);
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

    post_request_and_print_output(Some(system_prompts), &prompt, opts)
}

/// Fetches a GitHub issue and its comments, then sends them to the AI for triaging.
fn triage_issue(repo: &String, issue_id: u64, opts: &Opts) -> Result<(), Box<dyn Error>> {
    let issue = get_github_issue(repo, issue_id)?;
    let comments = get_github_issue_comments(repo, issue_id)?;
    let prompt = "Provide a triage for the specified issue, show a minimal reproducer for the issue reducing the dependencies needed to run it.".to_string();

    let system_prompts: Vec<String> = vec![
        serde_json::to_string(&issue)?,
        serde_json::to_string(&comments)?,
    ];

    post_request_and_print_output(Some(system_prompts), &prompt, opts)
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
    post_request_and_print_output(Some(system_prompts), &prompt, opts)
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

    post_request_and_print_output(Some(system_prompts), prompt, opts)
}

#[derive(Parser, Debug)]
#[clap(version = env!("CARGO_PKG_VERSION"))]
struct Opts {
    #[clap(short, long)]
    max_tokens: Option<u32>,
    #[clap(long)]
    model: Option<String>,

    #[clap(subcommand)]
    command: Command,

    #[clap()]
    args: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Prioritize the issues and pull requests happened in the last DAYS
    Prioritize {
        #[clap(long)]
        days: Option<u64>,
        repo: Vec<String>,
    },
    /// Analyze the issues and pull requests happened in the last DAYS
    Analyze {
        #[clap(long)]
        days: Option<u64>,
        repo: Vec<String>,
    },
    /// Triage a specific issue
    Triage { repo: String, issue: u64 },

    /// Review a pull request
    Review { repo: String, pr: u64 },

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
        Command::Analyze { days, ref repo } => analyze_repos(&repo, days, &opts)?,
        Command::Prioritize { days, ref repo } => prioritize_repos(&repo, days, &opts)?,
        Command::Triage { ref repo, issue } => triage_issue(&repo, issue, &opts)?,
        Command::Review { ref repo, pr } => review_pull_request(&repo, pr, &opts)?,
        Command::Prompt {
            ref prompt,
            ref files,
        } => prompt_command(&prompt, &files, &opts)?,
    }
    Ok(())
}
