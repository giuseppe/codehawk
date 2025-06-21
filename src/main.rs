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
use console::{Term, style};
use env_logger::Env;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::{debug, trace, warn};
use pathrs::{Root, flags::OpenFlags};
use prettytable::{Cell, Row, Table, format};
use rustyline::DefaultEditor;
use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::fs::Permissions;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use github::{
    Issues, PullRequests, get_github_issue, get_github_issue_comments, get_github_issues,
    get_github_pull_request, get_github_pull_request_patch, get_github_pull_requests,
};
use openai::{
    Message, OpenAIResponse, ProgressInfo, ResponseMode, StatusUpdate, ToolCallback, ToolItem,
    ToolsCollection, list_models, make_message, post_request, post_request_with_mode,
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
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        path: String,
    }

    #[derive(Serialize)]
    struct ReadFileResult {
        content: Option<String>,
        error: Option<String>,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Reading file: {}", params.path);

    let root = Root::open(".")?;
    let path = PathBuf::from(&params.path);
    let file = root.open_subpath(path, OpenFlags::O_RDONLY);

    let result = match file {
        Ok(mut file) => {
            let mut contents = String::new();
            match file.read_to_string(&mut contents) {
                Ok(_) => ReadFileResult {
                    content: Some(contents),
                    error: None,
                },
                Err(e) => ReadFileResult {
                    content: None,
                    error: Some(format!("Failed to read file: {}", e)),
                },
            }
        }
        Err(e) => ReadFileResult {
            content: None,
            error: Some(format!("File not found or cannot be opened: {}", e)),
        },
    };

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
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
                OpenFlags::O_WRONLY | OpenFlags::O_CREAT,
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

/// Helper function to wrap text to fit within a given width
fn wrap_text_to_width(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    for line in text.lines() {
        if line.len() <= width {
            lines.push(line.to_string());
        } else {
            // Split long lines at word boundaries when possible
            let mut current_line = String::new();
            for word in line.split_whitespace() {
                if current_line.is_empty() {
                    if word.len() > width {
                        // Word is too long, split it
                        let mut remaining = word;
                        while remaining.len() > width {
                            lines.push(remaining[..width].to_string());
                            remaining = &remaining[width..];
                        }
                        if !remaining.is_empty() {
                            current_line = remaining.to_string();
                        }
                    } else {
                        current_line = word.to_string();
                    }
                } else if current_line.len() + 1 + word.len() <= width {
                    current_line.push(' ');
                    current_line.push_str(word);
                } else {
                    lines.push(current_line);
                    if word.len() > width {
                        // Word is too long, split it
                        let mut remaining = word;
                        while remaining.len() > width {
                            lines.push(remaining[..width].to_string());
                            remaining = &remaining[width..];
                        }
                        current_line = remaining.to_string();
                    } else {
                        current_line = word.to_string();
                    }
                }
            }
            if !current_line.is_empty() {
                lines.push(current_line);
            }
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

/// State for tracking streaming box updates
struct StreamingBoxState {
    title: String,
    content: String,
    color: Option<String>,
    content_width: usize,
    max_content_lines: usize,
    lines_printed: usize,
}

impl StreamingBoxState {
    fn new(title: String, color: Option<&str>) -> Self {
        let term = Term::stdout();
        let term_width = term.size().1 as usize;
        let box_width = if term_width < 20 {
            20
        } else {
            term_width.saturating_sub(2)
        };
        let content_width = box_width.saturating_sub(4);

        Self {
            title,
            content: String::new(),
            color: color.map(|c| c.to_string()),
            content_width,
            max_content_lines: 15,
            lines_printed: 0,
        }
    }

    fn append_content(&mut self, new_content: &str) {
        self.content.push_str(new_content);
        print!("{}", new_content);
        use std::io::{Write, stdout};
        let _ = stdout().flush();
    }

    fn draw_box(&self, _box_width: usize) -> usize {
        let mut line_count = 0;

        let title_lines = wrap_text_to_width(&self.title, self.content_width);
        for line in title_lines {
            println!("📦 {}", line);
            line_count += 1;
        }
        line_count += 1;

        if self.content.is_empty() {
            let building_text = "(building...)";
            println!("   {}", building_text);
            line_count += 1;
        } else {
            let lines = self.content.lines().collect::<Vec<_>>();

            for line in lines.iter().take(self.max_content_lines) {
                let wrapped_lines = wrap_text_to_width(line, self.content_width);
                for wrapped_line in wrapped_lines {
                    let renderer = BoxRenderer::new();
                    let display_line =
                        renderer.apply_color_style(&wrapped_line, self.color.as_deref());

                    println!("   {}", display_line);
                    line_count += 1;
                }
            }

            if lines.len() > self.max_content_lines {
                let truncate_text = "... (more content follows)";
                println!("   {}", truncate_text);
                line_count += 1;
            }
        }
        line_count += 1;

        line_count
    }

    fn start(&mut self) {
        println!();

        let term_width = Term::stdout().size().1 as usize;
        let box_width = if term_width < 20 {
            20
        } else {
            term_width.saturating_sub(2)
        };
        self.lines_printed = self.draw_box(box_width);
    }
}

/// Box renderer for consistent box display across the application
struct BoxRenderer {
    content_width: usize,
}

impl BoxRenderer {
    fn new() -> Self {
        let term = Term::stdout();
        let term_width = term.size().1 as usize;
        let box_width = if term_width < 20 {
            20
        } else {
            term_width.saturating_sub(2)
        };
        let content_width = box_width.saturating_sub(4);

        Self { content_width }
    }

    fn render_title(&self, title: &str) {
        let title_lines = wrap_text_to_width(title, self.content_width);
        for line in title_lines {
            println!("📦 {}", line);
        }
    }

    fn apply_color_style(&self, text: &str, color: Option<&str>) -> String {
        if let Some(c) = color {
            let styled = match c {
                "green" => style(text).green(),
                "red" => style(text).red(),
                "yellow" => style(text).yellow(),
                "cyan" => style(text).cyan(),
                _ => style(text),
            };
            styled.to_string()
        } else {
            text.to_string()
        }
    }

    fn render_content_lines(
        &self,
        content: &str,
        color: Option<&str>,
        max_lines: usize,
        truncation_msg: &str,
        empty_msg: &str,
    ) {
        if content.is_empty() {
            println!("   {}", empty_msg);
            return;
        }

        let lines = content.lines().collect::<Vec<_>>();

        for line in lines.iter().take(max_lines) {
            let wrapped_lines = wrap_text_to_width(line, self.content_width);
            for wrapped_line in wrapped_lines {
                let display_line = self.apply_color_style(&wrapped_line, color);
                println!("   {}", display_line);
            }
        }

        if lines.len() > max_lines {
            println!("   {}", truncation_msg);
        }
    }

    fn flush_output(&self) {
        use std::io::{Write, stdout};
        let _ = stdout().flush();
    }
}

/// Function to display streaming content in a box
fn display_streaming_box(title: &str, content: &str, color: Option<&str>) {
    let renderer = BoxRenderer::new();

    println!();

    renderer.render_title(title);
    renderer.render_content_lines(
        content,
        color,
        15,
        "... (more content follows)",
        "(building...)",
    );
    renderer.flush_output();
}

/// Generic function to display content in a box
fn display_content_in_box(title: &str, sections: Vec<(&str, &str, Option<&str>)>) {
    let renderer = BoxRenderer::new();

    renderer.render_title(title);

    // Content sections
    for (_i, (section_title, content, color_opt)) in sections.iter().enumerate() {
        // Section title
        if !section_title.is_empty() {
            let styled_title = if let Some(color) = color_opt {
                match *color {
                    "green" => style(section_title).bold().green().to_string(),
                    "red" => style(section_title).bold().red().to_string(),
                    _ => section_title.to_string(),
                }
            } else {
                section_title.to_string()
            };

            println!("🔸 {}", styled_title);
        }

        // Section content
        let max_lines = if section_title.to_uppercase().contains("STDERR") {
            10
        } else {
            20
        };

        renderer.render_content_lines(
            content,
            None, // sections don't use content coloring
            max_lines,
            "... (output truncated)",
            "(no output)",
        );
    }
}

/// Helper function to display command output in a box
fn display_command_output_box(
    command: &str,
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
    success: bool,
) {
    let title = if success {
        format!("✅ Command: {} (exit: {})", command, exit_code.unwrap_or(0))
    } else {
        format!(
            "❌ Command: {} (exit: {})",
            command,
            exit_code.unwrap_or(-1)
        )
    };

    let mut sections = Vec::new();

    if !stdout.is_empty() {
        sections.push(("STDOUT:", stdout, Some("green")));
    }

    if !stderr.is_empty() {
        sections.push(("STDERR:", stderr, Some("red")));
    }

    if stdout.is_empty() && stderr.is_empty() {
        sections.push(("", "", None));
    }

    display_content_in_box(&title, sections);
}

/// entrypoint for the run_command tool
fn tool_run_command(params_str: &String) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        command: String,
        args: Option<Vec<String>>,
    }

    #[derive(Serialize)]
    struct CommandResult {
        stdout: String,
        stderr: String,
        exit_code: Option<i32>,
        success: bool,
    }

    debug!("run_command received params: {}", params_str);

    // Try normal parsing first, fallback to manual parsing if it fails
    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = if params.command.contains(' ') {
        // If command contains spaces
        let mut c = Command::new("sh");
        c.arg("-c").arg(&params.command);
        c
    } else {
        // Normal command execution
        let mut c = Command::new(&params.command);
        if let Some(ref args) = params.args {
            c.args(args);
        }
        c
    };

    trace!("Executing command: {:?}", cmd);

    let result = match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8(output.stdout)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let stderr = String::from_utf8(output.stderr)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let exit_code = output.status.code();
            let success = output.status.success();

            if success {
                debug!("Successfully run command {}", params.command);
            } else {
                debug!(
                    "Command {} failed with exit code {:?}",
                    params.command, exit_code
                );
            }

            CommandResult {
                stdout,
                stderr,
                exit_code,
                success,
            }
        }
        Err(e) => {
            debug!("Failed to execute command {}: {}", params.command, e);
            CommandResult {
                stdout: String::new(),
                stderr: format!("Failed to execute command: {}", e),
                exit_code: None,
                success: false,
            }
        }
    };

    // Display the command output in a box
    let display_command = if params.command.contains(' ') {
        params.command.clone()
    } else {
        let mut full_cmd = params.command.clone();
        if let Some(args) = &params.args {
            if !args.is_empty() {
                full_cmd.push_str(" ");
                // Handle potential JSON string in args for display only
                let args_display: Vec<String> = args
                    .iter()
                    .map(|arg| {
                        if arg.starts_with('[') && arg.ends_with(']') {
                            // This might be a JSON array string, try to parse for display
                            if let Ok(parsed_args) = serde_json::from_str::<Vec<String>>(arg) {
                                parsed_args.join(" ")
                            } else {
                                arg.clone()
                            }
                        } else {
                            arg.clone()
                        }
                    })
                    .collect();
                full_cmd.push_str(&args_display.join(" "));
            }
        }
        full_cmd
    };

    display_command_output_box(
        &display_command,
        &result.stdout,
        &result.stderr,
        result.exit_code,
        result.success,
    );

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
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
                "description": "Get the content of a file stored in the repository. Returns JSON with 'content' field containing file content on success, or 'error' field with error message on failure.",
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
                "description": "Run a command and return results in JSON format. Returns {\"stdout\": string, \"stderr\": string, \"exit_code\": number|null, \"success\": boolean}. Commands do not fail on non-zero exit codes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute, it must be the name of the executable only, e.g. /usr/bin/ls"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Additional arguments to pass to the command after the executable path, e.g. [\"-l\", \"-a\"]"
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

fn add_predefined_system_prompts(messages: &mut Vec<Message>) {
    let predefined_prompts = vec![
        "You are codehawk, an AI assistant that helps with software development and repository analysis.",
        "Always provide accurate, helpful responses and use available tools when appropriate for file operations or code analysis.",
        "When working with code, maintain best practices and consider security implications.",
        "Use the available tools as much as possible to find a solution.  Iterate until the problem is solved, Terminate only when you are sure to have found the solution, if a tool fails, analyze the failure, fix the issue and call again the tool.  Never ask to run commands manually, just do it.",
    ];

    for prompt in predefined_prompts {
        messages.push(make_message("system", prompt.to_string()));
    }
}

fn initialize_chat_messages(tools: &ToolsCollection, opts: &Opts) -> Vec<Message> {
    let mut messages: Vec<Message> = vec![];

    // Add predefined system prompts that are always loaded
    add_predefined_system_prompts(&mut messages);

    // Add conservative tool usage system prompt if tools are available but no explicit choice
    if !tools.is_empty() && opts.tool_choice.is_none() {
        add_conservative_prompt(&mut messages);
    }

    debug!("Initialized chat with {} system messages", messages.len());
    messages
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
        api_key: opts.api_key.clone(),
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
fn chat_command(
    opts: &Opts,
    global_multi_progress: Arc<MultiProgress>,
) -> Result<(), Box<dyn Error>> {
    debug!("Executing chat command");

    let mut rl = DefaultEditor::new()?;
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

    let mut messages = initialize_chat_messages(&tools, opts);

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
        api_key: opts.api_key.clone(),
    };

    loop {
        let line = rl.readline("> ")?.trim().to_string();
        if line.is_empty() {
            continue;
        }
        rl.add_history_entry(line.as_str())?;
        debug!("User input: '{}' (length: {})", line, line.len());

        if line == "\\quit" {
            return Ok(());
        }
        if line == "\\clear" {
            messages = initialize_chat_messages(&tools, opts);
            println!("Chat history cleared and system prompts restored.");
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
                        messages = initialize_chat_messages(&tools, opts);
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
                        messages = initialize_chat_messages(&tools, opts);
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

        // Store the current message count before adding user message
        let message_count_before = messages.len();
        messages.push(make_message("user", line));
        debug!(
            "Added user message. Message count: {} -> {}",
            message_count_before,
            messages.len()
        );

        fn format_tool_arguments(args_json: &str) -> String {
            // Just escape special characters, no truncation needed with boxes
            args_json
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t")
        }

        use std::sync::atomic::{AtomicBool, Ordering};

        let multi_progress = global_multi_progress.clone();

        let status_pb = multi_progress.add(ProgressBar::new_spinner());
        status_pb.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")?
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );

        let stream_pb = multi_progress.add(ProgressBar::new_spinner());
        stream_pb.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")?.tick_strings(&[
                "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂",
            ]),
        );

        let is_first_chunk = Arc::new(AtomicBool::new(true));
        let tool_active = Arc::new(AtomicBool::new(false));
        let token_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let streaming_box_state =
            Arc::new(std::sync::Mutex::new(Option::<StreamingBoxState>::None));
        let message_box_state = Arc::new(std::sync::Mutex::new(Option::<StreamingBoxState>::None));

        let is_first_chunk_clone = is_first_chunk.clone();
        let status_pb_stream_clone = status_pb.clone();
        let stream_pb_stream_clone = stream_pb.clone();
        let status_pb_progress_clone = status_pb.clone();
        let stream_pb_progress_clone = stream_pb.clone();
        let streaming_box_state_clone = streaming_box_state.clone();
        let message_box_state_clone = message_box_state.clone();
        let tool_active_stream_clone = tool_active.clone();

        let mode = ResponseMode::Streaming {
            stream_handler: Box::new(move |chunk: &str| {
                if chunk.is_empty() {
                    // Finalize and clear the message box when streaming is complete
                    if let Ok(mut state_guard) = message_box_state_clone.lock() {
                        if let Some(_box_state) = state_guard.as_mut() {
                            // Add a final newline to complete the box display
                            println!();
                        }
                        // Clear the state for next message
                        *state_guard = None;
                    }
                    return Ok(());
                }

                if is_first_chunk_clone.load(Ordering::Relaxed) {
                    // Clear all progress bars when streaming starts
                    status_pb_stream_clone.finish_and_clear();
                    stream_pb_stream_clone.finish_and_clear();
                    is_first_chunk_clone.store(false, Ordering::Relaxed);
                }

                // Don't process message content while tools are active
                if tool_active_stream_clone.load(Ordering::Relaxed) {
                    return Ok(());
                }

                // Handle message content streaming in a box
                if let Ok(mut state_guard) = message_box_state_clone.lock() {
                    match state_guard.as_mut() {
                        Some(box_state) => {
                            // Update existing message box with new content
                            box_state.append_content(chunk);
                        }
                        None => {
                            // Create new message streaming box
                            let title = "💬 Assistant Response".to_string();
                            let mut box_state = StreamingBoxState::new(title, None);
                            box_state.start();
                            box_state.append_content(chunk);
                            *state_guard = Some(box_state);
                        }
                    }
                }

                Ok(())
            }),
            progress_handler: Box::new(move |progress_info: &ProgressInfo| {
                let elapsed_secs = progress_info.elapsed_ms as f64 / 1000.0;

                match &progress_info.status {
                    StatusUpdate::ToolAccumulating { name, arguments } => {
                        // Clear progress bars and show streaming box
                        status_pb_progress_clone.finish_and_clear();
                        stream_pb_progress_clone.finish_and_clear();

                        if let Ok(mut state_guard) = streaming_box_state_clone.lock() {
                            match state_guard.as_mut() {
                                Some(box_state) => {
                                    // Update existing box with new content and timing
                                    box_state.title =
                                        format!("📝 Building {} call ({:.1}s)", name, elapsed_secs);
                                    let current_len = box_state.content.len();
                                    if arguments.len() > current_len {
                                        let new_content = &arguments[current_len..];
                                        box_state.append_content(new_content);
                                    }
                                }
                                None => {
                                    // Create new streaming box
                                    let title =
                                        format!("📝 Building {} call ({:.1}s)", name, elapsed_secs);
                                    let mut box_state = StreamingBoxState::new(title, Some("cyan"));
                                    box_state.start();
                                    if !arguments.is_empty() {
                                        box_state.append_content(arguments);
                                    }
                                    *state_guard = Some(box_state);
                                }
                            }
                        }
                    }
                    StatusUpdate::ToolStart { name, arguments } => {
                        // Set tool active to prevent streaming content display
                        tool_active.store(true, Ordering::Relaxed);

                        // Clear the streaming box states and show final arguments
                        if let Ok(mut state_guard) = streaming_box_state_clone.lock() {
                            *state_guard = None;
                        }
                        // Don't clear message_box_state - preserve it for continued streaming

                        // Add spacing between accumulation and execution boxes
                        println!();

                        // Show final arguments box before execution
                        let title = format!("🔧 Starting {} execution", name);
                        display_streaming_box(&title, arguments, Some("yellow"));

                        // Update status bar to show tool execution
                        status_pb_progress_clone.set_style(
                            ProgressStyle::with_template("🔧 {spinner:.yellow} {msg}")?
                                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
                        );
                        let formatted_args = format_tool_arguments(arguments);
                        status_pb_progress_clone.set_message(format!(
                            "Running {} tool ({}) ... {:.1}s",
                            name, formatted_args, elapsed_secs
                        ));
                        status_pb_progress_clone.enable_steady_tick(Duration::from_millis(100));
                    }
                    StatusUpdate::ToolExecuting { name, arguments } => {
                        // Update status bar to show tool execution in progress
                        status_pb_progress_clone.set_style(
                            ProgressStyle::with_template("⚡ {spinner:.red} {msg}")?
                                .tick_strings(&["●", "○", "◐", "◑", "◒", "◓"]),
                        );
                        let formatted_args = format_tool_arguments(arguments);
                        status_pb_progress_clone.set_message(format!(
                            "Executing {} ({}) ... {:.1}s",
                            name, formatted_args, elapsed_secs
                        ));
                    }
                    StatusUpdate::ToolComplete {
                        name,
                        arguments,
                        duration_ms,
                    } => {
                        let duration_secs = *duration_ms as f64 / 1000.0;
                        // Clear tool active flag to allow streaming content display again
                        tool_active.store(false, Ordering::Relaxed);
                        // Clear status bar
                        status_pb_progress_clone.finish_and_clear();

                        // Don't clear message_box_state - preserve it for continued streaming

                        // Show completion message in a box
                        let formatted_args = format_tool_arguments(arguments);
                        let title = format!("✅ {} completed ({:.1}s)", name, duration_secs);
                        display_streaming_box(
                            &title,
                            &format!("Arguments: {}", formatted_args),
                            Some("green"),
                        );

                        // Add some spacing after the box
                        println!();
                    }
                    StatusUpdate::Continuing => {
                        // Update status bar to show processing
                        status_pb_progress_clone.set_style(
                            ProgressStyle::with_template("🔄 {spinner:.cyan} {msg}")?
                                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
                        );
                        status_pb_progress_clone
                            .set_message(format!("Processing... {:.1}s", elapsed_secs));
                        status_pb_progress_clone.enable_steady_tick(Duration::from_millis(100));
                    }
                    StatusUpdate::StreamProcessing {
                        bytes_read,
                        chunks_processed,
                    } => {
                        // Update stream progress bar with streaming data info
                        stream_pb_progress_clone.set_style(
                            ProgressStyle::with_template("📡 {spinner:.blue} {msg}")?.tick_strings(
                                &[
                                    "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄",
                                    "▃", "▂",
                                ],
                            ),
                        );
                        stream_pb_progress_clone.set_message(format!(
                            "Streaming... {} bytes, {} chunks ({:.1}s)",
                            bytes_read, chunks_processed, elapsed_secs
                        ));
                        stream_pb_progress_clone.enable_steady_tick(Duration::from_millis(100));
                    }
                    StatusUpdate::Complete { usage } => {
                        println!();

                        if let Some(usage) = usage {
                            let mut parts = Vec::new();

                            if let Some(input_tokens) = usage.prompt_tokens {
                                parts.push(format!("Input: {} tokens", input_tokens));
                            }

                            if let Some(output_tokens) = usage.completion_tokens {
                                parts.push(format!("Output: {} tokens", output_tokens));
                            }

                            if let Some(total_tokens) = usage.total_tokens {
                                parts.push(format!("Total: {}", total_tokens));
                            }

                            if !parts.is_empty() {
                                println!(
                                    "🎯 Complete! {} ({:.1}s)",
                                    parts.join(" → "),
                                    elapsed_secs
                                );
                            } else {
                                println!("🎯 Complete! ({:.1}s)", elapsed_secs);
                            }
                        } else {
                            println!("🎯 Complete! ({:.1}s)", elapsed_secs);
                        }
                    }
                }

                // Reset first chunk flag for next stream
                is_first_chunk.store(true, Ordering::Relaxed);
                Ok(())
            }),
        };

        // Start the thinking progress bar with timing
        let start_time = std::time::Instant::now();
        status_pb.set_message("Thinking...");
        status_pb.enable_steady_tick(Duration::from_millis(100));

        // Spawn a thread to update thinking progress with elapsed time and tokens
        let thinking_pb = status_pb.clone();
        let thinking_start = start_time.clone();
        let thinking_token_count = token_count.clone();
        let thinking_handle = std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            while !thinking_pb.is_finished() {
                let elapsed = thinking_start.elapsed().as_secs_f64();
                let tokens = thinking_token_count.load(Ordering::Relaxed);
                if tokens > 0 {
                    thinking_pb
                        .set_message(format!("Thinking... {:.1}s • {} tokens", elapsed, tokens));
                } else {
                    thinking_pb.set_message(format!("Thinking... {:.1}s", elapsed));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        });

        let response = post_request_with_mode(messages, &tools, &openai_opts, mode)?;

        // Stop the thinking update thread
        status_pb.finish_and_clear();
        stream_pb.finish_and_clear();
        let _ = thinking_handle.join();

        // Clear all progress bars when done
        multi_progress.clear()?;

        // The actual response will be displayed here by the stream handler
        debug!(
            "Response history contains: {} messages",
            response.history.len()
        );
        messages = response.history;
        debug!("After updating history: {} messages", messages.len());
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
    #[clap(long)]
    /// Override the file path to read API key from
    api_key: Option<String>,

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

    // Set up indicatif log bridge to prevent logging interference with progress bars
    let logger = env_logger::Builder::from_env(env).build();
    let global_multi_progress = Arc::new(MultiProgress::new());
    LogWrapper::new((*global_multi_progress).clone(), logger).try_init()?;

    // Parse command line arguments
    let mut opts = Opts::parse();
    debug!("Command line options parsed");

    // Reset the model to use if an endpoint was provided
    if opts.model.is_none() && opts.endpoint.is_some() {
        opts.model = Some("".to_string());
    }

    // Execute the chosen command
    let result = match &opts.command {
        CliCommand::Analyze { days, repo } => analyze_repos(&repo, *days, &opts),
        CliCommand::Prioritize { days, repo } => prioritize_repos(&repo, *days, &opts),
        CliCommand::Triage { repo, issue } => triage_issue(&repo, *issue, &opts),
        CliCommand::Review { repo, pr } => review_pull_request(&repo, *pr, &opts),
        CliCommand::Prompt { prompt, files } => prompt_command(&prompt, &files, &opts),
        CliCommand::Chat {} => chat_command(&opts, global_multi_progress.clone()),
        CliCommand::Models {} => list_models_command(&opts),
    };

    result
}
