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

use reqwest::header::HeaderMap;

use reqwest::header::HeaderValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use std::error::Error;

use chrono::{Utc, SecondsFormat, Days};
use reqwest::blocking::{Client, Response};
use reqwest::header::USER_AGENT;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Issue {
    pub url: String,
    pub repository_url: String,
    pub labels_url: String,
    pub comments_url: String,
    pub events_url: String,
    pub html_url: String,
    pub id: u64,
    pub node_id: String,
    pub number: u64,
    pub title: String,
    pub user: User,

    pub labels: Vec<Type>,
    pub state: Option<String>,
    pub locked: bool,
    pub assignee: Option<User>,
    pub assignees: Vec<User>,
    pub milestone: Option<Value>,
    pub comments: Option<u64>,
    pub created_at: String,
    pub updated_at: String,
    pub closed_at: Option<String>,

    pub author_association: String,
    // Renamed because 'type' is a Rust keyword
    #[serde(rename = "type")]
    pub issue_type: Option<Type>,
    pub sub_issues_summary: SubIssuesSummary,
    pub active_lock_reason: Option<String>,
    pub body: Option<String>,
    pub closed_by: Option<Value>,
    pub reactions: Option<Reactions>,
    pub timeline_url: Option<String>,
    pub performed_via_github_app: Option<String>,
    pub state_reason: Option<String>,

    // Fields specific to Pull Requests (optional in the general issue list)
    #[serde(default)] // Treat missing field as None
    pub draft: Option<bool>,
    #[serde(default)] // Treat missing field as None
    pub pull_request: Option<PullRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Comment {
    pub url: String,
    pub html_url: String,
    pub issue_url: String,
    pub id: u64,
    pub node_id: String,
    pub user: User,
    pub created_at: String,
    pub updated_at: String,
    pub author_association: String,
    pub body: String,
    pub reactions: Option<Reactions>,
    pub performed_via_github_app: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Type {
    pub id: u64,
    pub node_id: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub color: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub is_enabled: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct User {
    pub login: String,
    pub id: u64,
    pub node_id: String,
    pub avatar_url: String,
    pub gravatar_id: String,
    pub url: String,
    pub html_url: String,
    pub followers_url: String,
    pub following_url: String,
    pub gists_url: String,
    pub starred_url: String,
    pub subscriptions_url: String,
    pub organizations_url: String,
    pub repos_url: String,
    pub events_url: String,
    pub received_events_url: String,
    // Renamed because 'type' is a Rust keyword
    #[serde(rename = "type")]
    pub user_type: String,
    pub user_view_type: String,
    pub site_admin: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SubIssuesSummary {
    pub total: u64,
    pub completed: u64,
    pub percent_completed: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Reactions {
    pub url: String,
    pub total_count: u64,
    #[serde(rename = "+1")]
    pub plus1: u64,
    #[serde(rename = "-1")]
    pub minus1: u64,
    pub laugh: u64,
    pub hooray: u64,
    pub confused: u64,
    pub heart: u64,
    pub rocket: u64,
    pub eyes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PullRequest {
    pub url: String,
    pub html_url: String,
    pub diff_url: String,
    pub patch_url: String,
    pub merged_at: Option<String>,
}

// Top-level
pub type Issues = Vec<Issue>;
pub type Comments = Vec<Comment>;

fn make_github_api_request(path: &String, query: Option<&String>) -> Result<Response, Box<dyn Error>> {
    let url = match query {
        Some(query) => format!("https://api.github.com/{}?{}", path, query),
        None => format!("https://api.github.com/{}", path),
    };

    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("codehawk"));

    let client = Client::new();
    let response = client
        .get(url)
        .headers(headers)
        .send()?
        .error_for_status()?;

    Ok(response)
}

pub fn get_all_github_issues(
    repo: &String,
    days: u64,
) -> Result<Issues, Box<dyn Error>> {
    let now = Utc::now();
    let since = now.checked_sub_days(Days::new(days)).unwrap();
    let query = format!("since={}", since.to_rfc3339_opts(SecondsFormat::Secs, true));

    let path = format!("repos/{}/issues", repo);
    let response = make_github_api_request(&path, Some(&query))?;
    let text = response.text()?;
    let issues = serde_json::from_str::<Issues>(&text)?;

    Ok(issues)
}

pub fn get_github_issue(repo: &String, issue: i64) -> Result<Issue, Box<dyn Error>> {
    let path = format!("repos/{}/issues/{}", repo, issue);
    let response = make_github_api_request(&path, None)?;

    let text = response.text()?;
    let issue = serde_json::from_str::<Issue>(&text)?;
    Ok(issue)
}

pub fn get_github_issue_comments(repo: &String, issue: i64) -> Result<Comments, Box<dyn Error>> {
    let path = format!("repos/{}/issues/{}/comments", repo, issue);
    let response = make_github_api_request(&path, None)?;

    let text = response.text()?;
    let comments = serde_json::from_str::<Comments>(&text)?;

    Ok(comments)
}
