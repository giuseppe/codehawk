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

use chrono::{Days, SecondsFormat, Utc};
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct PullRequest {
    pub url: String,
    pub html_url: String,
    pub diff_url: String,
    pub patch_url: String,
    pub merged_at: Option<String>,

    pub id: Option<u64>,
    pub node_id: Option<String>,
    pub issue_url: Option<String>,
    pub number: Option<u32>,
    pub state: Option<String>,
    pub locked: Option<bool>,
    pub title: Option<String>,
    pub user: Option<User>,
    pub body: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub closed_at: Option<String>,
    pub merge_commit_sha: Option<String>,
    pub assignee: Option<User>,
    pub assignees: Option<Vec<User>>,
    pub requested_reviewers: Option<Vec<User>>,
    pub requested_teams: Option<Vec<Team>>,
    pub labels: Option<Vec<Label>>,
    pub milestone: Option<Milestone>,
    pub draft: Option<bool>,
    pub commits_url: Option<String>,
    pub review_comments_url: Option<String>,
    pub review_comment_url: Option<String>,
    pub comments_url: Option<String>,
    pub statuses_url: Option<String>,
    pub head: Option<RepoRef>,
    pub base: Option<RepoRef>,
    #[serde(rename = "_links")]
    pub links: Option<Links>,
    pub author_association: Option<String>,
    pub auto_merge: Option<bool>,
    pub active_lock_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Team {
    pub id: Option<String>,
    pub node_id: Option<String>,
    pub url: Option<String>,
    pub html_url: Option<String>,
    pub name: Option<String>,
    pub slug: Option<String>,
    pub description: Option<String>,
    pub privacy: Option<String>,
    pub notication_setting: Option<String>,
    pub permission: Option<String>,
    pub members_url: Option<String>,
    pub repositories_url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Label {
    pub id: u64,
    pub node_id: String,
    pub url: String,
    pub name: String,
    pub color: String,
    pub default: bool,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Milestone {
    pub url: String,
    pub html_url: String,
    pub labels_url: String,
    pub id: u64,
    pub node_id: String,
    pub number: u32,
    pub state: String,
    pub title: String,
    pub description: Option<String>,
    pub creator: User,
    pub open_issues: u32,
    pub closed_issues: u32,
    pub created_at: String,
    pub updated_at: String,
    pub closed_at: Option<String>,
    pub due_on: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RepoRef {
    pub label: Option<String>,
    pub ref_field: Option<String>,
    #[serde(rename = "ref")]
    pub sha: Option<String>,
    pub user: Option<User>,
    pub repo: Option<Repository>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Repository {
    pub id: u64,
    pub node_id: Option<String>,
    pub name: Option<String>,
    pub full_name: Option<String>,
    pub private: bool,
    pub owner: User,
    pub html_url: Option<String>,
    pub description: Option<String>,
    pub fork: bool,
    pub url: Option<String>,
    pub forks_url: Option<String>,
    pub keys_url: Option<String>,
    pub collaborators_url: Option<String>,
    pub teams_url: Option<String>,
    pub hooks_url: Option<String>,
    pub issue_events_url: Option<String>,
    pub events_url: Option<String>,
    pub assignees_url: Option<String>,
    pub branches_url: Option<String>,
    pub tags_url: Option<String>,
    pub blobs_url: Option<String>,
    pub git_tags_url: Option<String>,
    pub git_refs_url: Option<String>,
    pub trees_url: Option<String>,
    pub statuses_url: Option<String>,
    pub languages_url: Option<String>,
    pub stargazers_url: Option<String>,
    pub contributors_url: Option<String>,
    pub subscribers_url: Option<String>,
    pub subscription_url: Option<String>,
    pub commits_url: Option<String>,
    pub git_commits_url: Option<String>,
    pub comments_url: Option<String>,
    pub issue_comment_url: Option<String>,
    pub contents_url: Option<String>,
    pub compare_url: Option<String>,
    pub merges_url: Option<String>,
    pub archive_url: Option<String>,
    pub downloads_url: Option<String>,
    pub issues_url: Option<String>,
    pub pulls_url: Option<String>,
    pub milestones_url: Option<String>,
    pub notifications_url: Option<String>,
    pub labels_url: Option<String>,
    pub releases_url: Option<String>,
    pub deployments_url: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub pushed_at: Option<String>,
    pub git_url: Option<String>,
    pub ssh_url: Option<String>,
    pub clone_url: Option<String>,
    pub svn_url: Option<String>,
    pub homepage: Option<String>,
    pub size: u64,
    pub stargazers_count: u64,
    pub watchers_count: u64,
    pub language: Option<String>,
    pub has_issues: bool,
    pub has_projects: bool,
    pub has_downloads: bool,
    pub has_wiki: bool,
    pub has_pages: bool,
    pub has_discussions: bool,
    pub forks_count: u64,
    pub mirror_url: Option<String>,
    pub archived: bool,
    pub disabled: bool,
    pub open_issues_count: u64,
    pub license: Option<License>,
    pub allow_forking: bool,
    pub is_template: bool,
    pub web_commit_signoff_required: bool,
    pub topics: Vec<String>,
    pub visibility: Option<String>,
    pub forks: u64,
    pub open_issues: u64,
    pub watchers: u64,
    pub default_branch: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct License {
    pub key: Option<String>,
    pub name: Option<String>,
    pub pdx_id: Option<String>,
    pub url: Option<String>,
    pub node_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Links {
    #[serde(rename = "self")]
    pub self_link: Link,
    pub html: Link,
    pub issue: Link,
    pub comments: Link,
    pub review_comments: Link,
    pub review_comment: Link,
    pub commits: Link,
    pub statuses: Link,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Link {
    pub href: String,
}

// Top-level
pub type Issues = Vec<Issue>;
pub type PullRequests = Vec<PullRequest>;
pub type Comments = Vec<Comment>;

fn make_github_api_request(
    path: &String,
    query: Option<&String>,
) -> Result<Response, Box<dyn Error>> {
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

pub fn get_github_issues(repo: &String, days: u64) -> Result<Issues, Box<dyn Error>> {
    let now = Utc::now();
    let since = now.checked_sub_days(Days::new(days)).unwrap();
    let query = format!("since={}", since.to_rfc3339_opts(SecondsFormat::Secs, true));

    let path = format!("repos/{}/issues", repo);
    let response = make_github_api_request(&path, Some(&query))?;
    let text = response.text()?;
    let issues = serde_json::from_str::<Issues>(&text)?;

    Ok(issues)
}

pub fn get_github_pull_requests(repo: &String, days: u64) -> Result<PullRequests, Box<dyn Error>> {
    let now = Utc::now();
    let since = now.checked_sub_days(Days::new(days)).unwrap();
    let query = format!("since={}", since.to_rfc3339_opts(SecondsFormat::Secs, true));

    let path = format!("repos/{}/pulls", repo);
    let response = make_github_api_request(&path, Some(&query))?;
    let text = response.text()?;
    let pull_requests = serde_json::from_str::<PullRequests>(&text)?;

    Ok(pull_requests)
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
