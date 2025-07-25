# codehawk

Codehawk is a tool that uses AI to analyze and triage GitHub issues.

### Setup

An API key is required. By default, the tool uses OpenRouter and expects the API key at the `~/.openrouter/key` path.

```bash
mkdir -p ~/.openrouter
echo "your-api-key" > ~/.openrouter/key
```

A GitHub token can be used for authenticated requests to the GitHub API,
which can help avoid rate limiting. If you have a personal access token,
store it in `~/.github/token`:

```bash
mkdir -p ~/.github
echo "your-github-token" > ~/.github/token
```


## Usage

### Analyze recent issues and pull requests

To get an AI-generated summary of issues and pull requests created or updated in one or
more repositories within a specific timeframe (defaulting to the last 7 days):

```bash
codehawk analyze [--days <number_of_days>] <owner/repo> [<owner/repo> ...]

--days <number_of_days>: Optional. Specifies the number of past days to fetch issues and pull requests from. Defaults to 7.

<owner/repo>: The GitHub repository path(s) (e.g., rust-lang/rust). You can specify multiple repositories.
```

```
codehawk analyze --days 14 owner1/repoA owner2/repoB
```

Example: Analyze issues and pull requests in `my-org/my-project` from the last 7 days (using default):

```
codehawk analyze my-org/my-project
```

### Prioritize recent issues and pull requests

To get an AI-generated prioritization of issues and pull requests for more repositories within a specific timeframe (defaulting to the last 7 days):

```bash
codehawk prioritize [--days <number_of_days>] <owner/repo> [<owner/repo> ...]

--days <number_of_days>: Optional. Specifies the number of past days to fetch issues and pull requests from. Defaults to 7.

<owner/repo>: The GitHub repository path(s) (e.g., rust-lang/rust). You can specify multiple repositories.
```

```
codehawk prioritize --days 14 owner1/repoA owner2/repoB
```

Example: Prioritize issues and pull requests in `my-org/my-project` from the last 7 days (using default):

```
codehawk prioritize my-org/my-project
```

### Triage a specific issue
To get an AI-generated triage report, potentially including a minimal reproducer, for a specific issue:

```
codehawk triage <owner/repo> <issue_number>
```

### Review a pull request
To get a review for a pull request

```
codehawk review <owner/repo> <pull_request_number>
```

### Chat

To start an interactive chat session with the AI:

```
codehawk chat
```

Within the chat session, you can use the following commands:
*   `\quit`: Exit the chat session.
*   `\clear`: Clear the chat history.
*   `\limit N`: Reduces the history of messages to the N most recent messages. For example, `\limit 10` keeps only the last 10 messages.
*   `\backtrace N`: Removes the last N messages from the history. For example, `\backtrace 5` removes the last 5 messages, effectively going back 5 steps.
*   `\show`: Displays the current chat history, with messages numbered and showing the role (user/assistant) and content.

### List available models

To list the available AI models:

```bash
codehawk models
```

This command fetches the list of models from the default endpoint and displays their IDs, names, pricing, and supported parameters.

The models command honors the `--endpoint` parameter, allowing you to list models from custom API endpoints:

```bash
codehawk --endpoint https://api.openai.com/v1 models
```

When using a custom endpoint, the command will automatically append `/models` to the endpoint URL to fetch the model list.

## Options
You can customize the AI interaction using global options placed before the command:

```
--model <model_name>: Specify a different AI model (Default: google/gemini-2.5-pro).

--max-tokens <number>: Set the maximum number of tokens for the AI's response (Default: 16384).

--endpoint <url>: Override the API endpoint URL. Automatically appends "/chat/completions" for requests.

--parameter <name>=<value>: Set model parameters to control AI behavior. Can be used multiple times.

--no-system-prompts: Skip adding any system prompts to the conversation.
```

### Model Parameters

Use `--parameter` to fine-tune the AI model's behavior. You can specify multiple parameters:

```bash
# Control creativity/randomness (0.0 = deterministic, 1.0 = very creative)
codehawk --parameter temperature=0.7 chat

# Control response diversity (nucleus sampling)
codehawk --parameter top_p=0.9 chat

# Limit token selection to top K choices
codehawk --parameter top_k=40 chat

# Reduce repetitive text
codehawk --parameter frequency_penalty=0.3 chat

# Encourage new topics/ideas
codehawk --parameter presence_penalty=0.6 chat

# Alternative repetition control
codehawk --parameter repetition_penalty=1.1 chat

# Set minimum probability threshold
codehawk --parameter min_p=0.05 chat

# Adaptive sampling
codehawk --parameter top_a=0.2 chat

# Set seed for reproducible outputs
codehawk --parameter seed=12345 chat
```

#### Parameter Examples by Use Case

**Creative Writing** (more random and diverse):
```bash
codehawk --parameter temperature=0.8 --parameter top_p=0.95 --parameter presence_penalty=0.6 prompt "Write a story"
```

**Code Generation** (more focused and deterministic):
```bash
codehawk --parameter temperature=0.2 --parameter top_p=0.9 --parameter frequency_penalty=0.1 prompt "Write a Python function"
```

**Analysis Tasks** (balanced and consistent):
```bash
codehawk --parameter temperature=0.3 --parameter top_p=0.8 analyze --days 7 owner/repo
```

#### Parameter Types
- **Numbers**: `0.7`, `40`, `12345` (automatically detected)
- **Booleans**: `true`, `false`
- **Strings**: `"text"` or text (for non-numeric values)
- **Null**: `null`

**Note**: Parameter availability depends on the model being used. Check your model's documentation for supported parameters.

### Use a raw query

```
codehawk prompt file1 [file2 ....]
```

It allows to pass a raw request to the AI model.

## License
codehawk is licensed under the GNU General Public License v2.0 or later.