# codehawk

Codehawk is a tool that uses AI (via OpenRouter) to analyze and triage GitHub issues.

### Setup

An OpenRouter API key is required. The OpenRouter API key is expected
at the `~/.openrouter/key` path.

```bash
mkdir -p ~/.openrouter
echo "your-openrouter-api-key" > ~/.openrouter/key
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

To list the AI models available through OpenRouter:

```bash
codehawk models
```

This command fetches the list of models from `https://openrouter.ai/api/v1/models` and displays their IDs.

## Options
You can customize the AI interaction using global options placed before the command (analyze or triage):

```
--model <model_name>: Specify a different model available on OpenRouter (Default: google/gemini-2.5-pro-preview-03-25).

--max-tokens <number>: Set the maximum number of tokens for the AI's response (Default: 16384).
```

### Use a raw query

```
codehawk prompt file1 [file2 ....]
```

It allows to pass a raw request to the AI model.

## License
codehawk is licensed under the GNU General Public License v2.0 or later.