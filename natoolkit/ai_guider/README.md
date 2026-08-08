# AI Guider

AI Guider is the project-scoped conversational manual for Neural Activity
Toolkit. It answers questions about the unified launcher, Activity Tracer, Sleep
State Staging, Activity–Sleep State Aligner, their file conventions, documented
algorithms, and current implementation. It is a support application alongside
`launcher` and `tools`; it does not analyze experiment data.

## Launch and Configuration

Open AI Guider from the unified launcher or run:

```bash
ai-guider
```

Configure the DeepSeek API key before starting the application:

```bash
export DEEPSEEK_API_KEY="your-token"
```

Optional environment variables are:

```text
NATOOLKIT_DEEPSEEK_MODEL     default: deepseek-v4-flash
NATOOLKIT_DEEPSEEK_BASE_URL  default: https://api.deepseek.com
```

The key is read from the process environment. AI Guider does not display, save,
or write it to project files.

## Question Processing

Each question follows a controlled two-stage flow:

1. DeepSeek classifies the question into a fixed intent and up to three known
   project topics using JSON output.
2. Local code validates every returned value. Invalid or uncertain routing fails
   closed.
3. Out-of-scope questions are refused without an answer-generation request.
4. In-scope questions receive the relevant packaged README files.
5. Algorithm, implementation, architecture, and troubleshooting questions also
   receive selected functions or classes from fixed source-code mappings.
6. The answer must cite the supplied source IDs. Unrecognized citations are
   removed; an answer without a valid citation is not shown as verified.

Usage questions use documentation only. The source-level intents are selected
semantically, so a question such as “Why is Wake-to-REM changed?” can retrieve
the staging implementation even when it does not contain the word “code.”

## Approved Knowledge

The primary knowledge is the English README for each application. Source code is
limited to explicit mappings under the installed `natoolkit` package. The model
does not receive a filesystem tool and cannot choose an arbitrary path.

The following are not part of AI Guider context:

- experiment TIFF, EEG/EMG, Note, CSV, JSON, or output files;
- `.git`, virtual environments, caches, and user directories;
- environment variables or API keys;
- historical reference scripts under `tests/ref`.

Conversation history is limited to the most recent eight messages. Questions
are limited to 4,000 characters, and approved context is bounded before it is
sent to the API.

## Supported Scope

AI Guider accepts software usage, troubleshooting, algorithms as implemented,
source behavior, architecture, installation, and toolkit data flow. It refuses
unrelated general knowledge, unrestricted code generation, medical advice, and
questions for which the approved project material is insufficient.

Answers explain what the current project implements. They do not replace expert
quality control of imaging, EEG, EMG, sleep-state, or biological results.

## User Interface

The Qt window provides a Markdown-rendered transcript, multiline prompt, Send,
Stop, and New Conversation controls. The transcript uses Qt's built-in
`QTextDocument.MarkdownDialectGitHub`, so headings, lists, tables, links, block
quotes, and fenced code blocks require no browser component or extra Markdown
package. User questions are escaped and displayed as literal text. Links are
not opened by the transcript.

`Ctrl+Enter` or `Cmd+Enter` sends a question. API work runs in a worker thread;
streamed Markdown is provisional and is replaced by the locally
citation-validated final answer when generation completes.
