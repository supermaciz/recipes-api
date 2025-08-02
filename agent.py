import asyncio
import os
from typing import Any

import dotenv
import github
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv("/home/mcizo/tmp/pr_review2.env")
repository = os.getenv("REPOSITORY")
pr_nb = os.getenv("PR_NUMBER")
repo_url = f"https://github.com/{repository}.git"

auth = github.Auth.Token(os.getenv("GITHUB_TOKEN"))
git = github.Github(auth=auth)

repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """
    Useful to get the details of a pull request from GitHub API: the author, title, body, diff_url, state, and head_sha.
    """
    repo = git.get_repo(full_repo_name)
    pull_request = repo.get_pull(pr_number)
    commit_shas = [c.sha for c in pull_request.get_commits()]

    return {
        "pr_author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "pr_diff_url": pull_request.diff_url,
        "pr_state": pull_request.state,
        "pr_commit_SHAs": commit_shas
    }


def get_repo_file(path: str) -> str:
    """Useful to get the content of a file from a GitHub repository."""
    repo = git.get_repo(full_repo_name)
    return repo.get_contents(path).decoded_content.decode('utf-8')


def get_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """
    Useful to get the details of a commit from GitHub API.

    :param commit_sha: The SHA hash of the commit to retrieve details for.
    :type commit_sha: str
    :return: A list of dictionaries where each dictionary contains details about a
        changed file in the commit, including filename, status, additions,
        deletions, changes, and patch.
    :rtype: list[dict[str, Any]]
    """
    repo = git.get_repo(full_repo_name)
    commit = repo.get_commit(commit_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


async def add_comment_to_state(draft_comment):
    """Useful to add a comment to the state of the CommentorAgent."""
    current_state = await context.store.get_state()
    current_state["review_comment"] = draft_comment
    await context.store.set_state(current_state)


def post_review(pr_number: int, comment: str) -> str:
    """
    Useful to post a review to a pull request on GitHub.

    Returns the URL of the posted review.
    """
    repo = git.get_repo(full_repo_name)
    pull_request = repo.get_pull(pr_number)
    review = pull_request.create_review(body=comment)
    # issue_comment = pull_request.create_issue_comment(comment)
    return review.html_url


async def add_review_to_state(review):
    """Useful to add a review to the state of the ReviewAndPostingAgent."""
    current_state = await context.store.get_state()
    current_state["final_review_comment"] = review
    await context.store.set_state(current_state)


llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

add_comment_to_state_tool = FunctionTool.from_defaults(add_comment_to_state)
commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_to_state_tool],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    system_prompt="""
You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
    - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
    - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
    - If you need any additional details, you must hand off to the CommentorAgent. \n
    - You should directly address the author. So your comments should sound like: "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
    - You MUST hand off to the ReviewAndPostingAgent once you are done drafting a review.
    - You MUST hand off to the ReviewAndPostingAgent once you are done drafting a review.
    - You MUST hand off to the ReviewAndPostingAgent once you are done drafting a review.
    - You MUST hand off to the ReviewAndPostingAgent once you are done drafting a review.
    - You MUST hand off to the ReviewAndPostingAgent once you are done drafting a review.    
 """
)

repo_file_tool = FunctionTool.from_defaults(get_repo_file)
commit_details_tool = FunctionTool.from_defaults(get_commit_details)
pr_details_tool = FunctionTool.from_defaults(get_pr_details)
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context ... ",
    tools=[pr_details_tool, repo_file_tool, commit_details_tool],
    system_prompt="""
    You are the context gathering agent. When gathering context, you MUST gather \n: 
        - The details: author, title, body, diff_url, state, and head_sha; \n
        - Changed files; \n
        - Any requested for files; \n
    Once you gather the requested info, you MUST hand control back to the Commentor Agent.
    """,
    can_handoff_to=["CommentorAgent"]
)

post_review_tool = FunctionTool.from_defaults(post_review)
add_review_to_state_tool = FunctionTool.from_defaults(add_review_to_state)
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Posts the review comment to the PR.",
    tools=[post_review_tool, add_review_to_state_tool],
    can_handoff_to=["CommentorAgent"],
    system_prompt="""
    You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
    Once a review is generated, you need to run a final check and post it to GitHub.
        - The review must: \n
        - Be a ~200-300 word review in markdown format. \n
        - Specify what is good about the PR: \n
        - Did the author follow ALL contribution rules? What is missing? \n
        - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
        - Are there notes on whether new endpoints were documented? \n
        - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
    \n
    If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
    When you are satisfied, post the review to GitHub.  
    """
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=commentor_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review_comment": ""
    },
)
context = Context(workflow_agent)


async def main():
    query =  "Write a review for PR: " + pr_nb
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
