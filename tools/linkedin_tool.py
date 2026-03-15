"""
LinkedIn Tool — A mock tool to post content to the user's LinkedIn profile.

For security reasons, actual posting requires OAuth tokens and user consent on the platform.
This tool simulates scheduling the post by saving it locally in a JSON file.
"""

import json
import os
from datetime import datetime
from langchain_core.tools import tool

LINKEDIN_POSTS_FILE = os.path.join(os.path.dirname(__file__), "..", "linkedin_posts.json")


@tool
def post_to_linkedin(post_content: str) -> str:
    """Publish a new post to the user's LinkedIn profile.

    Use this tool when the user asks to "Make a post on LinkedIn", "Share to LinkedIn", or "Post my resume on LinkedIn".

    Args:
        post_content: The full text content of the post to be published.
    """
    if not post_content or not post_content.strip():
        return "Error: Post content cannot be empty."

    # In a real application, you would make an API request with an OAuth token:
    # requests.post("https://api.linkedin.com/v2/ugcPosts", headers={"Authorization": f"Bearer {TOKEN}"}, json={...})
    
    post_data = {
        "content": post_content,
        "posted_at": datetime.now().isoformat(),
        "status": "published_locally"
    }

    # Simulate saving it to a local file
    try:
        posts = []
        if os.path.exists(LINKEDIN_POSTS_FILE):
            with open(LINKEDIN_POSTS_FILE, "r", encoding="utf-8") as f:
                posts = json.load(f)
        
        posts.append(post_data)
        
        with open(LINKEDIN_POSTS_FILE, "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        return f"❌ Failed to publish post: {str(e)}"

    summary = (
        f"✅ Successfully created a new LinkedIn post!\n\n"
        f"**Preview:**\n"
        f"> {post_content}\n\n"
        f"*(Note: Since this is a demonstration environment, the post was recorded locally to '{os.path.basename(LINKEDIN_POSTS_FILE)}' rather than sent to the actual LinkedIn servers)*"
    )
    
    return summary
