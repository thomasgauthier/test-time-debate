import markdown
from bs4 import BeautifulSoup
import dspy
import asyncio
import time
from typing import Any, Callable, Optional
from persistence import durable_memo
from config import console, VERBOSE

@durable_memo
async def _retry_acall(
    component: Any,
    validator: Optional[Callable[[Any], None]] = None,
    **kwargs,
):
    import dspy
    
    start_time = time.perf_counter()
    last_exception = None
    attempt = 0
    
    while True:
        try:
            if attempt > 0:
                lm = dspy.settings.lm.copy()
                lm.cache = False
                with dspy.context(lm=lm):
                    result = await component.acall(**kwargs)
            else:
                result = await component.acall(**kwargs)

            if validator:
                validator(result)

            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if VERBOSE:
                component_name = component.__class__.__name__
                if hasattr(component, 'signature'):
                    component_name = f"{component_name}({component.signature.__name__})"
                
                retry_str = f" after {attempt} retries" if attempt > 0 else ""
                console.print(
                    f"[bold green]✓[/] [cyan]{component_name}[/] completed in [yellow]{duration:.2f}s[/]{retry_str}"
                )

            return result

        except Exception as e:
            attempt += 1
            last_exception = e
            wait_time = min(2**attempt, 60)
            console.print(
                f"[bold red]⚠️  {component.__class__.__name__} failed (Attempt {attempt + 1}). "
                f"Error: {str(e)}. Retrying in {wait_time}s...[/]"
            )
            await asyncio.sleep(wait_time)
    raise last_exception

def extract_quotes_from_markdown(text: str) -> list[str]:
    """Extract quoted text that appears in strong/bold formatting in a markdown document."""
    html_content = markdown.markdown(text)
    soup = BeautifulSoup(html_content, "lxml")
    strong_tags = soup.find_all("strong")
    
    quotes_list = []
    for tag in strong_tags:
        tag_text = tag.get_text().strip()
        if ((tag_text.startswith('"') or tag_text.startswith('“')) and 
            (tag_text.endswith('"') or tag_text.endswith('”'))) and len(tag_text) > 1:
            quote_content = tag_text[1:-1]
            quotes_list.append(quote_content)
    
    return quotes_list


def format_congress_result(result: dict) -> str:
    """Format a single congress result for output."""
    formatted = f"CONGRESS {result['congress_id']}\n"
    formatted += "-" * 30 + "\n"
    formatted += f"Documents: {result['num_documents']}\n"
    formatted += f"Document Titles: {', '.join(result['document_titles'])}\n"
    formatted += f"Query: {result['query']}\n\n"
    formatted += "TRANSCRIPT:\n"
    formatted += result['transcript']
    return formatted

def verify_and_wrap_quotes(response_text: str, source_document: str) -> str:
    """Verify quotes against source document and wrap with appropriate tags."""
    quotes = extract_quotes_from_markdown(response_text)
    
    for quote in quotes:
        # Check if quote exists in source document
        if quote in source_document:
            # Replace with verified quote wrapper
            response_text = response_text.replace(
                f'**"{quote}"**', 
                f'<v_quote>"{quote}"</v_quote>'
            )

            response_text = response_text.replace(
                f'**"{quote}"**', 
                f'<v_quote>"{quote}"</v_quote>'
            )
        else:
            # Replace with unverified quote wrapper  
            response_text = response_text.replace(
                f'**"{quote}"**', 
                f'<u_quote>"{quote}"</u_quote>'
            )

            response_text = response_text.replace(
                f'**"{quote}"**', 
                f'<u_quote>"{quote}"</u_quote>'
            )
    
    return response_text
