from dotenv import load_dotenv

load_dotenv()

from langfuse import observe, get_client

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

DSPyInstrumentor().instrument()
LiteLLMInstrumentor().instrument()

import dspy
import random
import json
import sqlite3
import asyncio
import hashlib
import os
from typing import List, Optional, Callable, Any
from tqdm.asyncio import tqdm
from utils import verify_and_wrap_quotes, format_congress_result, _retry_acall
from conversational_agent import ConversationalAgent
from swarm_intelligence_reducer import SwarmIntelligenceReducer
from insight_reporter import InsightReporter
from collective_intelligence_synthetizer import CollectiveIntelligenceSynthetizer
from config import console

lm_kwargs = {
    "api_key": "hello",
    "api_base": "http://localhost:9292/v1",
    "model": "openai/qwen3.5-35b-a3b",
}

lm = dspy.LM(**lm_kwargs)


dspy.configure(lm=lm)


class ConversationTranscript:
    """Stateful conversation transcript manager for the Swarm Intelligence Congress."""

    def __init__(
        self, query: str, documents: List[dict] = None, congress_id: int = None
    ):
        self.query = query
        self.documents = documents or []
        self.congress_id = congress_id
        self.turns = []
        self.welcome_message = (
            "Welcome to the Swarm Intelligence Congress of Documents. This congressional session is one of many "
            "parallel groups operating simultaneously as part of a broader collective intelligence system. "
            "Each representative here has been sent by their source document to advocate for its perspectives, "
            "insights, and knowledge. As document representatives, you are tasked with ensuring your document's "
            "voice is heard while collaborating to build collective intelligence through respectful congressional "
            "discourse. Your deliberations will contribute to the emergent wisdom of the entire document swarm."
        )

    def add_turn(self, speaker: str, content: str) -> None:
        """Add a new turn to the conversation transcript.

        Args:
            speaker (str): Name/identifier of the speaker (e.g., "Representative 1 (Doc Title)" or "Intelligence Reporter")
            content (str): The content of the turn
        """
        self.turns.append({"speaker": speaker, "content": content})

    def to_prompt_fragment(self) -> str:
        """Convert the conversation transcript to a string format for prompting."""
        transcript = self.welcome_message

        # Add congress session information
        if self.congress_id is not None:
            transcript += f"\n\nCONGRESS SESSION {self.congress_id}"

        # Add document information
        if self.documents:
            transcript += f"\nDocuments in this session: {len(self.documents)}"
            document_titles = [
                doc.get("title", f"Document {doc.get('id', 'Unknown')}")
                for doc in self.documents
            ]
            transcript += f"\nDocument Titles: {', '.join(document_titles)}"

        transcript += (
            f"\n\nCongress is debating the following question: {self.query}\n\n"
        )

        for turn in self.turns:
            speaker = turn["speaker"]
            if "Intelligence Reporter" in speaker or "Insight Reporter" in speaker:
                transcript += f"\n--- INTELLIGENCE BRIEFING FROM {speaker} ---\n"
                transcript += f"{turn['content']}\n"
                transcript += "--- END INTELLIGENCE BRIEFING ---\n\n"
            else:
                transcript += f"{speaker}: {turn['content']}\n"

        if self.is_empty():
            transcript += (
                "\n\n [CONVERSATION EMPTY, YOU ARE THE FIRST REPRESENTATIVE TO SPEAK]"
            )

        return transcript

    def is_empty(self) -> bool:
        """Check if there are any turns in the conversation."""
        return len(self.turns) == 0

    def get_turn_count(self) -> int:
        """Get the total number of turns in the conversation."""
        return len(self.turns)


def get_all_documents_from_db(db_path: str) -> List[dict]:
    """
    Get all documents from the database with their metadata.

    Args:
        db_path (str): Path to the SQLite database

    Returns:
        list: List of document dictionaries with id, title, and txt_content
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
        SELECT id, title, txt_content 
        FROM documents 
        WHERE txt_content IS NOT NULL AND txt_content != ''
        """

        cursor.execute(query)
        results = cursor.fetchall()

        documents = []
        for row in results:
            doc_id, title, txt_content = row
            documents.append(
                {
                    "id": doc_id,
                    "title": title or f"Document {doc_id}",
                    "content": f"# {title or f'Document {doc_id}'}\n\n{txt_content}",
                }
            )

        conn.close()
        return documents

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def split_documents_into_groups(
    documents: List[dict], group_size: int = 5, randomize: bool = False
) -> List[List[dict]]:
    """
    Split documents into groups of specified size.

    Args:
        documents (List[dict]): List of document dictionaries
        group_size (int): Size of each group (default: 5)
        randomize (bool): Whether to shuffle documents randomly (default: False)

    Returns:
        List[List[dict]]: List of document groups
    """
    # Shuffle documents randomly if requested
    if randomize:
        shuffled_docs = documents.copy()
        # Set a fixed seed for deterministic randomization
        random.seed(42)
        random.shuffle(shuffled_docs)
    else:
        shuffled_docs = documents

    # Split into groups
    groups = []
    for i in range(0, len(shuffled_docs), group_size):
        group = shuffled_docs[i : i + group_size]
        groups.append(group)

    return groups


async def run_congress(
    documents: List[dict],
    query: str,
    congress_id: int = 1,
    semaphore: asyncio.Semaphore = None,
    conversation: ConversationTranscript = None,
):
    """Main async function to run the congressional debate.

    Args:
        documents (List[dict]): List of document dictionaries with content
        query (str): The question/topic for the congressional debate
        congress_id (int): Identifier for this congress
        semaphore (asyncio.Semaphore): Semaphore to control concurrent operations
        conversation (ConversationTranscript): Optional existing conversation to continue

    Returns:
        tuple: (results_dict, conversation_transcript)
    """
    agent = ConversationalAgent()

    # Create or use existing conversation transcript
    if conversation is None:
        conversation = ConversationTranscript(query, documents, congress_id)

    results = {
        "congress_id": congress_id,
        "num_documents": len(documents),
        "document_titles": [doc["title"] for doc in documents],
        "query": query,
        "transcript": "",
        "turns": [],
    }

    for i, doc in enumerate(documents):
        # Prepare conversation context
        context = conversation.to_prompt_fragment()

        # Use semaphore before awaited call if provided
        if semaphore:
            async with semaphore:
                doc_response = await _retry_acall(
                    agent,
                    query=query,
                    conversation_transcript=context,
                    source_document=doc["content"],
                )
        else:
            doc_response = await _retry_acall(
                agent,
                query=query,
                conversation_transcript=context,
                source_document=doc["content"],
            )

        # Verify quotes and wrap them
        verified_response = verify_and_wrap_quotes(
            doc_response.next_turn, doc["content"]
        )

        # Add turn to conversation
        representative_name = f"Representative {i + 1} ({doc['title']})"
        conversation.add_turn(representative_name, verified_response)

        # Store turn result
        results["turns"].append(
            {
                "turn": 1,
                "representative": representative_name,
                "document_id": doc["id"],
                "response": verified_response,
            }
        )

    # Store final transcript
    results["transcript"] = conversation.to_prompt_fragment()

    return results, conversation


async def run_multiple_congresses(
    query: str, group_size: int = 5, max_concurrent: int = 1
):
    """
    Run multiple congresses simultaneously with documents from the database.

    Args:
        db_path (str): Path to the SQLite database
        query (str): The question/topic for all congressional debates
        group_size (int): Size of each congress group (default: 5)
        max_concurrent (int): Maximum number of concurrent operations (default: 3)

    Returns:
        List[dict]: Results from all congresses
    """

    # Get all documents from database
    with open("all_documents.json", "r") as file:
        all_documents = json.load(file)

    # Split documents into groups
    document_groups = split_documents_into_groups(
        all_documents, group_size, randomize=True
    )

    # Create semaphore to control concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for parallel execution
    congress_tasks = []
    for i, group in enumerate(document_groups):
        task = run_congress(group, query, congress_id=i + 1, semaphore=semaphore)
        congress_tasks.append(task)

    # Run all congresses simultaneously with progress tracking using tqdm.asyncio
    congress_results = await tqdm.gather(*congress_tasks, desc="Congresses")

    # Extract results and conversations (both elements of each tuple)
    results = [result for result, _ in congress_results]
    conversations = [conversation for _, conversation in congress_results]

    return results, conversations, document_groups


def get_query_directory(query: str) -> str:
    """
    Generate a directory path based on the SHA256 hash of the query.

    Args:
        query (str): The query string to hash

    Returns:
        str: Directory path based on query hash
    """
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:7]
    return query_hash


def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_congress_results(
    results: List[dict], query: str, output_file: str = "congress_results.txt"
):
    """
    Save congress results to a text file in a query-specific directory.

    Args:
        results (List[dict]): Results from all congresses
        query (str): The original query to generate directory path
        output_file (str): Output file name (default: "congress_results.txt")
    """

    # Create query-specific directory
    query_dir = get_query_directory(query)
    ensure_directory_exists(query_dir)

    # Create full path
    full_path = os.path.join(query_dir, output_file)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write("SWARM INTELLIGENCE CONGRESS RESULTS\n")
        f.write("=" * 50 + "\n\n")

        for result in results:
            f.write(format_congress_result(result))
            f.write("\n\n" + "=" * 50 + "\n\n")

    print(f"📄 Results saved to: {full_path}")


async def main():
    question = "I'm trying to quit weed after years of a daily smoking habit. It's now 8PM, I'm home and I'm bored and this is exactly the kind of situation where I'd roll a joint and get high but now I decided I won't but I just feel the urge so bad. I can't think about anytyhing else. What am I gonna do? I can't live like this for the rest of my days."

    query = f"""What would the author(s) of your documents say about this: <current_question>{question}</current_question>?"""

    console.print(f"\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print(f"[bold magenta]🚀 SWARM INTELLIGENCE CONGRESS: {question}[/]")
    console.print(f"[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]\n")

    # Run multiple congresses and get results, conversations, and document groups
    console.print(f"[bold blue]🏛️  STAGE 1: Parallel Congresses (Initial Debate)[/]")
    results, conversations, document_groups = await run_multiple_congresses(
        query, group_size=5
    )

    # Create query-specific directory and save the original question
    query_dir = get_query_directory(query)
    ensure_directory_exists(query_dir)

    # Save the original question
    question_path = os.path.join(query_dir, "question.txt")
    with open(question_path, "w", encoding="utf-8") as f:
        f.write(question)
    console.print(f"❓ Saved original question to '{question_path}'")

    save_congress_results(results, query)

    console.print(f"\n[bold blue]🧠 STAGE 2: Global Insight Reduction[/]")
    transcripts = [format_congress_result(r) for r in results]

    reducer = SwarmIntelligenceReducer(chunk_size=12)
    insights_results = await _retry_acall(reducer, conversations=transcripts)

    console.print(f"\n[bold blue]📊 STAGE 3: Intelligence Reporting (Cross-Congress Diffusion)[/]")
    reporter = InsightReporter()

    sem = asyncio.Semaphore(25)

    async def bounded_report(transcript):
        async with sem:
            return await _retry_acall(
                reporter,
                local_transcript=transcript,
                external_insights=insights_results.final_insights,
            )

    tasks = [
        asyncio.create_task(bounded_report(transcript)) for transcript in transcripts
    ]
    console.print(f"🔄 Generating {len(tasks)} parallel reports...")
    reports = await tqdm.gather(*tasks)

    # Add intelligence briefings to conversations that need them
    briefing_count = 0
    for conversation, report in zip(conversations, reports):
        if report.intelligence_briefing != "NO BRIEFING NEEDED":
            conversation.add_turn("Intelligence Reporter", report.intelligence_briefing)
            briefing_count += 1
    
    console.print(f"🎯 Diffusion: {briefing_count}/{len(reports)} congresses received briefings")

    # Run second round of debates with intelligence briefings
    console.print(f"\n[bold blue]🏛️  STAGE 4: Recursive Deliberation (Informed Debate)[/]")

    agent = ConversationalAgent()
    sem_second_round = asyncio.Semaphore(25)

    async def run_second_round_for_congress(conversation, documents):
        async with sem_second_round:
            # Only run second round if intelligence briefing was added
            has_intelligence = any(
                "Intelligence Reporter" in turn["speaker"]
                for turn in conversation.turns
            )
            if not has_intelligence:
                return conversation

            # Run each representative again with the enhanced context
            for i, doc in enumerate(documents):
                context = conversation.to_prompt_fragment()

                doc_response = await _retry_acall(
                    agent,
                    query=conversation.query,
                    conversation_transcript=context,
                    source_document=doc["content"],
                )

                # Verify quotes and wrap them
                verified_response = verify_and_wrap_quotes(
                    doc_response.next_turn, doc["content"]
                )

                # Add turn to conversation
                representative_name = f"Representative {i + 1} ({doc['title']})"
                conversation.add_turn(representative_name, verified_response)

            return conversation

    # Create tasks for second round
    second_round_tasks = []
    for conversation, documents in zip(conversations, document_groups):
        task = run_second_round_for_congress(conversation, documents)
        second_round_tasks.append(task)

    # Run second round
    enhanced_conversations = await tqdm.gather(*second_round_tasks, desc="Second Round")

    # Save conversations with intelligence briefings and second round
    conversations_path = os.path.join(query_dir, "conversations_with_briefings.txt")
    with open(conversations_path, "w", encoding="utf-8") as f:
        f.write("SWARM INTELLIGENCE CONGRESS WITH COLLECTIVE INTELLIGENCE BRIEFINGS\n")
        f.write("=" * 70 + "\n\n")

        for conversation in enhanced_conversations:
            f.write(f"CONGRESS SESSION {conversation.congress_id}\n")
            f.write("-" * 40 + "\n")
            f.write(conversation.to_prompt_fragment())
            f.write("\n\n" + "=" * 70 + "\n\n")

    # save reports to a file
    reports_path = os.path.join(query_dir, "reports.txt")
    with open(reports_path, "w") as f:
        for report in reports:
            f.write(f"# INTELLIGENCE BRIEFING\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"## RELEVANCE ASSESSMENT\n")
            f.write(report.relevance_assessment)
            f.write("\n\n" + "=" * 30 + "\n\n")
            f.write(f"## INTELLIGENCE BRIEFING\n")
            f.write(report.intelligence_briefing)
            f.write("\n\n" + "=" * 50 + "\n\n")

    console.print(f"💾 Saved outputs to '{query_dir}'")

    # Final synthesis step
    console.print(f"\n[bold blue]🧠 STAGE 5: Collective Intelligence Synthesis[/]")

    # Get all final transcripts
    final_transcripts = [
        conversation.to_prompt_fragment() for conversation in enhanced_conversations
    ]

    # Create synthesizer and generate final answer
    synthesizer = CollectiveIntelligenceSynthetizer(chunk_size=12)
    final_synthesis = await _retry_acall(
        synthesizer, original_query=query, congress_transcripts=final_transcripts
    )

    # Save final synthesis
    final_answer_path = os.path.join(query_dir, "final_collective_answer.txt")
    with open(final_answer_path, "w", encoding="utf-8") as f:
        f.write("COLLECTIVE INTELLIGENCE SYNTHESIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"ORIGINAL QUERY:\n{query}\n\n")
        f.write("=" * 50 + "\n\n")
        f.write("COLLECTIVE ANSWER FROM DOCUMENT SWARM:\n")
        f.write(final_synthesis.collective_answer)
        f.write(f"\n\n" + "=" * 50 + "\n")
        f.write(
            f"Processed {final_synthesis.total_chunks_processed} transcript chunks\n"
        )
        f.write(
            f"Synthesized from {len(enhanced_conversations)} congressional sessions\n"
        )

    console.print(f"\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print(f"[bold magenta]🎯 FINAL SYNTHESIS COMPLETED![/]")
    console.print(f"[bold magenta]📄 Answer saved to: {final_answer_path}[/]")
    console.print(f"[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]\n")


if __name__ == "__main__":
    asyncio.run(main())
