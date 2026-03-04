import dspy
from typing import List, Optional
from tqdm.asyncio import tqdm
from utils import _retry_acall

class CollectiveIntelligenceSynthesis(dspy.Signature):  
    """  
    Incremental synthesis that combines previous summary with new congress transcript chunks.  
    This follows the Chain-of-Agents pattern where each agent builds upon previous agents' work.  
    """  
      
    original_query: str = dspy.InputField(desc="The original question that all congressional sessions debated")  
    previous_summary: Optional[str] = dspy.InputField(desc="accumulated synthesis summary from previously processed transcript chunks")  
    current_transcripts: List[str] = dspy.InputField(desc="current chunk of congressional session transcripts to analyze")  
      
    synthesis_summary: str = dspy.OutputField(desc="enhanced synthesis summary incorporating both previous knowledge and new transcript analysis")  
  
class CollectiveIntelligenceSynthetizer(dspy.Module):  
    """  
    Chain-of-Agents implementation that processes congress transcripts in chunks,  
    building synthesis incrementally to handle context length limitations.  
    """  
      
    def __init__(self, chunk_size: int = 10):  
        super().__init__()  
        self.chunk_size = chunk_size  
        self.reducer = dspy.ChainOfThought(CollectiveIntelligenceSynthesis)  
      
    def _chunk_transcripts(self, transcripts: List[str]) -> List[List[str]]:  
        """Split transcripts into manageable chunks."""  
        return [transcripts[i:i + self.chunk_size]   
                for i in range(0, len(transcripts), self.chunk_size)]  
      
    def forward(self, original_query: str, congress_transcripts: List[str]) -> dspy.Prediction:  
        """  
        Process transcripts using Chain-of-Agents reduce pattern.  
        Each chunk builds upon synthesis from previous chunks.  
        """  
        chunks = self._chunk_transcripts(congress_transcripts)  
        accumulated_summary = ""  
          
        for chunk_idx, chunk in enumerate(chunks):  
            result = self.reducer(  
                original_query=original_query,  
                previous_summary=accumulated_summary,  
                current_transcripts=chunk  
            )  
              
            # Update accumulated summary with refined version  
            accumulated_summary = result.synthesis_summary  
          
        return dspy.Prediction(  
            collective_answer=accumulated_summary,  
            total_chunks_processed=len(chunks)  
        )  
      
    async def aforward(self, original_query: str, congress_transcripts: List[str]) -> dspy.Prediction:  
        """Async version of the chain-of-agents reducer."""  
        chunks = self._chunk_transcripts(congress_transcripts)  
        accumulated_summary = ""  
          
        for chunk in tqdm(chunks, desc="Synthesizing"):  
            result = await _retry_acall(
                self.reducer,
                original_query=original_query,  
                previous_summary=accumulated_summary,  
                current_transcripts=chunk  
            )  
              
            accumulated_summary = result.synthesis_summary  
          
        return dspy.Prediction(  
            collective_answer=accumulated_summary,  
            total_chunks_processed=len(chunks)  
        )
