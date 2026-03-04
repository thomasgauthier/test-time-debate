from typing import List, Dict, Any
import dspy
from tqdm.asyncio import tqdm
from utils import _retry_acall
  
class SwarmIntelligenceReduction(dspy.Signature):  
    """  
    Incremental insight extraction that combines previous insights with new conversation chunks.  
    This follows the Chain-of-Agents pattern where each agent builds upon previous agents' work.  
    """  
      
    previous_insights: List[str] = dspy.InputField(desc="insights extracted from previously processed conversation chunks")  
    current_conversations: List[str] = dspy.InputField(desc="current chunk of congressional session transcripts to analyze")  
      
    updated_insights: List[str] = dspy.OutputField(desc="refined and expanded insights combining previous knowledge with new findings")  
    new_discoveries: List[str] = dspy.OutputField(desc="novel insights discovered in the current chunk that weren't present before")  
  
  
class SwarmIntelligenceReducer(dspy.Module):  
    """  
    Chain-of-Agents implementation that processes conversations in chunks,  
    building insights incrementally to handle context length limitations.  
    """  
      
    def __init__(self, chunk_size: int = 10):  
        super().__init__()  
        self.chunk_size = chunk_size  
        self.reducer = dspy.ChainOfThought(SwarmIntelligenceReduction)  
      
    def _chunk_conversations(self, conversations: List[str]) -> List[List[str]]:  
        """Split conversations into manageable chunks."""  
        return [conversations[i:i + self.chunk_size]   
                for i in range(0, len(conversations), self.chunk_size)]  
      
    def forward(self, conversations: List[str]) -> dspy.Prediction:  
        """  
        Process conversations using Chain-of-Agents reduce pattern.  
        Each chunk builds upon insights from previous chunks.  
        """  
        chunks = self._chunk_conversations(conversations)  
        accumulated_insights = []  
        all_discoveries = []  
          
        for chunk_idx, chunk in enumerate(chunks):  
            result = self.reducer(  
                previous_insights=accumulated_insights,  
                current_conversations=chunk  
            )  
              
            # Update accumulated insights with refined version  
            accumulated_insights = result.updated_insights  
            all_discoveries.extend(result.new_discoveries)  
          
        return dspy.Prediction(  
            final_insights=accumulated_insights,  
            discoveries_by_chunk=all_discoveries,  
            total_chunks_processed=len(chunks)  
        )  
      
    async def aforward(self, conversations: List[str]) -> dspy.Prediction:  
        """Async version of the chain-of-agents reducer."""  
        
        chunks = self._chunk_conversations(conversations)  
        accumulated_insights = []  
        all_discoveries = []  
          
        for chunk in tqdm(chunks, desc="Processing conversation chunks"):  
            result = await _retry_acall(
                self.reducer,
                previous_insights=accumulated_insights,  
                current_conversations=chunk  
            )
              
            accumulated_insights = result.updated_insights  
            all_discoveries.extend(result.new_discoveries)  
          
        return dspy.Prediction(  
            final_insights=accumulated_insights,  
            discoveries_by_chunk=all_discoveries,  
            total_chunks_processed=len(chunks)  
        )