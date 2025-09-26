"""
Interactive Provenance-Aware QA System

Streamlined interface for asking questions about contracts with full provenance tracking.
"""
import sys
from pathlib import Path
import time
import json
from typing import List

# Add src directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from src.provenance_qa import create_qa_engine, QAResponse
    print("✅ Successfully imported Module 9 QA system")
except ImportError as e:
    print(f"❌ Error importing Module 9: {e}")
    print("Make sure you're running from the ContractSense directory")
    sys.exit(1)

class InteractiveQA:
    def __init__(self):
        self.workspace_path = Path.cwd()
        self.qa_engine = None
        self.session_history = []
        
    def initialize_engine(self):
        """Initialize the QA engine with optional API key"""
        print("🔧 Initializing ContractSense QA System...")
        
        # Check for API key
        gemini_key = None
        try:
            import os
            gemini_key = os.getenv('GEMINI_API_KEY')
            if not gemini_key:
                user_key = input("Enter Gemini API key (or press Enter to use fallback): ").strip()
                if user_key:
                    gemini_key = user_key
        except:
            pass
        
        # Create QA engine
        try:
            self.qa_engine = create_qa_engine(
                workspace_path=str(self.workspace_path),
                gemini_api_key=gemini_key
            )
            
            api_status = "Gemini API" if gemini_key else "Fallback Mode"
            print(f"✅ QA Engine initialized successfully! ({api_status})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize QA engine: {e}")
            return False
    
    def display_welcome(self):
        """Display welcome message and suggested questions"""
        print("\n" + "="*80)
        print("🤖 CONTRACTSENSE INTERACTIVE QA SYSTEM")
        print("="*80)
        print("Ask natural language questions about contracts!")
        print("The system provides answers with full source citations and confidence scores.")
        print("\n💡 SUGGESTED QUESTIONS TO TRY:")
        
        suggestions = [
            # Factual Questions
            ("📋 FACTUAL QUESTIONS", [
                "What are the termination clauses in the contract?",
                "Who are the parties to this agreement?",
                "What is the contract's effective date?",
                "What are the payment terms?",
                "Are there any renewal options?",
                "What is the contract duration?",
                "What are the key deliverables?"
            ]),
            
            # Comparative Questions  
            ("🔍 COMPARATIVE QUESTIONS", [
                "How do liability provisions differ between agreements?",
                "Compare the termination clauses across contracts",
                "What are the differences in payment terms?",
                "How do indemnification clauses vary?",
                "Compare renewal options between contracts"
            ]),
            
            # Risk Analysis Questions
            ("⚖️ RISK ANALYSIS QUESTIONS", [
                "What are the potential risks in this indemnification clause?",
                "What legal exposure does this contract create?", 
                "What are the penalty provisions?",
                "What compliance requirements are imposed?",
                "What are the liability caps and limitations?"
            ]),
            
            # Procedural Questions
            ("🔧 PROCEDURAL QUESTIONS", [
                "How should we handle a breach of contract?",
                "What steps are required for contract termination?",
                "How do we exercise the renewal option?",
                "What notice requirements exist?",
                "How are disputes resolved?"
            ])
        ]
        
        for category, questions in suggestions:
            print(f"\n{category}:")
            for i, question in enumerate(questions, 1):
                print(f"  {i}. {question}")
        
        print("\n" + "="*80)
        print("💬 COMMANDS:")
        print("  • Type your question and press Enter")
        print("  • 'help' - Show this help again")
        print("  • 'stats' - Show session statistics") 
        print("  • 'history' - Show question history")
        print("  • 'save' - Save session to file")
        print("  • 'q' or 'quit' - Exit")
        print("="*80)
    
    def format_answer(self, response: QAResponse) -> None:
        """Format and display the answer beautifully"""
        # Get answer text from QAResponse
        answer_text = response.answer if hasattr(response, 'answer') else str(response)
        
        # Get confidence and quality scores directly from QAResponse
        confidence = getattr(response, 'overall_confidence', 0.0)
        quality = getattr(response, 'answer_quality', 0.0) 
        processing_time = getattr(response, 'processing_time', 0.0)
        
        # Get citations directly from QAResponse
        citations = getattr(response, 'citations', [])
        
        # Get sources from QAResponse
        sources = getattr(response, 'source_documents', [])
            
        print(f"\n📝 ANSWER:")
        print(f"   {answer_text}")
        
        print(f"\n📊 RESPONSE DETAILS:")
        print(f"   🎯 Confidence: {confidence:.2f}/1.00")
        print(f"   ⭐ Quality: {quality:.2f}/1.00") 
        print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
        print(f"   📚 Citations: {len(citations)}")
        print(f"   📄 Sources: {len(sources)} document(s)")
        
        # Show confidence interpretation
        if confidence >= 0.8:
            conf_desc = "🟢 High (Very Reliable)"
        elif confidence >= 0.6:
            conf_desc = "🟡 Good (Generally Reliable)"
        elif confidence >= 0.4:
            conf_desc = "🟠 Moderate (Use with Caution)"
        else:
            conf_desc = "🔴 Low (Requires Verification)"
        
        print(f"   📈 Confidence Level: {conf_desc}")
        
        # Show top citations
        if citations:
            print(f"\n📚 TOP SOURCES:")
            for i, citation in enumerate(citations[:3], 1):
                if hasattr(citation, 'source_title') and citation.source_title:
                    print(f"   {i}. {citation.source_title}")
                elif hasattr(citation, 'document_title'):
                    print(f"   {i}. {citation.document_title}")
                elif hasattr(citation, 'title'):
                    print(f"   {i}. {citation.title}")
                else:
                    print(f"   {i}. Document {i}")
                    
                if hasattr(citation, 'cited_text') and citation.cited_text:
                    preview = citation.cited_text[:100] + "..." if len(citation.cited_text) > 100 else citation.cited_text
                    print(f"      \"{preview}\"")
                elif hasattr(citation, 'relevant_text') and citation.relevant_text:
                    preview = citation.relevant_text[:100] + "..." if len(citation.relevant_text) > 100 else citation.relevant_text
                    print(f"      \"{preview}\"")
                elif hasattr(citation, 'content') and citation.content:
                    preview = citation.content[:100] + "..." if len(citation.content) > 100 else citation.content
                    print(f"      \"{preview}\"")
        
        print("\n" + "-"*80)
    
    def show_session_stats(self):
        """Display session statistics"""
        if not self.session_history:
            print("📊 No questions asked yet in this session.")
            return
            
        total_questions = len(self.session_history)
        avg_confidence = sum(q['confidence'] for q in self.session_history) / total_questions
        avg_quality = sum(q['quality'] for q in self.session_history) / total_questions
        avg_time = sum(q['processing_time'] for q in self.session_history) / total_questions
        total_citations = sum(q['citations'] for q in self.session_history)
        
        print(f"\n📊 SESSION STATISTICS:")
        print(f"   Questions Asked: {total_questions}")
        print(f"   Average Confidence: {avg_confidence:.2f}")
        print(f"   Average Quality: {avg_quality:.2f}")  
        print(f"   Average Processing Time: {avg_time:.2f}s")
        print(f"   Total Citations Generated: {total_citations}")
    
    def show_history(self):
        """Show question history"""
        if not self.session_history:
            print("📚 No questions asked yet.")
            return
            
        print(f"\n📚 QUESTION HISTORY ({len(self.session_history)} questions):")
        for i, item in enumerate(self.session_history, 1):
            print(f"\n{i}. Q: {item['question']}")
            print(f"   A: {item['answer'][:100]}{'...' if len(item['answer']) > 100 else ''}")
            print(f"   📊 Confidence: {item['confidence']:.2f} | Time: {item['processing_time']:.1f}s")
    
    def save_session(self):
        """Save session to file"""
        if not self.session_history:
            print("📄 No session data to save.")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"qa_session_{timestamp}.json"
        
        session_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(self.session_history),
            "questions_and_answers": self.session_history
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Session saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving session: {e}")
    
    def run_interactive_mode(self):
        """Run the interactive question-answering session"""
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 Your question: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("\n👋 Thank you for using ContractSense QA! Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.display_welcome()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_session_stats()
                    continue  
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'save':
                    self.save_session()
                    continue
                
                # Process the question
                print(f"\n🤔 Processing: {user_input}")
                start_time = time.time()
                
                try:
                    response = self.qa_engine.ask_question(user_input)
                    processing_time = time.time() - start_time
                    
                    # Display the answer
                    self.format_answer(response)
                    
                    # Extract values for history (with safe defaults)
                    answer_text = getattr(response, 'answer', str(response))
                    confidence = getattr(response, 'overall_confidence', 0.0)
                    quality = getattr(response, 'answer_quality', 0.0)
                    citations = getattr(response, 'citations', [])
                    
                    # Save to history
                    self.session_history.append({
                        'question': user_input,
                        'answer': answer_text,
                        'confidence': confidence,
                        'quality': quality,
                        'processing_time': processing_time,
                        'citations': len(citations)
                    })
                    
                except Exception as e:
                    print(f"❌ Error processing question: {e}")
                    print("Please try rephrasing your question or check the system status.")
                    # Print traceback for debugging
                    import traceback
                    traceback.print_exc()
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue

def main():
    """Main entry point"""
    print("🚀 Starting ContractSense Interactive QA System...")
    
    interactive_qa = InteractiveQA()
    
    if not interactive_qa.initialize_engine():
        print("Failed to initialize. Exiting.")
        return
    
    interactive_qa.run_interactive_mode()

if __name__ == "__main__":
    main()