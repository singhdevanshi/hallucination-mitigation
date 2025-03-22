import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import wikipedia
from scholarly import scholarly
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sentence_transformers import SentenceTransformer
import spacy
from crossref_commons.retrieval import get_publication_as_json
import re

class ComprehensiveEvaluator:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B"):
        """Initialize the comprehensive evaluation framework."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models and tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize entailment model for logical consistency
        self.entailment_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large-mnli",
            device_map="auto"
        )
        self.entailment_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large-mnli")
        
        # Initialize zero-shot classifier for uncertainty detection
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize spaCy for citation parsing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load evaluation datasets
        self.datasets = self._load_datasets()
        
        # Initialize Wikidata SPARQL endpoint
        self.wikidata_endpoint = "https://query.wikidata.org/sparql"
        
        # Initialize Semantic Scholar API
        self.semantic_scholar_api = "https://api.semanticscholar.org/v1"
        
        # Initialize citation databases
        self.crossref_api = "https://api.crossref.org/works"
        self.semantic_scholar_citations = "https://api.semanticscholar.org/v1/paper"
        
    def _load_datasets(self) -> Dict[str, Any]:
        """Load all evaluation datasets."""
        datasets = {}
        
        # Load TruthfulQA
        datasets['truthful_qa'] = load_dataset("truthful_qa", "multiple_choice")
        
        # Load HaluEval
        datasets['halu_eval'] = load_dataset("HaluEval/halu_eval")
        
        # Load FActScore
        datasets['fact_score'] = load_dataset("fact_score")
        
        # Load TRUE
        datasets['true'] = load_dataset("true")
        
        # Load FACTOR
        datasets['factor'] = load_dataset("factor")
        
        return datasets
    
    def evaluate_factual_accuracy(self, responses: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate factual accuracy using knowledge base verification."""
        scores = []
        
        for response, reference in zip(responses, references):
            # Extract claims from response
            claims = self._extract_claims(response)
            
            # Verify each claim against knowledge bases
            verified_claims = []
            for claim in claims:
                # Check Wikipedia
                wiki_verified = self._verify_wikipedia(claim)
                
                # Check Wikidata
                wikidata_verified = self._verify_wikidata(claim)
                
                # Check Semantic Scholar
                scholar_verified = self._verify_semantic_scholar(claim)
                
                # Consider claim verified if any source confirms it
                verified = wiki_verified or wikidata_verified or scholar_verified
                verified_claims.append(verified)
            
            # Calculate accuracy for this response
            if claims:
                accuracy = sum(verified_claims) / len(claims)
                scores.append(accuracy)
        
        return {
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores)
        }
    
    def evaluate_logical_consistency(self, responses: List[str]) -> Dict[str, float]:
        """Evaluate logical consistency using entailment models."""
        scores = []
        
        for response in responses:
            # Split response into sentences
            sentences = self._split_into_sentences(response)
            
            # Check consistency between consecutive sentences
            consistency_scores = []
            for i in range(len(sentences) - 1):
                # Check if sentence i entails sentence i+1
                inputs = self.entailment_tokenizer(
                    sentences[i],
                    sentences[i + 1],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.entailment_model(**inputs)
                entailment_score = torch.softmax(outputs.logits, dim=1)[0][1].item()
                consistency_scores.append(entailment_score)
            
            if consistency_scores:
                scores.append(np.mean(consistency_scores))
        
        return {
            "mean_consistency": np.mean(scores),
            "std_consistency": np.std(scores)
        }
    
    def evaluate_uncertainty_expression(self, responses: List[str]) -> Dict[str, float]:
        """Evaluate how well the model expresses confidence levels."""
        scores = []
        
        for response in responses:
            # Use zero-shot classification to detect uncertainty markers
            uncertainty_markers = [
                "I'm not sure",
                "I think",
                "probably",
                "might",
                "could",
                "possibly",
                "seems",
                "appears"
            ]
            
            result = self.zero_shot(
                response,
                uncertainty_markers,
                multi_label=True
            )
            
            # Calculate uncertainty score based on confidence in uncertainty markers
            uncertainty_score = np.mean(result['scores'])
            scores.append(uncertainty_score)
        
        return {
            "mean_uncertainty": np.mean(scores),
            "std_uncertainty": np.std(scores)
        }
    
    def evaluate_attribution_quality(self, responses: List[str]) -> Dict[str, float]:
        """Evaluate the quality of citations to external sources."""
        scores = []
        
        for response in responses:
            # Extract citations
            citations = self._extract_citations(response)
            
            # Score each citation
            citation_scores = []
            for citation in citations:
                # Check if citation is valid
                is_valid = self._validate_citation(citation)
                
                # Check if citation is relevant
                relevance = self._check_citation_relevance(citation, response)
                
                # Combine scores
                citation_score = (is_valid + relevance) / 2
                citation_scores.append(citation_score)
            
            if citation_scores:
                scores.append(np.mean(citation_scores))
        
        return {
            "mean_attribution": np.mean(scores),
            "std_attribution": np.std(scores)
        }
    
    def run_ablation_studies(self, responses: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
        """Run ablation studies to test contribution of each component."""
        results = {}
        
        # Full evaluation
        results['full'] = self.evaluate_all(responses, references)
        
        # Without factual accuracy
        results['no_factual'] = self.evaluate_all(responses, references, skip_factual=True)
        
        # Without logical consistency
        results['no_logical'] = self.evaluate_all(responses, references, skip_logical=True)
        
        # Without uncertainty expression
        results['no_uncertainty'] = self.evaluate_all(responses, references, skip_uncertainty=True)
        
        # Without attribution quality
        results['no_attribution'] = self.evaluate_all(responses, references, skip_attribution=True)
        
        return results
    
    def evaluate_all(self, responses: List[str], references: List[str],
                    skip_factual: bool = False,
                    skip_logical: bool = False,
                    skip_uncertainty: bool = False,
                    skip_attribution: bool = False) -> Dict[str, float]:
        """Run comprehensive evaluation across all dimensions."""
        results = {}
        
        if not skip_factual:
            results.update(self.evaluate_factual_accuracy(responses, references))
        
        if not skip_logical:
            results.update(self.evaluate_logical_consistency(responses))
        
        if not skip_uncertainty:
            results.update(self.evaluate_uncertainty_expression(responses))
        
        if not skip_attribution:
            results.update(self.evaluate_attribution_quality(responses))
        
        return results
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simple implementation - split by sentences and filter
        sentences = self._split_into_sentences(text)
        claims = []
        
        for sentence in sentences:
            # Basic filtering for factual statements
            if any(word in sentence.lower() for word in ["is", "was", "are", "were", "has", "have", "had"]):
                claims.append(sentence)
        
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple implementation - split by periods
        return [s.strip() for s in text.split('.') if s.strip()]
    
    def _verify_wikipedia(self, claim: str) -> bool:
        """Verify claim against Wikipedia."""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(claim)
            if not search_results:
                return False
            
            # Get first result
            page = wikipedia.page(search_results[0])
            
            # Check if claim appears in summary
            return claim.lower() in page.summary.lower()
        except:
            return False
    
    def _verify_wikidata(self, claim: str) -> bool:
        """Verify claim against Wikidata using SPARQL queries."""
        try:
            # Extract key entities and relations from the claim
            doc = self.nlp(claim)
            entities = [ent.text for ent in doc.ents]
            
            if not entities:
                return False
            
            # Construct SPARQL query based on the claim structure
            # This is a simplified example - you might want to make it more sophisticated
            query = f"""
            SELECT DISTINCT ?item WHERE {{
                ?item rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE("{entities[0]}")))
            }}
            LIMIT 1
            """
            
            # Execute SPARQL query
            response = requests.get(
                self.wikidata_endpoint,
                params={"query": query, "format": "json"}
            )
            
            if response.status_code == 200:
                results = response.json()
                return len(results.get("results", {}).get("bindings", [])) > 0
            
            return False
            
        except Exception as e:
            print(f"Error in Wikidata verification: {str(e)}")
            return False
    
    def _verify_semantic_scholar(self, claim: str) -> bool:
        """Verify claim against Semantic Scholar database."""
        try:
            # Extract key terms from the claim
            doc = self.nlp(claim)
            key_terms = [token.text for token in doc if not token.is_stop and token.is_alpha]
            
            if not key_terms:
                return False
            
            # Search for papers containing the key terms
            search_query = " ".join(key_terms[:3])  # Use top 3 key terms
            response = requests.get(
                f"{self.semantic_scholar_api}/paper/search",
                params={"query": search_query, "limit": 5}
            )
            
            if response.status_code == 200:
                papers = response.json().get("data", [])
                
                # Check if any paper's abstract contains the claim
                for paper in papers:
                    if "abstract" in paper:
                        # Use sentence transformer to check semantic similarity
                        claim_embedding = self.sentence_transformer.encode(claim)
                        abstract_embedding = self.sentence_transformer.encode(paper["abstract"])
                        
                        similarity = np.dot(claim_embedding, abstract_embedding) / (
                            np.linalg.norm(claim_embedding) * np.linalg.norm(abstract_embedding)
                        )
                        
                        if similarity > 0.7:  # Threshold for semantic similarity
                            return True
                
            return False
            
        except Exception as e:
            print(f"Error in Semantic Scholar verification: {str(e)}")
            return False
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text."""
        # Look for citation patterns like [1], (Smith et al., 2020), etc.
        citations = []
        
        # Match [1], [2], etc.
        import re
        bracket_citations = re.findall(r'\[(\d+)\]', text)
        citations.extend(bracket_citations)
        
        # Match (Author et al., Year)
        author_citations = re.findall(r'\(([^)]+)\)', text)
        citations.extend(author_citations)
        
        return citations
    
    def _validate_citation(self, citation: str) -> float:
        """Validate if a citation is real and accessible using multiple databases."""
        try:
            # Parse the citation
            citation_type = self._identify_citation_type(citation)
            
            if citation_type == "doi":
                # Check DOI in Crossref
                doi = self._extract_doi(citation)
                if doi:
                    response = requests.get(f"{self.crossref_api}/{doi}")
                    if response.status_code == 200:
                        return 1.0
            
            elif citation_type == "semantic_scholar":
                # Check in Semantic Scholar
                paper_id = self._extract_semantic_scholar_id(citation)
                if paper_id:
                    response = requests.get(f"{self.semantic_scholar_citations}/{paper_id}")
                    if response.status_code == 200:
                        return 1.0
            
            elif citation_type == "arxiv":
                # Check arXiv paper
                arxiv_id = self._extract_arxiv_id(citation)
                if arxiv_id:
                    response = requests.get(f"https://arxiv.org/abs/{arxiv_id}")
                    if response.status_code == 200:
                        return 1.0
            
            return 0.0
            
        except Exception as e:
            print(f"Error in citation validation: {str(e)}")
            return 0.0
    
    def _identify_citation_type(self, citation: str) -> str:
        """Identify the type of citation."""
        # DOI pattern
        if re.search(r'10\.\d{4,9}/[-._;()/:\w]+', citation):
            return "doi"
        
        # Semantic Scholar ID pattern
        if re.search(r'[a-zA-Z0-9]{40}', citation):
            return "semantic_scholar"
        
        # arXiv pattern
        if re.search(r'\d{4}\.\d{4,5}', citation):
            return "arxiv"
        
        return "unknown"
    
    def _extract_doi(self, citation: str) -> str:
        """Extract DOI from citation."""
        doi_match = re.search(r'10\.\d{4,9}/[-._;()/:\w]+', citation)
        return doi_match.group(0) if doi_match else None
    
    def _extract_semantic_scholar_id(self, citation: str) -> str:
        """Extract Semantic Scholar ID from citation."""
        id_match = re.search(r'[a-zA-Z0-9]{40}', citation)
        return id_match.group(0) if id_match else None
    
    def _extract_arxiv_id(self, citation: str) -> str:
        """Extract arXiv ID from citation."""
        arxiv_match = re.search(r'\d{4}\.\d{4,5}', citation)
        return arxiv_match.group(0) if arxiv_match else None
    
    def _check_citation_relevance(self, citation: str, context: str) -> float:
        """Check if citation is relevant to the surrounding text using semantic similarity."""
        try:
            # Extract the main content from the citation
            citation_content = self._extract_citation_content(citation)
            if not citation_content:
                return 0.0
            
            # Calculate semantic similarity
            context_embedding = self.sentence_transformer.encode(context)
            citation_embedding = self.sentence_transformer.encode(citation_content)
            
            similarity = np.dot(context_embedding, citation_embedding) / (
                np.linalg.norm(context_embedding) * np.linalg.norm(citation_embedding)
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error in citation relevance check: {str(e)}")
            return 0.0
    
    def _extract_citation_content(self, citation: str) -> str:
        """Extract relevant content from citation using appropriate API."""
        try:
            citation_type = self._identify_citation_type(citation)
            
            if citation_type == "doi":
                doi = self._extract_doi(citation)
                if doi:
                    response = requests.get(f"{self.crossref_api}/{doi}")
                    if response.status_code == 200:
                        data = response.json()
                        return data.get("message", {}).get("abstract", "")
            
            elif citation_type == "semantic_scholar":
                paper_id = self._extract_semantic_scholar_id(citation)
                if paper_id:
                    response = requests.get(f"{self.semantic_scholar_citations}/{paper_id}")
                    if response.status_code == 200:
                        data = response.json()
                        return data.get("abstract", "")
            
            elif citation_type == "arxiv":
                arxiv_id = self._extract_arxiv_id(citation)
                if arxiv_id:
                    response = requests.get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'xml')
                        return soup.find('summary').text if soup.find('summary') else ""
            
            return ""
            
        except Exception as e:
            print(f"Error in citation content extraction: {str(e)}")
            return ""

def main():
    """Main function to demonstrate usage."""
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Example responses and references
    responses = [
        "The capital of France is Paris. This is supported by historical records.",
        "I think the population of Tokyo is around 37 million people, but I'm not entirely certain.",
        "According to recent studies [1], climate change is accelerating."
    ]
    
    references = [
        "Paris is the capital of France.",
        "Tokyo's population is approximately 37.4 million people.",
        "Recent climate studies show accelerating global warming trends."
    ]
    
    # Run full evaluation
    print("Running full evaluation...")
    results = evaluator.evaluate_all(responses, references)
    print("\nFull evaluation results:")
    print(json.dumps(results, indent=2))
    
    # Run ablation studies
    print("\nRunning ablation studies...")
    ablation_results = evaluator.run_ablation_studies(responses, references)
    print("\nAblation study results:")
    print(json.dumps(ablation_results, indent=2))

if __name__ == "__main__":
    main() 

    # to test


    