import numpy as np
from utils.helper_functions import rank_list
from utils.explanation import AttributionExplanation, SelectionExplanation
from scipy.stats import kendalltau

try:
    from sharp import ShaRP
    SHARP_AVAILABLE = True
except ImportError:
    SHARP_AVAILABLE = False
    print("Warning: xai-sharp package not installed. Please install it with: pip install xai-sharp")


class RankingSharp:
    """
    Wrapper class for ShaRP (Shapley for Rankings and Preferences) explainer.
    ShaRP explains rankings with Shapley values.
    
    Memory-efficient implementation that creates ShaRP instance once and reuses it.
    """
    
    def __init__(
        self,
        background_data,
        original_model,
        explanation_size=3,
        name="rankingsharp",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
        sharp_sample_size=None,  # ShaRP's internal sample_size (None = use all, but can be memory-intensive)
        max_background_samples=None,  # Limit background data for fitting (None = use all)
        max_docs_per_query=None,  # Limit documents processed per query (None = process all)
        n_jobs=1,  # Reduce parallelism to save memory
    ):
        """
        Initialize RankingSharp explainer.
        
        Args:
            background_data: Background data for ShaRP fitting
            original_model: Model prediction function
            explanation_size: Number of top features to select
            name: Name of the explainer
            rank_similarity_coefficient: Similarity function (not used by ShaRP, kept for compatibility)
            sharp_sample_size: ShaRP's sample_size parameter (None = auto/use all, can be memory-intensive)
                               Controls number of perturbations for Shapley value computation
            max_background_samples: Max background samples for fitting (None = use all)
                                    Only needed if background_data is very large (>1000 samples)
            max_docs_per_query: Max documents to process per query (None = process all)
                                Only needed if queries have many documents (>20)
            n_jobs: Number of parallel jobs (1 = sequential, saves memory)
        """
        if not SHARP_AVAILABLE:
            raise ImportError(
                "xai-sharp package is required. Install it with: pip install xai-sharp"
            )
        
        self.background_data = background_data
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.name = name
        self.rank_similarity_coefficient = rank_similarity_coefficient
        self.sharp_sample_size = sharp_sample_size
        self.max_background_samples = max_background_samples
        self.max_docs_per_query = max_docs_per_query
        self.n_jobs = n_jobs
        
        self.feature_shape = np.shape(background_data[0])
        self.num_features = len(background_data[0])
        
        # Only sample background data if it's very large (typically it's already ~100 samples)
        # This is OPTIONAL - only needed if background_data is huge
        if max_background_samples is not None and len(background_data) > max_background_samples:
            # Sample background data to reduce memory
            indices = np.random.choice(len(background_data), size=max_background_samples, replace=False)
            self.background_data_fit = background_data[indices]
            print(f"Warning: Sampling background data from {len(background_data)} to {max_background_samples} samples")
        else:
            # Use all background data (typically already small, ~100 samples)
            self.background_data_fit = background_data
        
        # Initialize ShaRP ONCE in __init__ and reuse it
        # This avoids creating new instances for each query
        self.sharp_explainer = self._initialize_sharp()
        
        self.feature_attribution_explanation = None
        self.feature_selection_explanation = None
    
    def _initialize_sharp(self):
        """
        Initialize ShaRP explainer once and fit it on background data.
        
        This is the CRITICAL fix - creating ShaRP once instead of per-query prevents OOM.
        """
        try:
            # ShaRP parameters:
            # - sample_size: Number of perturbations for Shapley computation
            #   None = use all (most accurate but memory-intensive)
            #   Integer = limit perturbations (faster, less memory, slightly less accurate)
            sharp = ShaRP(
                qoi="rank",
                target_function=self._score_function,
                measure="shapley",
                sample_size=self.sharp_sample_size,  # None = use all (best accuracy)
                replace=False,
                random_state=42,  # Fixed seed for reproducibility
                n_jobs=self.n_jobs,  # 1 = sequential (saves memory)
                verbose=0,  # Reduce output
            )
            
            # Fit once on background data (CRITICAL: do this once, not per-query)
            sharp.fit(self.background_data_fit)
            
            return sharp
        except Exception as e:
            print(f"Warning: Failed to initialize ShaRP: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _score_function(self, X):
        """
        Score function for ShaRP.
        Takes feature matrix X and returns scores for each row.
        This is used by ShaRP to determine rankings.
        """
        # X is expected to be shape (n_samples, n_features)
        predictions = self.original_model(X)
        return predictions
    
    def get_query_explanation(self, query_features, query_id=""):
        """
        Get feature attribution explanation for a query using ShaRP.
        
        Args:
            query_features: numpy array of feature vectors for documents in the query
            query_id: optional query identifier
            
        Returns:
            tuple: (feature_selection, feature_attributes)
        """
        # Check if ShaRP was initialized successfully
        if self.sharp_explainer is None:
            print(f"Warning: ShaRP not initialized, returning zero attributions for query {query_id}")
            return self._get_fallback_explanation(query_id)
        
        try:
            # Use the pre-initialized and fitted ShaRP explainer
            sharp = self.sharp_explainer
            
            # Get feature attributions for the query
            # ShaRP's individual() method returns attributions for a single instance
            if len(query_features) > 0:
                # Process documents in the query
                # OPTIONAL: Limit documents if max_docs_per_query is set (for very large queries)
                if self.max_docs_per_query is not None and len(query_features) > self.max_docs_per_query:
                    # Sample documents to reduce memory (only if query is very large)
                    indices = np.linspace(0, len(query_features)-1, self.max_docs_per_query, dtype=int)
                    docs_to_process = query_features[indices]
                    print(f"Warning: Processing {self.max_docs_per_query} out of {len(query_features)} documents for query {query_id}")
                else:
                    # Process all documents (best accuracy)
                    docs_to_process = query_features
                
                # Get attributions for each document
                attributions_list = []
                for doc_idx, doc_features in enumerate(docs_to_process):
                    # individual() expects a single instance (1D array)
                    try:
                        # Use background_data_fit (which may be sampled if max_background_samples was set)
                        doc_attributions = sharp.individual(
                            doc_features, 
                            self.background_data_fit
                        )
                        attributions_list.append(doc_attributions)
                    except Exception as e:
                        print(f"Warning: ShaRP.individual() failed for document {doc_idx} in query {query_id}: {e}")
                        # Use zero attributions for this document
                        attributions_list.append(np.zeros(self.num_features))
                
                # Average attributions across all processed documents
                if attributions_list:
                    shapley_values = np.mean(attributions_list, axis=0)
                else:
                    shapley_values = np.zeros(self.num_features)
            else:
                shapley_values = np.zeros(self.num_features)
            
            # Convert ShaRP output to our format
            # ShaRP typically returns a dictionary or array of feature attributions
            if isinstance(shapley_values, dict):
                exp_dict = {int(k): float(v) for k, v in shapley_values.items()}
            elif isinstance(shapley_values, (np.ndarray, list)):
                exp_dict = {i + 1: float(shapley_values[i]) for i in range(len(shapley_values))}
            else:
                # Fallback: try to extract values
                exp_dict = {i + 1: 0.0 for i in range(self.num_features)}
                if hasattr(shapley_values, 'values'):
                    values = shapley_values.values
                    for i in range(min(len(values), self.num_features)):
                        exp_dict[i + 1] = float(values[i])
            
            # Sort by attribution value (descending)
            exp_dict = sorted(exp_dict.items(), key=lambda item: item[1], reverse=True)
            
            # Create AttributionExplanation
            feature_attributes = AttributionExplanation(
                explanation=exp_dict,
                num_features=self.num_features,
                query_id=query_id
            )
            
            # Create SelectionExplanation (top k features)
            feature_selection = SelectionExplanation(
                [exp_dict[i][0] for i in range(min(self.explanation_size, len(exp_dict)))],
                num_features=self.num_features,
                query_id=query_id
            )
            
            return feature_selection, feature_attributes
            
        except Exception as e:
            print(f"Error in ShaRP explanation for query {query_id}: {e}")
            print("Falling back to zero attributions.")
            import traceback
            traceback.print_exc()
            return self._get_fallback_explanation(query_id)
    
    def _get_fallback_explanation(self, query_id=""):
        """Return zero attributions as fallback."""
        exp_dict = [(i + 1, 0.0) for i in range(self.num_features)]
        feature_attributes = AttributionExplanation(
            explanation=exp_dict,
            num_features=self.num_features,
            query_id=query_id
        )
        feature_selection = SelectionExplanation(
            list(range(1, self.explanation_size + 1)),
            num_features=self.num_features,
            query_id=query_id
        )
        return feature_selection, feature_attributes

