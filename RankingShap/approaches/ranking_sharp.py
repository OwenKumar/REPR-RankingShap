import numpy as np
from utils.helper_functions import rank_list
from utils.explanation import AttributionExplanation, SelectionExplanation
from scipy.stats import kendalltau

# Import ShaRP library
# The xai-sharp package provides the 'sharp' module, which exports 'ShaRP' class
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
    """
    
    def __init__(
        self,
        background_data,
        original_model,
        explanation_size=3,
        name="rankingsharp",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    ):
        if not SHARP_AVAILABLE:
            raise ImportError(
                "xai-sharp package is required. Install it with: pip install xai-sharp"
            )
        
        self.background_data = background_data
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.name = name
        self.rank_similarity_coefficient = rank_similarity_coefficient
        
        self.feature_shape = np.shape(background_data[0])
        self.num_features = len(background_data[0])
        
        # Initialize ShaRP explainer
        # ShaRP uses qoi="rank" to explain feature contributions to rankings
        # target_function is the scoring function that determines rankings
        try:
            print(f"[RankingSharp.__init__] Initializing ShaRP explainer...", flush=True)
            self.sharp = ShaRP(
                qoi="rank",
                target_function=self._score_function,
                measure="shapley",
                sample_size=10,  
                replace=False,
                random_state=None,
            )
            
            print(f"[RankingSharp.__init__] Fitting ShaRP on background data (shape: {np.shape(background_data)})...", flush=True)
            # Fit ShaRP on background data
            self.sharp.fit(self.background_data)
            print(f"[RankingSharp.__init__] ShaRP initialization complete!", flush=True)
        except Exception as e:  
            print(f"[RankingSharp.__init__] ERROR during initialization: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        
    
    def _score_function(self, X):
        """
        Score function for ShaRP.
        Takes feature matrix X and returns scores for each row.
        This is used by ShaRP to determine rankings.
        """
        # X is expected to be shape (n_samples, n_features)
        # Get predictions from the model
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
        print(f"[RankingSharp.get_query_explanation] Called for query {query_id} with {len(query_features)} documents", flush=True)
        try:
            print(f"[RankingSharp.get_query_explanation] Starting Sharp for query {query_id}", flush=True)
            # Get feature attributions for the query
            # ShaRP's individual() method returns attributions for a single instance
            # We'll aggregate attributions across all documents in the query
            # For now, we'll use the first document or average across documents
            if len(query_features) > 0:
                # Get attributions for each document and average them
                attributions_list = []
                for doc_features in query_features:
                    # individual() expects a single instance (1D array)
                    print(doc_features)
                    doc_attributions = self.sharp.individual(doc_features, self.background_data, sample_size = 10)
                    attributions_list.append(doc_attributions)
                
                # Average attributions across all documents in the query
                shapley_values = np.mean(attributions_list, axis=0)
            else:
                shapley_values = np.zeros(self.num_features)
            
            print("Starting reformatting")
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
            

            print("Starting sorting")
            # Sort by attribution value (descending)
            exp_dict = sorted(exp_dict.items(), key=lambda item: item[1], reverse=True)
            
            print("Starting attribution")
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
            
            print(f"[RankingSharp.get_query_explanation] Returning results for query {query_id}", flush=True)
            
            return feature_selection, feature_attributes
            
        except Exception as e:
            print(f"[RankingSharp.get_query_explanation] ERROR for query {query_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("[RankingSharp.get_query_explanation] Falling back to zero attributions.", flush=True)
            
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

