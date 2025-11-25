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
        # ShaRP typically needs a ranking function and feature names
        self.sharp_explainer = None
        self.feature_attribution_explanation = None
        self.feature_selection_explanation = None
    
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
        try:
            # Initialize ShaRP
            # ShaRP uses qoi="rank" to explain feature contributions to rankings
            # target_function is the scoring function that determines rankings
            sharp = ShaRP(
                qoi="rank",
                target_function=self._score_function,
                measure="shapley",
                sample_size=None,  # Use all samples
                replace=False,
                random_state=None,
            )
            
            # Fit ShaRP on background data
            sharp.fit(self.background_data)
            
            # Get feature attributions for the query
            # ShaRP's individual() method returns attributions for a single instance
            # We'll aggregate attributions across all documents in the query
            # For now, we'll use the first document or average across documents
            if len(query_features) > 0:
                # Get attributions for each document and average them
                attributions_list = []
                for doc_features in query_features:
                    # individual() expects a single instance (1D array)
                    doc_attributions = sharp.individual(doc_features, self.background_data)
                    attributions_list.append(doc_attributions)
                
                # Average attributions across all documents in the query
                shapley_values = np.mean(attributions_list, axis=0)
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
            print(f"Error in ShaRP explanation: {e}")
            print("Falling back to zero attributions. Please check ShaRP API documentation.")
            
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

