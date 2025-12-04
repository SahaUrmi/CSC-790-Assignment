# from models.mamba import HAR_Mamba
# from models.itransformer import HAR_iTransformer


# def get_model(arg):

#     if arg.dataset_name ==  'har70+':    
#         num_features = 6      
#         num_classes = 7 
        
#     if arg.model_name == 'mamba':   
#         hidden_dim = arg.hidden_dim    
#         # Instantiate model
#         model = HAR_Mamba(input_dim=num_features, hidden_dim=hidden_dim, num_classes=num_classes)
#         return model
    
#     if arg.model_name == 'itransformer':
#         hidden_dim = arg.hidden_dim    
#         # Instantiate model
#         model = iTransformer.Model(arg)
#         return model

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mamba import HAR_Mamba
from models.itransformer import HAR_iTransformer

def get_model(arg):
    # Set dataset-specific defaults
    if arg.dataset_name == 'har70+':
        num_features = arg.no_features if hasattr(arg, 'no_features') else 6
        num_classes = arg.no_classes if hasattr(arg, 'no_classes') else 7
    else:
        raise ValueError(f"Unknown dataset: {arg.dataset_name}")

    # Select model
    if arg.model_name == 'mamba':
        hidden_dim = arg.hidden_dim
        return HAR_Mamba(input_dim=num_features, hidden_dim=hidden_dim, num_classes=num_classes)

    elif arg.model_name == 'itransformer':
        return HAR_iTransformer(arg)

    else:
        raise ValueError(f"Unknown model: {arg.model_name}")

    