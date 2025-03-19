import os
import torch
from pathlib import Path

def save_model(
    model,
    save_path,
    optimizer=None,
    epoch=None,
    best_result=None,
    metadata=None,
    is_best=False,
    use_safetensors=False,
    distributed=False,
    local_rank=0,
    world_size=1,
):
    """
    Standardized function to save a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model to save
        save_path (str or Path): Path where the model will be saved
        optimizer (torch.optim.Optimizer, optional): Optimizer state to save
        epoch (int, optional): Current epoch number
        best_result (dict, optional): Best performance metrics
        metadata (dict, optional): Additional information to save
        is_best (bool): Whether this is the best model so far (saves to a separate file)
        use_safetensors (bool): Whether to use safetensors format (if available)
        distributed (bool): Whether in distributed training environment
        local_rank (int): Local process rank in distributed training
        world_size (int): Total number of processes in distributed training
    
    Returns:
        str: Path to the saved model file
    """
    # Only save from the main process in distributed training
    if distributed and local_rank != 0:
        return None
    
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    
    # Prepare the state dict - unwrap from DataParallel/DDP if needed
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Prepare the checkpoint dict with all the information we want to save
    checkpoint = {
        'model': model_state_dict,
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if best_result is not None:
        checkpoint['best_result'] = best_result
    
    if metadata is not None:
        checkpoint.update(metadata)
    
    # Try using safetensors if requested
    if use_safetensors:
        try:
            import safetensors
            from safetensors.torch import save_file
            
            # safetensors only supports tensors, so we need to extract them
            tensors_dict = {k: v for k, v in model_state_dict.items() 
                           if isinstance(v, torch.Tensor)}
            
            # Save tensors with safetensors
            save_file(tensors_dict, str(save_path) + ".safetensors")
            
            # Save other data with PyTorch
            other_dict = {k: checkpoint for k, v in checkpoint.items() 
                         if k != 'model' or not isinstance(v, torch.Tensor)}
            if other_dict:
                torch.save(other_dict, str(save_path) + ".meta.pt")
                
            print(f"Model saved with safetensors at {save_path}.safetensors")
            path_saved = str(save_path) + ".safetensors"
        except ImportError:
            print("safetensors not available, falling back to PyTorch saving")
            torch.save(checkpoint, save_path)
            path_saved = str(save_path)
    else:
        # Standard PyTorch saving
        torch.save(checkpoint, save_path)
        path_saved = str(save_path)
    
    # If this is the best model, also save it to a dedicated path
    if is_best:
        best_path = save_path.parent / "best_model.pth"
        if use_safetensors:
            try:
                import shutil
                shutil.copy2(str(save_path) + ".safetensors", str(best_path) + ".safetensors")
                if os.path.exists(str(save_path) + ".meta.pt"):
                    shutil.copy2(str(save_path) + ".meta.pt", str(best_path) + ".meta.pt")
                path_saved = str(best_path) + ".safetensors"
            except (ImportError, FileNotFoundError):
                torch.save(checkpoint, best_path)
                path_saved = str(best_path)
        else:
            torch.save(checkpoint, best_path)
            path_saved = str(best_path)
    
    return path_saved
