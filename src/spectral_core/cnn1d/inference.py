import argparse
import logging
import numpy as np
import torch
import os

from models import Lightning1DCNNModel

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = '/home/mykola/projects/ftir-hba1c-prediction/data/cnn/checkpoints/last.ckpt'
DEFAULT_DATA_FOLDER = '/home/mykola/projects/ftir-hba1c-prediction/data/cnn'
DEFAULT_OUTPUT_PATH = '/home/mykola/projects/ftir-hba1c-prediction/data/pred.npy'

def load_data(folder, filename):
    file_path = os.path.join(folder, filename)
    return np.load(file_path).astype(np.float32)


def main(
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    data_folder=DEFAULT_DATA_FOLDER,
    output_path=DEFAULT_OUTPUT_PATH
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = Lightning1DCNNModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    
    X_test = load_data(data_folder, "X_test.npy")
    y_test = load_data(data_folder, "y_test.npy")
    logger.info(f"Test data loaded: {X_test.shape[0]} samples")
    
    X_test_tensor = torch.tensor(X_test, device=device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    predictions_np = predictions.detach().cpu().numpy().reshape(-1)
    
    results = np.stack([predictions_np, y_test], axis=1)
    np.save(output_path, results)
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Predictions shape: {results.shape}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Run inference with trained CNN model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f'Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})'
    )
    parser.add_argument(
        '--data-folder',
        type=str,
        default=DEFAULT_DATA_FOLDER,
        help=f'Path to folder containing test data (default: {DEFAULT_DATA_FOLDER})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Path to save predictions (default: {DEFAULT_OUTPUT_PATH})'
    )
    
    args = parser.parse_args()
    
    main(
        checkpoint_path=args.checkpoint,
        data_folder=args.data_folder,
        output_path=args.output
    )
