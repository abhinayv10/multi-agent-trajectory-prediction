import sys
import os
sys.path.append(os.path.abspath('.'))
from data.dataset import DatasetFactory

config = {
    'dataset': {
        'type': 'nuscenes',
        'data_root': './',
        'past_steps': 8,
        'future_steps': 12,
        'num_polylines': 6,
        'polyline_points': 20,
        'num_agents_max': 10
    }
}

print("Initializing dataset...")
dataset = DatasetFactory(config, split='train')
print("Dataset size:", len(dataset))
sample = dataset[0]
print("Past shape:", sample['past'].shape)
print("Future shape:", sample['future'].shape)
print("Mask shape:", sample['mask'].shape)
print("Polylines shape:", sample['map_polylines'].shape)
print("Test completed successfully.")
