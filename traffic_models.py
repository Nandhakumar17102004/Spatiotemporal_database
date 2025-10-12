import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: prophet not installed. Install with: pip install prophet")

# ==================== SYNTHETIC DATA GENERATOR ====================
class SyntheticTrafficGenerator:
    """Generate realistic synthetic intersection traffic data for PG-STGNN training"""
    
    def __init__(self, num_approaches=12, days=7, interval_minutes=5):
        self.num_approaches = num_approaches
        self.days = days
        self.interval_minutes = interval_minutes
        
    def generate_intersection_data(self):
        """Generate complete synthetic intersection dataset"""
        approaches = self._define_intersection_approaches()
        time_series = self._generate_time_series_data(approaches)
        targets = self._create_targets(time_series, approaches)
        
        return {
            'approaches': approaches,
            'time_series': time_series,
            'targets': targets
        }
    
    def _define_intersection_approaches(self):
        """Define the intersection geometry and approaches"""
        approaches = []
        directions = ['north', 'south', 'east', 'west']
        movements = ['through', 'left', 'right']
        
        for direction in directions:
            for movement in movements:
                approach = {
                    'id': f"{direction}_{movement}",
                    'direction': direction,
                    'movement_type': movement,
                    'lane_number': 1 if movement == 'left' else (2 if movement == 'right' else 3),
                    'num_lanes': 1 if movement in ['left', 'right'] else 2,
                    'road_type': 'arterial',
                    'longitude': np.random.uniform(116.53, 116.54),
                    'latitude': np.random.uniform(39.79, 39.80),
                    'speed_limit': 60,
                    'capacity': 1200 if movement == 'through' else (800 if movement == 'left' else 600),
                    'saturation_flow': 1800 if movement == 'through' else (1200 if movement == 'left' else 900),
                    'free_flow_time': np.random.uniform(25, 35),
                    'green_time': np.random.randint(30, 45),
                    'cycle_time': 120
                }
                approaches.append(approach)
        return approaches
    
    def _generate_time_series_data(self, approaches):
        """Generate realistic time series traffic data"""
        time_series = []
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        total_intervals = self.days * 24 * (60 // self.interval_minutes)
        
        for i in range(total_intervals):
            current_time = start_time + timedelta(minutes=i * self.interval_minutes)
            hour = current_time.hour
            weekday = current_time.weekday() < 5
            day_of_week = current_time.weekday()
            
            flows = {}
            for approach in approaches:
                base_flow = self._get_base_traffic_pattern(
                    approach['movement_type'], hour, weekday, day_of_week
                )
                
                noise = np.random.normal(0, 0.15 * base_flow)
                if i > 0:
                    prev_flow = time_series[i-1]['flows'][approach['id']]
                    flow = 0.7 * prev_flow + 0.3 * base_flow + noise
                else:
                    flow = base_flow + noise
                
                flows[approach['id']] = max(0, int(flow))
            
            time_series.append({
                'timestamp': current_time,
                'flows': flows,
                'hour': hour,
                'minute': current_time.minute,
                'is_weekday': weekday,
                'day_of_week': day_of_week
            })
        
        return time_series
    
    def _create_targets(self, time_series, approaches):
        """Create targets for prediction (next time step)"""
        targets = []
        for i in range(len(time_series) - 1):
            target_flows = {}
            for approach in approaches:
                target_flow = time_series[i + 1]['flows'][approach['id']]
                target_flows[approach['id']] = target_flow
            targets.append(target_flows)
        return targets
    
    def _get_base_traffic_pattern(self, movement_type, hour, weekday, day_of_week):
        """Generate realistic base traffic patterns"""
        if 7 <= hour < 9:
            base_flow = 45 if weekday else 25
        elif 16 <= hour < 19:
            base_flow = 50 if weekday else 35
        elif hour >= 22 or hour < 6:
            base_flow = 8
        else:
            base_flow = 30 if weekday else 20
        
        if movement_type == 'through':
            base_flow *= 1.3
        elif movement_type == 'left':
            base_flow *= 0.9
        elif movement_type == 'right':
            base_flow *= 0.7
            
        if day_of_week == 4 and 16 <= hour < 19:
            base_flow *= 1.2
            
        return base_flow

# ==================== NETWORK GRAPH BUILDER ====================
class NetworkGraphBuilder:
    """Step 1: Fixed Network Graph Construction"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def construct_network_graph(self, intersection_data: Dict) -> Dict:
        """Construct intersection graph with all required keys"""
        graph = {
            'nodes': [],
            'edges': [],
            'node_features': [],
            'edge_features': [],
            'adjacency_matrix': None,
            'node_mapping': {},
            'edge_index': None,
            'edge_attr': None
        }
        
        for idx, approach in enumerate(intersection_data['approaches']):
            node_id = approach['id']
            node_features = self._extract_node_features(approach)
            graph['nodes'].append(node_id)
            graph['node_features'].append(node_features)
            graph['node_mapping'][node_id] = idx
        
        graph['edges'] = self._establish_connectivity(intersection_data)
        graph['edge_features'] = self._extract_edge_features(graph['edges'])
        graph['adjacency_matrix'] = self._build_adjacency_matrix(len(graph['nodes']), graph['edges'])
        
        pyg_data = self._create_pyg_data(graph)
        graph['pyg_data'] = pyg_data
        graph['edge_index'] = pyg_data.edge_index
        graph['edge_attr'] = pyg_data.edge_attr
        graph['node_features'] = torch.tensor(graph['node_features'], dtype=torch.float)
        
        return graph
    
    def _extract_node_features(self, approach: Dict) -> List[float]:
        """Extract simplified node features"""
        features = [
            approach['num_lanes'] / 3.0,
            self._encode_road_type(approach['road_type']),
            approach['speed_limit'] / 80.0,
            approach['capacity'] / 2000.0,
            self._encode_movement_type(approach['movement_type']),
            self._encode_direction(approach['direction'])
        ]
        return features
    
    def _encode_road_type(self, road_type: str) -> float:
        encoding = {'highway': 1.0, 'arterial': 0.7, 'collector': 0.4, 'local': 0.1}
        return encoding.get(road_type.lower(), 0.5)
    
    def _encode_movement_type(self, movement_type: str) -> float:
        encoding = {'through': 1.0, 'left': 0.6, 'right': 0.3}
        return encoding.get(movement_type.lower(), 0.5)
    
    def _encode_direction(self, direction: str) -> float:
        encoding = {'north': 0.9, 'south': 0.7, 'east': 0.5, 'west': 0.3}
        return encoding.get(direction.lower(), 0.5)
    
    def _establish_connectivity(self, intersection_data: Dict) -> List[Tuple[int, int]]:
        """Establish basic intersection connectivity"""
        edges = []
        num_approaches = len(intersection_data['approaches'])
        for i in range(num_approaches):
            j = (i + 1) % num_approaches
            edges.append((i, j))
            edges.append((j, i))
        return edges
    
    def _extract_edge_features(self, edges: List[Tuple]) -> List[List[float]]:
        """Extract basic edge features"""
        edge_features = []
        for src, dst in edges:
            features = [1.0, 0.5, 0.3]
            edge_features.append(features)
        return edge_features
    
    def _build_adjacency_matrix(self, num_nodes: int, edges: List[Tuple]) -> np.ndarray:
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, j in edges:
            adj_matrix[i, j] = 1
        return adj_matrix
    
    def _create_pyg_data(self, graph: Dict) -> Data:
        node_features = torch.tensor(graph['node_features'], dtype=torch.float)
        edge_index = torch.tensor(graph['edges'], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(graph['edge_features'], dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# ==================== PHYSICS TRAFFIC EMBEDDER ====================
class PhysicsTrafficEmbedder:
    """Step 2: Simplified Physics Traffic Performance Embedding"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def embed_traffic_performance(self, intersection_data: Dict, graph_data: Dict) -> Dict:
        """Embed simplified physics-based features"""
        physics_features = {
            'flow_characteristics': self._analyze_flow_characteristics(intersection_data),
            'vc_ratios': self._calculate_vc_ratios(intersection_data),
            'queue_dynamics': self._model_queue_dynamics(intersection_data),
        }
        physics_features['tensor_features'] = self._convert_to_tensors(physics_features, graph_data)
        return physics_features
    
    def _analyze_flow_characteristics(self, data: Dict) -> Dict:
        """Analyze basic flow characteristics"""
        characteristics = {}
        for approach in data['approaches']:
            approach_id = approach['id']
            flows = [ts['flows'][approach_id] for ts in data['time_series']]
            characteristics[approach_id] = {
                'mean_flow': np.mean(flows) / 100.0,
                'std_flow': np.std(flows) / 100.0,
                'peak_flow': np.max(flows) / 100.0,
            }
        return characteristics
    
    def _calculate_vc_ratios(self, data: Dict) -> Dict:
        """Calculate volume-to-capacity ratios"""
        vc_ratios = {}
        for approach in data['approaches']:
            capacity = approach['capacity']
            volumes = [ts['flows'][approach['id']] for ts in data['time_series']]
            current_volume = volumes[-1] if volumes else 0
            vc_ratios[approach['id']] = {
                'current_vc': current_volume / capacity if capacity > 0 else 0,
                'max_vc': max(volumes) / capacity if capacity > 0 else 0,
            }
        return vc_ratios
    
    def _model_queue_dynamics(self, data: Dict) -> Dict:
        """Model basic queue dynamics"""
        queue_data = {}
        for approach in data['approaches']:
            approach_id = approach['id']
            current_volume = data['time_series'][-1]['flows'][approach_id] if data['time_series'] else 0
            capacity = approach['capacity']
            vc_ratio = current_volume / capacity if capacity > 0 else 0
            queue_length = max(0, (vc_ratio - 0.8) * 10)
            queue_data[approach_id] = {
                'queue_length': queue_length / 10.0,
                'congestion_level': min(1.0, vc_ratio),
            }
        return queue_data
    
    def _convert_to_tensors(self, physics_features: Dict, graph_data: Dict) -> Dict:
        """Convert all physics features to tensors"""
        tensor_features = {}
        
        flow_tensors = []
        for node_id in graph_data['nodes']:
            if node_id in physics_features['flow_characteristics']:
                char_dict = physics_features['flow_characteristics'][node_id]
                flow_tensor = torch.tensor([
                    char_dict['mean_flow'],
                    char_dict['std_flow'],
                    char_dict['peak_flow']
                ], dtype=torch.float)
                flow_tensors.append(flow_tensor)
        tensor_features['flow_chars'] = torch.stack(flow_tensors)
        
        vc_tensors = []
        for node_id in graph_data['nodes']:
            if node_id in physics_features['vc_ratios']:
                vc_dict = physics_features['vc_ratios'][node_id]
                vc_tensor = torch.tensor([
                    vc_dict['current_vc'],
                    vc_dict['max_vc']
                ], dtype=torch.float)
                vc_tensors.append(vc_tensor)
        tensor_features['vc_ratios'] = torch.stack(vc_tensors)
        
        queue_tensors = []
        for node_id in graph_data['nodes']:
            if node_id in physics_features['queue_dynamics']:
                queue_dict = physics_features['queue_dynamics'][node_id]
                queue_tensor = torch.tensor([
                    queue_dict['queue_length'],
                    queue_dict['congestion_level']
                ], dtype=torch.float)
                queue_tensors.append(queue_tensor)
        tensor_features['queue_dynamics'] = torch.stack(queue_tensors)
        
        return tensor_features

# ==================== PG-STGNN MODEL COMPONENTS ====================
class SpatialFeatureModeling(nn.Module):
    def __init__(self, config: Dict):
        super(SpatialFeatureModeling, self).__init__()
        self.config = config
        input_dim = config.get('spatial_input_dim', 6)
        hidden_dim = config.get('hidden_dim', 64)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.get('dropout', 0.3))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class TemporalSeriesModeling(nn.Module):
    def __init__(self, config: Dict):
        super(TemporalSeriesModeling, self).__init__()
        self.config = config
        input_dim = config.get('temporal_input_dim', 6)
        hidden_dim = config.get('hidden_dim', 64)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.get('dropout', 0.3)
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, input_dim = x.shape
        node_outputs = []
        for node_idx in range(num_nodes):
            node_data = x[:, :, node_idx, :]
            lstm_out, (h_n, c_n) = self.lstm(node_data)
            node_output = self.fc(h_n[-1])
            node_outputs.append(node_output)
        return torch.stack(node_outputs, dim=1)

class PhysicsIntegrationLayer(nn.Module):
    def __init__(self, config: Dict):
        super(PhysicsIntegrationLayer, self).__init__()
        self.config = config
        hidden_dim = config.get('hidden_dim', 64)
        combined_dim = hidden_dim * 2
        
        self.flow_processor = nn.Linear(3, hidden_dim)
        self.vc_processor = nn.Linear(2, hidden_dim)
        self.queue_processor = nn.Linear(2, hidden_dim)
        
        fusion_input_dim = combined_dim + (hidden_dim * 3)
        self.fusion_layer = nn.Linear(fusion_input_dim, hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, features: torch.Tensor, physics_features: Dict) -> torch.Tensor:
        batch_size, num_nodes, combined_dim = features.shape
        
        flow_tensor = physics_features['tensor_features']['flow_chars'].unsqueeze(0).expand(batch_size, -1, -1)
        vc_tensor = physics_features['tensor_features']['vc_ratios'].unsqueeze(0).expand(batch_size, -1, -1)
        queue_tensor = physics_features['tensor_features']['queue_dynamics'].unsqueeze(0).expand(batch_size, -1, -1)
        
        flow_embed = self.flow_processor(flow_tensor)
        vc_embed = self.vc_processor(vc_tensor)
        queue_embed = self.queue_processor(queue_tensor)
        
        fused = torch.cat([features, flow_embed, vc_embed, queue_embed], dim=-1)
        physics_informed = self.activation(self.fusion_layer(fused))
        return physics_informed

class PredictionModule(nn.Module):
    def __init__(self, config: Dict):
        super(PredictionModule, self).__init__()
        self.config = config
        hidden_dim = config.get('hidden_dim', 64)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(features)

# ==================== PG-STGNN MODEL ====================
class PGSTGNNModel(nn.Module):
    """Step 3: Simplified Physics-Guided Spatio-Temporal Graph Neural Network"""
    
    def __init__(self, config: Dict):
        super(PGSTGNNModel, self).__init__()
        self.config = config
        self.spatial_model = SpatialFeatureModeling(config)
        self.temporal_model = TemporalSeriesModeling(config)
        self.physics_integration = PhysicsIntegrationLayer(config)
        self.prediction_module = PredictionModule(config)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.get('learning_rate', 0.001))
        
    def forward(self, x_spatial: torch.Tensor, edge_index: torch.Tensor, 
                x_temporal: torch.Tensor, physics_features: Dict) -> torch.Tensor:
        """Simplified forward pass with corrected dimensions"""
        batch_size = x_temporal.shape[0]
        spatial_out = self.spatial_model(x_spatial, edge_index)
        temporal_out = self.temporal_model(x_temporal)
        spatial_out_expanded = spatial_out.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([spatial_out_expanded, temporal_out], dim=-1)
        physics_informed = self.physics_integration(combined, physics_features)
        predictions = self.prediction_module(physics_informed)
        return predictions
    
    def train_model(self, train_loader, val_loader, graph_data, physics_features, epochs=50):
        """Simplified training loop"""
        train_losses = []
        val_losses = []
        
        if 'edge_index' not in graph_data:
            raise KeyError("graph_data missing 'edge_index' key")
        if 'node_features' not in graph_data:
            raise KeyError("graph_data missing 'node_features' key")
        
        for epoch in range(epochs):
            self.train()
            epoch_train_loss = 0.0
            for batch_idx, (x_temporal, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                predictions = self.forward(
                    graph_data['node_features'],
                    graph_data['edge_index'],
                    x_temporal,
                    physics_features
                )
                loss = self.criterion(predictions.squeeze(), targets)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for x_temporal, targets in val_loader:
                    predictions = self.forward(
                        graph_data['node_features'],
                        graph_data['edge_index'],
                        x_temporal,
                        physics_features
                    )
                    loss = self.criterion(predictions.squeeze(), targets)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': val_losses}

# ==================== DATA PREPROCESSING ====================
class DataPreprocessor:
    """Prepare data for PG-STGNN training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.seq_len = config.get('seq_len', 12)
        
    def prepare_training_data(self, intersection_data: Dict):
        """Prepare sequences and targets for training"""
        sequences = []
        targets = []
        
        time_series = intersection_data['time_series']
        approaches = intersection_data['approaches']
        num_nodes = len(approaches)
        
        for i in range(len(time_series) - self.seq_len - 1):
            sequence = self._create_sequence(time_series, i, approaches)
            target = self._create_target(time_series, i + self.seq_len, approaches)
            sequences.append(sequence)
            targets.append(target)
        
        if sequences:
            sequences_tensor = torch.stack(sequences)
            targets_tensor = torch.stack(targets)
            dataset = TensorDataset(sequences_tensor, targets_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            if train_size > 0 and val_size > 0:
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 32), shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 32))
                return train_loader, val_loader
        
        empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
        return DataLoader(empty_dataset), DataLoader(empty_dataset)
    
    def _create_sequence(self, time_series, start_idx, approaches):
        """Create input sequence for a given start index"""
        sequence_data = []
        for i in range(start_idx, start_idx + self.seq_len):
            time_point = time_series[i]
            features = []
            for approach in approaches:
                flow = time_point['flows'][approach['id']]
                feature_vec = [
                    flow / 100.0,
                    time_point['hour'] / 24.0,
                    float(time_point['is_weekday']),
                    time_point['day_of_week'] / 7.0,
                    np.sin(2 * np.pi * time_point['hour'] / 24.0),
                    np.cos(2 * np.pi * time_point['hour'] / 24.0)
                ]
                features.append(feature_vec)
            sequence_data.append(features)
        return torch.tensor(sequence_data, dtype=torch.float)
    
    def _create_target(self, time_series, target_idx, approaches):
        """Create target for a given time index"""
        target_point = time_series[target_idx]
        target_flows = [target_point['flows'][approach['id']] / 100.0 for approach in approaches]
        return torch.tensor(target_flows, dtype=torch.float)

# ==================== BASELINE MODELS ====================
class NaiveBaseline:
    """Naive baseline: predict same as previous timestep or seasonal pattern"""
    
    def __init__(self, method='last_value'):
        self.method = method
        self.seasonal_data = None
        
    def fit(self, train_data):
        """Store training data for seasonal prediction"""
        if self.method == 'seasonal':
            self.seasonal_data = train_data
        
    def predict(self, X_test, X_train=None):
        """
        Args:
            X_test: [n_samples, n_nodes] - used to determine output shape
            X_train: [n_train_samples, n_nodes] - training data
        Returns:
            predictions: [n_samples, n_nodes]
        """
        n_samples = X_test.shape[0]
        n_nodes = X_test.shape[1] if len(X_test.shape) > 1 else 1
        
        if self.method == 'last_value':
            # Repeat last training value for all test samples
            last_value = X_train[-1]  # Shape: [n_nodes]
            predictions = np.tile(last_value, (n_samples, 1))
            return predictions
        
        elif self.method == 'seasonal':
            # Use value from same time yesterday (288 intervals = 24 hours * 60 / 5 min)
            seasonal_period = 288
            predictions = []
            
            for i in range(n_samples):
                idx = max(0, i - seasonal_period)
                if idx < len(X_train):
                    predictions.append(X_train[idx])
                else:
                    predictions.append(X_train[-1])
            
            return np.array(predictions)

class SpatialRegression:
    """Spatial regression: use neighboring nodes to predict current node"""
    
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.models = {}
        self.scalers = {}
        
    def fit(self, X_train):
        """
        Args:
            X_train: [n_samples, n_nodes] historical traffic flows
        """
        n_nodes = X_train.shape[1]
        
        for node_id in range(n_nodes):
            neighbors = np.where(self.adjacency_matrix[node_id] > 0)[0]
            if len(neighbors) == 0:
                neighbors = np.array([node_id])
            
            X_features = []
            y_target = []
            
            for t in range(1, len(X_train)):
                features = []
                features.extend(X_train[t-1, neighbors])
                features.append(X_train[t-1, node_id])
                X_features.append(features)
                y_target.append(X_train[t, node_id])
            
            X_features = np.array(X_features)
            y_target = np.array(y_target)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)
            
            model = LinearRegression()
            model.fit(X_scaled, y_target)
            
            self.models[node_id] = model
            self.scalers[node_id] = scaler
    
    def predict(self, X_test):
        """
        Args:
            X_test: [n_samples, n_nodes]
        Returns:
            predictions: [n_samples, n_nodes]
        """
        predictions = []
        n_nodes = X_test.shape[1]
        
        for t in range(len(X_test)):
            node_predictions = []
            for node_id in range(n_nodes):
                if node_id not in self.models:
                    node_predictions.append(np.mean(X_test[t]))
                    continue
                
                neighbors = np.where(self.adjacency_matrix[node_id] > 0)[0]
                if len(neighbors) == 0:
                    neighbors = np.array([node_id])
                
                features = []
                features.extend(X_test[t-1, neighbors] if t > 0 else X_test[0, neighbors])
                features.append(X_test[t-1, node_id] if t > 0 else X_test[0, node_id])
                
                features = np.array(features).reshape(1, -1)
                features_scaled = self.scalers[node_id].transform(features)
                pred = self.models[node_id].predict(features_scaled)[0]
                node_predictions.append(max(0, pred))
            
            predictions.append(node_predictions)
        
        return np.array(predictions)

class ARIMAModel:
    """ARIMA model for each node independently"""
    
    def __init__(self, order=(1, 0, 1)):
        self.order = order
        self.models = {}
        
    def fit(self, X_train):
        """
        Args:
            X_train: [n_samples, n_nodes]
        """
        if not ARIMA_AVAILABLE:
            print("ARIMA not available - skipping")
            return
        
        n_nodes = X_train.shape[1]
        
        for node_id in range(n_nodes):
            try:
                model = ARIMA(X_train[:, node_id], order=self.order)
                fitted_model = model.fit()
                self.models[node_id] = fitted_model
            except Exception as e:
                print(f"Error fitting ARIMA for node {node_id}: {e}")
                self.models[node_id] = None
    
    def predict(self, steps, n_nodes):
        """
        Args:
            steps: number of timesteps to predict
            n_nodes: number of nodes
        Returns:
            predictions: [steps, n_nodes]
        """
        if not ARIMA_AVAILABLE:
            return np.zeros((steps, n_nodes))
        
        predictions = []
        
        for node_id in range(n_nodes):
            if node_id not in self.models or self.models[node_id] is None:
                predictions.append(np.zeros(steps))
                continue
            
            try:
                forecast = self.models[node_id].get_forecast(steps=steps)
                pred = forecast.predicted_mean.values
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting for node {node_id}: {e}")
                predictions.append(np.zeros(steps))
        
        return np.array(predictions).T

class ProphetModel:
    """Facebook Prophet model for each node"""
    
    def __init__(self):
        self.models = {}
        
    def fit(self, X_train, freq='5min'):
        """
        Args:
            X_train: [n_samples, n_nodes]
            freq: frequency of data
        """
        if not PROPHET_AVAILABLE:
            print("Prophet not available - skipping")
            return
        
        n_nodes = X_train.shape[1]
        
        for node_id in range(n_nodes):
            try:
                df = pd.DataFrame({
                    'ds': pd.date_range(start='2024-01-01', periods=len(X_train), freq=freq),
                    'y': X_train[:, node_id]
                })
                
                model = Prophet(yearly_seasonality=False, 
                               daily_seasonality=True,
                               interval_width=0.95,
                               changepoint_prior_scale=0.05)
                model.fit(df)
                self.models[node_id] = model
            except Exception as e:
                print(f"Error fitting Prophet for node {node_id}: {e}")
                self.models[node_id] = None
    
    def predict(self, steps, n_nodes, freq='5min'):
        """
        Args:
            steps: number of timesteps
            n_nodes: number of nodes
            freq: frequency
        Returns:
            predictions: [steps, n_nodes]
        """
        if not PROPHET_AVAILABLE:
            return np.zeros((steps, n_nodes))
        
        predictions = []
        
        for node_id in range(n_nodes):
            if node_id not in self.models or self.models[node_id] is None:
                predictions.append(np.zeros(steps))
                continue
            
            try:
                future = self.models[node_id].make_future_dataframe(periods=steps, freq=freq)
                forecast = self.models[node_id].predict(future)
                pred = forecast['yhat'].tail(steps).values
                pred = np.maximum(pred, 0)
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting with Prophet for node {node_id}: {e}")
                predictions.append(np.zeros(steps))
        
        return np.array(predictions).T

# ==================== EVALUATION METRICS ====================
class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    @staticmethod
    def calculate_metrics(predictions, targets):
        """
        Args:
            predictions: [n_samples, ...] or flattened
            targets: [n_samples, ...] or flattened
        Returns:
            Dict with metrics
        """
        pred_flat = np.array(predictions).flatten()
        target_flat = np.array(targets).flatten()
        
        mae = mean_absolute_error(target_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
        mape = np.mean(np.abs((target_flat - pred_flat) / (target_flat + 1e-8))) * 100
        r2 = r2_score(target_flat, pred_flat)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    @staticmethod
    def calculate_directional_accuracy(predictions, targets):
        """Percentage of correct direction predictions (up/down)"""
        pred_flat = np.array(predictions).flatten()
        target_flat = np.array(targets).flatten()
        
        pred_direction = np.diff(pred_flat) > 0
        target_direction = np.diff(target_flat) > 0
        
        accuracy = np.mean(pred_direction == target_direction) * 100
        return accuracy

# ==================== MODEL COMPARISON FRAMEWORK ====================
class ModelComparison:
    """Compare all models on same dataset"""
    
    def __init__(self, intersection_data, graph_data, pgstgnn_model, config):
        self.intersection_data = intersection_data
        self.graph_data = graph_data
        self.pgstgnn_model = pgstgnn_model
        self.config = config
        self.results = {}
        self.evaluator = ModelEvaluator()
    
    def prepare_data(self):
        """Prepare data for comparison"""
        time_series = self.intersection_data['time_series']
        approaches = self.intersection_data['approaches']
        n_nodes = len(approaches)
        
        flows = []
        for ts in time_series:
            node_flows = [ts['flows'][app['id']] for app in approaches]
            flows.append(node_flows)
        
        flows = np.array(flows)
        
        n = len(flows)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        X_train = flows[:train_size]
        X_val = flows[train_size:train_size + val_size]
        X_test = flows[train_size + val_size:]
        
        return X_train, X_val, X_test
    
    def run_comparison(self):
        """Run all models and compare"""
        print("=" * 80)
        print("TRAFFIC PREDICTION MODELS COMPARISON")
        print("=" * 80)
        
        X_train, X_val, X_test = self.prepare_data()
        print(f"\nData split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        print(f"Nodes: {X_train.shape[1]}\n")
        
        # 1. NAIVE BASELINE (Last Value)
        print("1️⃣  Training Naive Baseline (Last Value)...")
        naive_last = NaiveBaseline(method='last_value')
        naive_last.fit(X_train)
        naive_pred = naive_last.predict(X_test, X_train)
        self.results['Naive (Last Value)'] = {
            'predictions': naive_pred,
            'metrics': self.evaluator.calculate_metrics(naive_pred, X_test)
        }
        print(f"   ✓ MAE: {self.results['Naive (Last Value)']['metrics']['MAE']:.4f}\n")
        
        # 2. SEASONAL NAIVE
        print("2️⃣  Training Naive Baseline (Seasonal)...")
        naive_seasonal = NaiveBaseline(method='seasonal')
        naive_seasonal.fit(X_train)
        seasonal_pred = naive_seasonal.predict(X_test, X_train)
        self.results['Naive (Seasonal)'] = {
            'predictions': seasonal_pred,
            'metrics': self.evaluator.calculate_metrics(seasonal_pred, X_test)
        }
        print(f"   ✓ MAE: {self.results['Naive (Seasonal)']['metrics']['MAE']:.4f}\n")
        
        # 3. SPATIAL REGRESSION
        print("3️⃣  Training Spatial Regression...")
        try:
            adj_matrix = self.graph_data['adjacency_matrix']
            spatial_reg = SpatialRegression(adj_matrix)
            spatial_reg.fit(X_train)
            spatial_pred = spatial_reg.predict(X_test)
            self.results['Spatial Regression'] = {
                'predictions': spatial_pred,
                'metrics': self.evaluator.calculate_metrics(spatial_pred, X_test)
            }
            print(f"   ✓ MAE: {self.results['Spatial Regression']['metrics']['MAE']:.4f}\n")
        except Exception as e:
            print(f"   ✗ Error: {e}\n")
        
        # 4. ARIMA
        if ARIMA_AVAILABLE:
            print("4️⃣  Training ARIMA...")
            try:
                arima = ARIMAModel(order=(2, 1, 2))
                arima.fit(X_train)
                arima_pred = arima.predict(len(X_test), X_test.shape[1])
                self.results['ARIMA'] = {
                    'predictions': arima_pred,
                    'metrics': self.evaluator.calculate_metrics(arima_pred, X_test)
                }
                print(f"   ✓ MAE: {self.results['ARIMA']['metrics']['MAE']:.4f}\n")
            except Exception as e:
                print(f"   ✗ Error: {e}\n")
        else:
            print("4️⃣  ARIMA - Not available (install statsmodels)\n")
        
        # 5. PROPHET
        if PROPHET_AVAILABLE:
            print("5️⃣  Training Prophet...")
            try:
                prophet = ProphetModel()
                prophet.fit(X_train)
                prophet_pred = prophet.predict(len(X_test), X_test.shape[1])
                self.results['Prophet'] = {
                    'predictions': prophet_pred,
                    'metrics': self.evaluator.calculate_metrics(prophet_pred, X_test)
                }
                print(f"   ✓ MAE: {self.results['Prophet']['metrics']['MAE']:.4f}\n")
            except Exception as e:
                print(f"   ✗ Error: {e}\n")
        else:
            print("5️⃣  Prophet - Not available (install prophet)\n")
        
        # 6. PG-STGNN
        print("6️⃣  Evaluating PG-STGNN...")
        try:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float).unsqueeze(-1)
            X_test_tensor = X_test_tensor.expand(-1, -1, 6)
            
            self.pgstgnn_model.eval()
            with torch.no_grad():
                pgstgnn_pred = self.pgstgnn_model(
                    self.graph_data['node_features'],
                    self.graph_data['edge_index'],
                    X_test_tensor,
                    {'tensor_features': {
                        'flow_chars': torch.zeros(X_test.shape[1], 3),
                        'vc_ratios': torch.zeros(X_test.shape[1], 2),
                        'queue_dynamics': torch.zeros(X_test.shape[1], 2)
                    }}
                )
            
            pgstgnn_pred_np = pgstgnn_pred.detach().cpu().numpy() * 100
            self.results['PG-STGNN'] = {
                'predictions': pgstgnn_pred_np,
                'metrics': self.evaluator.calculate_metrics(pgstgnn_pred_np, X_test)
            }
            print(f"   ✓ MAE: {self.results['PG-STGNN']['metrics']['MAE']:.4f}\n")
        except Exception as e:
            print(f"   ✗ Error: {e}\n")
        
        return self.results
    
    def print_comparison_table(self):
        """Print comparison table"""
        print("\n" + "=" * 80)
        print("RESULTS COMPARISON TABLE")
        print("=" * 80 + "\n")
        
        metrics_df = []
        for model_name, data in self.results.items():
            metrics = data['metrics']
            metrics_df.append({
                'Model': model_name,
                'MAE': f"{metrics['MAE']:.4f}",
                'RMSE': f"{metrics['RMSE']:.4f}",
                'MAPE': f"{metrics['MAPE']:.2f}%",
                'R²': f"{metrics['R2']:.4f}"
            })
        
        df = pd.DataFrame(metrics_df)
        print(df.to_string(index=False))
        print("\n" + "=" * 80)
        
        return df
    
    def visualize_comparison(self):
        """Visualize model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Traffic Prediction Models Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        maes = [self.results[m]['metrics']['MAE'] for m in models]
        rmses = [self.results[m]['metrics']['RMSE'] for m in models]
        mapes = [self.results[m]['metrics']['MAPE'] for m in models]
        r2s = [self.results[m]['metrics']['R2'] for m in models]
        
        # MAE
        ax = axes[0, 0]
        colors = ['#1f77b4' if 'PG-STGNN' not in m else '#d62728' for m in models]
        ax.barh(models, maes, color=colors)
        ax.set_xlabel('MAE', fontweight='bold')
        ax.set_title('Mean Absolute Error')
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(maes):
            ax.text(v, i, f' {v:.4f}', va='center')
        
        # RMSE
        ax = axes[0, 1]
        ax.barh(models, rmses, color=colors)
        ax.set_xlabel('RMSE', fontweight='bold')
        ax.set_title('Root Mean Square Error')
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(rmses):
            ax.text(v, i, f' {v:.4f}', va='center')
        
        # MAPE
        ax = axes[1, 0]
        ax.barh(models, mapes, color=colors)
        ax.set_xlabel('MAPE (%)', fontweight='bold')
        ax.set_title('Mean Absolute Percentage Error')
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(mapes):
            ax.text(v, i, f' {v:.2f}%', va='center')
        
        # R²
        ax = axes[1, 1]
        ax.barh(models, r2s, color=colors)
        ax.set_xlabel('R²', fontweight='bold')
        ax.set_title('R² Score')
        ax.set_xlim([-0.1, 1.1])
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(r2s):
            ax.text(v, i, f' {v:.4f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        best_mae = min(self.results.items(), key=lambda x: x[1]['metrics']['MAE'])
        best_rmse = min(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])
        best_r2 = max(self.results.items(), key=lambda x: x[1]['metrics']['R2'])
        
        print(f"\n🏆 Best MAE: {best_mae[0]} ({best_mae[1]['metrics']['MAE']:.4f})")
        print(f"🏆 Best RMSE: {best_rmse[0]} ({best_rmse[1]['metrics']['RMSE']:.4f})")
        print(f"🏆 Best R²: {best_r2[0]} ({best_r2[1]['metrics']['R2']:.4f})")
        
        if 'PG-STGNN' in self.results:
            pgstgnn_metrics = self.results['PG-STGNN']['metrics']
            print(f"\n📊 PG-STGNN Performance:")
            print(f"   MAE: {pgstgnn_metrics['MAE']:.4f}")
            print(f"   RMSE: {pgstgnn_metrics['RMSE']:.4f}")
            print(f"   MAPE: {pgstgnn_metrics['MAPE']:.2f}%")
            print(f"   R²: {pgstgnn_metrics['R2']:.4f}")
        
        print("\n" + "=" * 80)

# ==================== MAIN PG-STGNN SYSTEM ====================
class PGSTGNNSystem:
    """Main PG-STGNN System"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_builder = NetworkGraphBuilder(config)
        self.physics_embedder = PhysicsTrafficEmbedder(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.model = PGSTGNNModel(config)
        self.synthetic_generator = SyntheticTrafficGenerator(days=3)
        
    def execute_stepwise_framework(self):
        """Execute complete 4-step framework"""
        print("🚀 Starting PG-STGNN Framework with Synthetic Data")
        print("=" * 80)
        
        results = {}
        
        try:
            # Step 1: Generate synthetic data
            print("\n📊 Step 1: Generating synthetic traffic data...")
            intersection_data = self.synthetic_generator.generate_intersection_data()
            print(f"✅ Generated {len(intersection_data['time_series'])} time points with {len(intersection_data['approaches'])} approaches")
            
            # Step 2: Build network graph
            print("\n🔧 Step 2: Building Network Graph...")
            graph_data = self.graph_builder.construct_network_graph(intersection_data)
            results['graph_data'] = graph_data
            print(f"✅ Graph built with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
            
            # Step 3: Physics embedding
            print("\n🎯 Step 3: Embedding Physics Traffic Performance...")
            physics_features = self.physics_embedder.embed_traffic_performance(
                intersection_data, graph_data
            )
            results['physics_features'] = physics_features
            print("✅ Physics features embedded successfully")
            
            # Step 4: Prepare training data
            print("\n📈 Step 4: Preparing Training Data...")
            train_loader, val_loader = self.data_preprocessor.prepare_training_data(intersection_data)
            print(f"✅ Prepared {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")
            
            # Step 5: Train model
            if len(train_loader.dataset) > 0:
                print("\n🧠 Step 5: Training PG-STGNN Model...")
                training_info = self.model.train_model(
                    train_loader, val_loader, graph_data, physics_features, 
                    epochs=self.config.get('epochs', 30)
                )
                results['training_info'] = training_info
                print("✅ Model training completed")
            
        except Exception as e:
            print(f"❌ Error in framework execution: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results

# ==================== MAIN EXECUTION ====================
def main():
    """Main function with complete comparison"""
    
    config = {
        'spatial_model': 'GCN',
        'spatial_input_dim': 6,
        'hidden_dim': 32,
        'temporal_input_dim': 6,
        'learning_rate': 0.001,
        'batch_size': 16,
        'seq_len': 6,
        'epochs': 30,
        'dropout': 0.2
    }
    
    # Initialize and run the system
    traffic_system = PGSTGNNSystem(config)
    results = traffic_system.execute_stepwise_framework()
    
    print("\n" + "=" * 80)
    print("PG-STGNN FRAMEWORK EXECUTION COMPLETED")
    print("=" * 80)
    
    # Run model comparison
    if 'error' not in results and len(results) > 0:
        print("\n\n")
        print("⚡" * 40)
        print("STARTING MODEL COMPARISON")
        print("⚡" * 40)
        
        intersection_data = traffic_system.synthetic_generator.generate_intersection_data()
        graph_data = results['graph_data']
        physics_features = results['physics_features']
        
        comparison = ModelComparison(
            intersection_data,
            graph_data,
            traffic_system.model,
            config
        )
        
        comparison.run_comparison()
        comparison.print_comparison_table()
        comparison.print_summary()
        comparison.visualize_comparison()
        
        results['comparison'] = comparison.results

if __name__ == "__main__":
    main()